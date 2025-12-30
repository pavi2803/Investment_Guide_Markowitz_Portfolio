import numpy as np
import yfinance as yf
from cvxopt import matrix, solvers
import pandas as pd
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt


# def download_stock_data(tickers, start_date, end_date):
#     data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
#     st.write(data)
#     return data

def download_stock_data(tickers, start_date, end_date):
    raw_data = yf.download(tickers, start=start_date, end=end_date)


    if raw_data.empty:
        st.error("No data found. Please check the ticker symbols and date range.")
        st.stop()

    # Check for MultiIndex columns
    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            adj_close = raw_data['Close']
        except KeyError:
            st.error("'Adj Close' not found in the data. Please try different tickers.")
            st.stop()
    else:
        if 'Close' in raw_data.columns:
            adj_close = raw_data[['Close']]
        else:
            st.error("'Closing price not found in the data.")
            st.stop()

    st.write("Adjusted Close data Overview:", adj_close.head())
    return adj_close



def calculate_returns_and_covariance(data):
    # Remove any remaining NaN values
    clean_data = data.dropna()
    
    if clean_data.empty or len(clean_data) < 2:
        raise ValueError("Insufficient data after removing NaN values")
    
    returns = clean_data.pct_change().dropna().mean() * 252
    cov_matrix = clean_data.pct_change().dropna().cov() * 252
    
    # Check for NaN or infinite values
    if returns.isna().any() or np.isinf(returns).any():
        raise ValueError("Returns contain NaN or infinite values")
    if cov_matrix.isna().any().any() or np.isinf(cov_matrix).any().any():
        raise ValueError("Covariance matrix contains NaN or infinite values")
    
    return returns, cov_matrix

def optimize_portfolio(returns, cov_matrix):
    num_assets = len(returns)
    returns = np.asmatrix(returns)
    cov_matrix = np.asmatrix(cov_matrix)
    
    # Add small regularization to covariance matrix to ensure positive definiteness
    cov_matrix = cov_matrix + np.eye(num_assets) * 1e-8
    
    # Define the parameters for the quadratic optimization problem
    # Minimize: 1/2 x^T Q x
    # Subject to: Gx <= h, Ax = b
    Q = matrix(cov_matrix)
    c = matrix(np.zeros((num_assets, 1)))
    
    # G matrix: [weights >= 0] -> [-I * weights <= 0]
    G = matrix(-np.identity(num_assets))
    h = matrix(np.zeros((num_assets, 1)))
    
    # A matrix: sum of weights = 1
    A = matrix(1.0, (1, num_assets))
    b = matrix(1.0)
    
    # Solve with suppressed output
    solvers.options['show_progress'] = False
    
    try:
        sol = solvers.qp(Q, c, G, h, A, b)
        if sol['status'] != 'optimal':
            raise ValueError(f"Optimization failed with status: {sol['status']}")
        optimal_weights = np.array(sol['x'])
        return optimal_weights
    except Exception as e:
        raise ValueError(f"Optimization error: {str(e)}")


if __name__ == "__main__":
    # Input parameters
    
    st.header("Portfolio Optimization For Stock Returns")
    dropdown_yf = ['AAPL', 'MSFT', 'AMZN', 'V', 'B', 'CMG','OLVI','LYFT','NVDA','NKE','TSLA', 'GOOG','JNJ','WMT','MCD','SBUX','ABBV','DIS','NFLX','META','UNH']
    selected_category = st.multiselect("Choose the stocks which you want to invest in ", dropdown_yf)
    
    
    if(selected_category!=[] and len(selected_category)>2):
        start_date = st.date_input('Select a start date')
        end_date = st.date_input('Select an end date')
            
        # Display the button
        if st.button("Get Stock Insights"):
            if start_date and end_date:
                st.write('You selected:')
                st.write('Start date:', start_date)
                st.write('End date:', end_date)
 
                
                stock_data = download_stock_data(selected_category, start_date, end_date)
        
                # Check for stocks with NaN values
                nan_counts = stock_data.isna().sum()
                stocks_with_nan = nan_counts[nan_counts > 0]
                
                if not stocks_with_nan.empty:
                    st.error("⚠️ One or more stocks have missing data and cannot be used for optimization:")
                    for stock, count in stocks_with_nan.items():
                        st.write(f"- **{stock}**: {count} missing values out of {len(stock_data)} days")
                    st.info("Please remove these stocks from your selection or choose a different date range.")
                    st.stop()
                
                adj_close = stock_data.dropna(axis=1, how='all')
                if adj_close.shape[1] < 2:
                    st.error("At least two valid tickers are required with available data.")
                    st.stop()

                    # Calculate returns and covariance matrix
                returns, cov_matrix = calculate_returns_and_covariance(stock_data)
                    
                    # Optimize portfolio
                optimal_weights = optimize_portfolio(returns, cov_matrix)
    
                st.write("Optimal proportion split for your investment:")
                for i in range(0,len(selected_category)):
                    st.write(selected_category[i])
                    st.text(f"{optimal_weights[i]}")
            
                weights=optimal_weights.tolist()
                
                simple_weights = [item[0] for item in weights]
                    
            
                fig, ax = plt.subplots()
                ax.bar(selected_category, simple_weights, color='skyblue')
                
                # Add title and labels
                ax.set_title('Investment Split')
                ax.set_xlabel('Stock')
                ax.set_ylabel('Suggested Proportions of Investing')
                
                # Display the plot in Streamlit
                st.pyplot(fig)

        else:
            st.write('No date selected.')
            
        
    else:
        st.text("Please select atleast 2 options to invest (This helps to split your investment)")
    # Print results
    
    
####################################################################
