import numpy as np
import yfinance as yf
from cvxopt import matrix, solvers
import pandas as pd
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt

start_date = '2025-07-01'
end_date = '2025-07-04'

single_raw = yf.download('AAPL', start=start_date, end=end_date)

print(single_raw.columns)