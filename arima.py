# import pandas as pd

# # Load the dataset to inspect the column names
# uploaded_file = 'BBCA.JK.csv'  # Replace with the actual path to your CSV file
# df = pd.read_csv(uploaded_file)
# print(df.head())
# print(df.columns)


import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime

# Title of the app
st.title('ARIMA Model Walk-Forward Validation')

# Function to parse dates
def parser(x):
    try:
        return datetime.strptime(x, '%Y-%m-%d')  # Adjust the format based on your CSV file
    except ValueError as e:
        st.error(f"Date parsing error: {e}")
        return None

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read dataset
    try:
        series = pd.read_csv(uploaded_file, header=0, index_col=0, parse_dates=True, date_parser=parser)
        series.index = series.index.to_period('M')
        
        st.write("Dataset Columns:")
        st.write(series.columns.tolist())
        
        # Select the specific column for ARIMA
        column_name = st.selectbox("Select the column for ARIMA modeling:", series.columns.tolist())
        
        series = series[column_name]

        # Split into train and test sets
        X = series.values
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()

        # Walk-forward validation
        for t in range(len(test)):
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast(steps=1)
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            st.write('predicted=%f, expected=%f' % (yhat, obs))

        # Evaluate forecasts
        rmse = sqrt(mean_squared_error(test, predictions))
        st.write('Test RMSE: %.3f' % rmse)

        # Plot forecasts against actual outcomes
        fig, ax = plt.subplots()
        ax.plot(test, label='Actual')
        ax.plot(predictions, color='red', label='Predicted')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.write("Please upload a CSV file to proceed.")

