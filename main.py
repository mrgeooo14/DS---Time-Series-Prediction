# ~~~~~~~TP3 - Data Science - Clustering Approaches and Autoregression~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Genis Skura~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from autocorrelation import autocorrelation

from plotting_functions import plot_365, plot_dataset

sns.set()

## Calculates and append Simple Moving Averages (SMA) of various time periods (weekly, monthly, and yearly) to our dataset
## SMA-7 [Weekly], SMA-30 [Monthly], SMA-365 [Yearly]
def create_feature(dataset, K):
    for k in K:
        title = 'SMA_' + k.__str__()
        dataset[title] = dataset.iloc[:, 1].rolling(window=k).mean()
    return dataset


# Next Task - Original Values - 365
# We calculate the yearly deviation from the original value and the 30-day moving average of this yearly deviation
def value_365(dataset):
    dataset['value-365'] = dataset.apply(lambda row: row[1] - row['SMA_365'], axis=1)
    dataset['SMA_30(value-365)'] = dataset['value-365'].rolling(window=30).mean()
    dataset['value-365-30'] = dataset.apply(lambda row: row['value-365'] - row['SMA_30(value-365)'], axis=1)
    return dataset

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~2.3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Return the predicted value [scalar] as according to the coefficients
def predicted_value(y, a, sigma):
    return np.dot(y, a) + sigma


#### Function constructed as required on exercise 2.5
#### It leverages autoregressive modeling to make future predictions in a time series, and it returns a DataFrame containing these predictions
def predict_time_series(time_series, deep, n_pred, n_train):
    # Autoregression Formula and the Yule Walker coefficients rho and sigma
    rho, sigma = sm.regression.yule_walker(time_series[:n_train], order=deep, method="mle")
    # List to hold prediction values
    p = list(time_series[n_train-deep:n_train].values)

    for i in range(n_pred):
        # Predictor depends on the parameters of Autoregression
        p.append(predicted_value(p[-deep:], rho, sigma))
    p = p[-n_pred:]

    # Concatinate the previously computed prediction in our dataset
    # Add a new column 'Temp_Pred' holding the prediction values
    prediction = pd.concat([pd.DataFrame(time_series[n_train:n_train + n_pred].index.values, columns=[time_series.index.name]),
                 pd.DataFrame(time_series[n_train:n_train + n_pred].values, columns=[time_series.name]),
                 pd.DataFrame(p, columns=['Temp_Pred'])], axis=1)
    return prediction


if __name__ == "__main__":

    print('Program Start... \n')
# Read the dataframe from the .csv files and separate them in three datasets
# Total Energy Consumption, Wind+Solar Consumption, Temperature change in a timeline
# They represent daily electrical consumption of an area for a timeline of 11 years (2006-2017) and the temperature change over a timeline of 9 years (1981 - 1990)
    consumption = pd.read_csv('data_cons.csv', header = 0, usecols = [0, 1])
    wind_solar = pd.read_csv('data_cons.csv', header = 0, usecols = [0, 4])
    temperature = pd.read_csv('data_mintemp.csv', header = 0, usecols = [0, 1])
    temperature2 = pd.read_csv('data_mintemp.csv', header = 0, index_col = 0)
    print('Consumption Data Sample: ')
    print(consumption)
    print('')

# Weekly, Monthly, Yearly and lagging value coefficients
    time_units = [7, 30, 365]
    time_units_lag = [365, 182, 91] 

    
# Add the moving average features to our three datasets
    consumption2 = create_feature(consumption, time_units)
    ws2 = create_feature(wind_solar, time_units)
    temps2 = create_feature(temperature, time_units)

# Create a 1x3 grid of SMA subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Adjust figsize as needed
    plot_dataset(consumption2, axs[0])
    plot_dataset(ws2, axs[1])
    plot_dataset(temps2, axs[2])
    fig.suptitle('Original Dataset Timeline')
    for ax in axs:
        ax.legend(loc='upper left')
    plt.show()

# Add the three columns correlated with value_365 to our datasets
# To perform the yearly deviation from the original value (in terms of SMA)
    consumption2 = value_365(consumption2)
    ws2 = value_365(ws2)
    temps2 = value_365(temps2)
    
# Create a 1x3 grid of yearly deviation subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Adjust figsize as needed
    plot_365(consumption2, axs[0])
    plot_365(ws2, axs[1])
    plot_365(temps2, axs[2])
    fig.suptitle('Simple Moving Average Yearly (365) Deviations')
    for ax in axs:
        ax.legend(loc='upper left')
    plt.show()
    
# Perform autocorrelation of the lag value
    autocorrelation(temperature2, time_units_lag)
    
# ~~~~~~~~~~~~~~~Predicted Values~~~~~~~~~~~~~~~~~~~~~~~~
# Get a and sigma values from library built in function of regression
# The AR model has an order of 4, meaning it will consider the previous 4 time steps to predict the next one.
# The method used for parameter estimation is Maximum Likelihood Estimation (MLE), a common approach to estimate AR model parameters.
    a, sigma = sm.regression.yule_walker(temperature2['Temp'][:-365], order = 4, method = "mle")


# Create a 3x1 grid of autoregression subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Adjust figsize as needed
    fig.suptitle('Autoregressive Model (AR) Temperature Prediction')
    for i, k in enumerate(time_units):
        ax = axs[i]
        predict = predict_time_series(temperature2['Temp'], k, 365, len(temperature2['Temp']) - 365)
        predict.plot(alpha=0.66, ax=ax)
        mse = ((predict['Temp'] - predict['Temp_Pred']) ** 2).mean()  # Mean Square Error
        ax.set_title('deep: %i MSE: %.2f' % (k, mse))
    plt.tight_layout()
    plt.show()