import pandas as pd

# Function to plot the original values & the moving averages
def plot_dataset(dataset, ax):
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    time = dataset['Date']
    ax.plot(time, dataset[dataset.columns[1]], label=dataset.columns[1], color='black', alpha=0.66)
    ax.plot(time, dataset['SMA_7'], label='SMA 7 Days', color='yellow', alpha=0.5)
    ax.plot(time, dataset['SMA_30'], label='SMA 30 Days', color='green', alpha=0.5)
    ax.plot(time, dataset['SMA_365'], label='SMA 1 Year', color='red', linewidth=2)
    ax.set_ylabel(dataset.columns[1])


# Plot the last three added columns
def plot_365(dataset, ax):
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    time = dataset['Date']
    ax.plot(time, dataset['value-365'], label='values-365', color='black', alpha=0.66)
    ax.plot(time, dataset['SMA_30(value-365)'], label='values-365 (SMA 30)', color='yellow', alpha=0.5)
    ax.plot(time, dataset['value-365-30'], label='(values-365) - (SMA 30)', color='green', alpha=0.5)
    ax.set_ylabel('value-365')