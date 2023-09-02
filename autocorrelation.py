from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot

# Function to compute autocorrelation using the library functions [shift and autocorrelation]
# Calculates and plot autocorrelation values for various lag periods using the Pandas shift and correlation functions. 
def autocorrelation(dataset, lags):
    ac = {}
    for lag in lags:  # Three different lagging values
        # Shift the new data from the original data by the lag value
        dataframe = pd.concat([pd.DataFrame(dataset.values).shift(lag), pd.DataFrame(dataset.values)], axis=1)
        dataframe.columns = ['t-%i' % lag, 't']
        # Store the correlation value (for plotting)
        ac[lag] = dataframe.corr()

    # Plot the autocorrelation signal
    x = autocorrelation_plot(dataset, color='red', alpha=0.3)
    x.plot()

    # Plot the autocorrelation values
    for lag in lags:
        plt.plot(lag, ac[lag]['t-%i' % lag]['t'], 'o', label='shift_%i' % lag)
        plt.title('Autocorrelation for the lag periods')
        plt.gca().grid(which='both')
        plt.legend()
    plt.show()
