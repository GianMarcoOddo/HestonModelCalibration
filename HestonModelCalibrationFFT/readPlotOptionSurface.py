
import numpy as np  # importing the numpy library for numerical computations
import matplotlib.pyplot as plt  # importing the main plotting library from matplotlib
from matplotlib import cm  # importing color maps from matplotlib for visual enhancement
from scipy import interpolate  # importing interpolation functions from scipy to fit data points

# ----------------------------------------------------------------------------------------------------------------------------

# defining a function to read option data and plot it
def readNPlot(df, ticker, figsize=(8,6)):
    
    # extracting unique strike values from the DataFrame and sorting them
    strikes = np.sort(df.Strike.unique())  
    
    # extracting unique maturity days from the DataFrame and sorting them
    maturities = np.sort(df.Maturity_days.unique())  
    
    # creating a meshgrid for strikes and maturities, essential for 3D plotting
    X, Y = np.meshgrid(strikes, maturities)  
    
    # initializing an empty array to store option prices
    callPrices = np.empty([len(maturities), len(strikes)])  
    
    # looping through each maturity to interpolate option prices
    for i in range(len(maturities)):  

        # selecting strike prices for the given maturity
        s = df[df.Maturity_days == maturities[i]]['Strike']  
        
        # selecting corresponding option prices for the given maturity
        price = df[df.Maturity_days == maturities[i]]['Mid']  
        
        # using linear interpolation to estimate option prices for all strikes
        f = interpolate.interp1d(s, price, kind='linear', bounds_error=False, fill_value="extrapolate")  
        
        # storing interpolated prices into the callPrices array
        callPrices[i, :] = f(strikes)  

    # creating a new figure to plot the surface
    fig = plt.figure(figsize = figsize)
    
    # adding a 3D subplot to the figure
    ax = fig.add_subplot(111, projection='3d')
    
    # plotting the surface using the meshgrid and call prices
    ax.plot_surface(X, Y, callPrices, cmap=cm.coolwarm)
    
    # setting the y-axis label as 'Maturity (days)'
    ax.set_ylabel('Maturity (days)') 
    
    # setting the x-axis label as 'Strike'
    ax.set_xlabel('Strike') 
    
    # setting the z-axis label as 'C(K, T)'
    ax.set_zlabel('C(K, T)')
    
    # setting the title of the plot to include the ticker and describing it as an option surface
    ax.set_title(ticker + ' Calls - Option Surface')

    # returning maturities, strikes, callPrices, and the figure for future use
    return maturities, strikes, callPrices, fig

# ----------------------------------------------------------------------------------------------------------------------------











