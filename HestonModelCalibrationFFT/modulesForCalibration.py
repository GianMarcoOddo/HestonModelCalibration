
import numpy as np  # importing the numpy library for numerical operations
import cmath  # importing the complex math library for operations on complex numbers
import math  # importing the math library for mathematical operations

# ----------------------------------------------------------------------------------------------------------------------------

# defining a custom range function with start, finish, and increment as arguments
def myRange(start, finish, increment):
    myZero = 1e-17  # defining a very small number close to zero
    while (start <= finish+myZero):  # looping until start is less than or equal to finish plus the small number
        yield start  # yielding the current start value
        start += increment  # incrementing the start value
        
# ----------------------------------------------------------------------------------------------------------------------------

# defining a function to perform periodic linear extension mapping
def paramMapping(x, c, d):

    if ((x>=c) & (x<=d)):  # checking if x is between c and d
        y = x  # assigning x to y
    else:
        range = d-c  # calculating the difference between d and c
        n = math.floor((x-c)/range)  # calculating the floor value of the division
        if (n%2 == 0):  # checking if n is even
            y = x - n*range  # calculating y based on n and range
        else:
            y = d + n*range - (x-c)  # calculating y based on d, n, range and x
            
    return y  # returning the calculated y value

# ----------------------------------------------------------------------------------------------------------------------------

# defining a function to calculate the error value
def eValue(params, *args):
    
    marketPrices = args[0]  # extracting market prices from args
    maturities = args[1]  # extracting maturities from args
    strikes = args[2]  # extracting strikes from args
    r = args[3]  # extracting r value from args
    q = args[4]  # extracting q value from args
    S0 = args[5]  # extracting initial stock price from args
    alpha = args[6]  # extracting alpha value from args
    eta = args[7]  # extracting eta value from args
    n = args[8]  # extracting n value from args
    model = args[9]  # extracting model type from args

    lenT = len(maturities)  # getting the length of maturities
    lenK = len(strikes)  # getting the length of strikes
    
    modelPrices = np.zeros((lenT, lenK))  # initializing an array of zeros for model prices

    count = 0  # initializing a counter
    mae = 0  # initializing the mean absolute error
    for i in range(lenT):  # looping through each maturity
        for j in range(lenK):  # looping through each strike
            count  = count+1  # incrementing the counter
            T = maturities[i]  # getting the current maturity
            K = strikes[j]  # getting the current strike
            [km, cT_km] = genericFFT(params, S0, K, r, q, T, alpha, eta, n, model)  # computing km and cT_km using the genericFFT function
            modelPrices[i,j] = cT_km[0]  # assigning the calculated model price to the matrix
            tmp = marketPrices[i,j]-modelPrices[i,j]  # computing the difference between market and model prices
            mae += tmp**2  # accumulating the square of differences
    
    rmse = math.sqrt(mae/count)  # calculating the root mean square error
    return rmse  # returning the calculated error

# ----------------------------------------------------------------------------------------------------------------------------

# defining a function to compute the characteristic function of a given model
def generic_CF(u, params, S0, r, q, T, model):
    # logic for Geometric Brownian Motion (GBM) model
    if (model == 'GBM'):
        sig = params[0]  # extracting sigma from parameters
        mu = np.log(S0) + (r-q-sig**2/2)*T  # calculating mu value
        a = sig*np.sqrt(T)  # calculating a value
        phi = np.exp(1j*mu*u-(a*u)**2/2)  # calculating the characteristic function for GBM
        
    # logic for Heston model
    elif(model == 'Heston'):
        # extracting parameters for the Heston model
        kappa  = params[0]
        theta  = params[1]
        sigma  = params[2]
        rho    = params[3]
        v0     = params[4]
        
        # mapping parameters to specific ranges
        kappa = paramMapping(kappa,0.1, 20)
        theta = paramMapping(theta,0.001, 0.4)
        sigma = paramMapping(sigma,0.01, 0.6)
        rho   = paramMapping(rho  ,-1.0, 1.0)
        v0    = paramMapping(v0   ,0.005, 0.25)
        
        # calculations for Heston model characteristic function
        tmp = (kappa-1j*rho*sigma*u)
        g = np.sqrt((sigma**2)*(u**2+1j*u)+tmp**2)
        pow1 = 2*kappa*theta/(sigma**2)
        numer1 = (kappa*theta*T*tmp)/(sigma**2) + 1j*u*T*r + 1j*u*math.log(S0)
        log_denum1 = pow1 * np.log(np.cosh(g*T/2)+(tmp/g)*np.sinh(g*T/2))
        tmp2 = ((u*u+1j*u)*v0)/(g/np.tanh(g*T/2)+tmp)
        log_phi = numer1 - log_denum1 - tmp2
        phi = np.exp(log_phi)
        
    # logic for Variance Gamma (VG) model
    elif (model == 'VG'):
        # extracting parameters for the VG model
        sigma  = params[0]
        nu     = params[1]
        theta  = params[2]
        
        # calculations for VG model characteristic function
        if (nu == 0):
            mu = math.log(S0) + (r-q - theta -0.5*sigma**2)*T
            phi  = math.exp(1j*u*mu) * math.exp((1j*theta*u-0.5*sigma**2*u**2)*T)
        else:
            mu  = math.log(S0) + (r-q + math.log(1-theta*nu-0.5*sigma**2*nu)/nu)*T
            phi = cmath.exp(1j*u*mu)*((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)**(-T/nu))

    return phi  # returning the calculated characteristic function

# ----------------------------------------------------------------------------------------------------------------------------

# defining a function to compute option prices using the Fast Fourier Transform (FFT) method
def genericFFT(params, S0, K, r, q, T, alpha, eta, n, model):
    
    N = 2**n  # calculating the size of the FFT
    
    lda = (2*np.pi/N)/eta  # calculating the step-size in log strike space
    
    beta = np.log(K)  # choosing beta as the logarithm of strike price
    
    km = np.zeros((N))  # initializing an array of zeros for km values
    xX = np.zeros((N))  # initializing an array of zeros for x values
    
    df = math.exp(-r*T)  # calculating the discount factor
    
    nuJ = np.arange(N)*eta  # creating an array of nu values multiplied by eta

    psi_nuJ = generic_CF(nuJ-(alpha+1)*1j, params, S0, r, q, T, model)/((alpha + 1j*nuJ)*(alpha+1+1j*nuJ))  # calculating psi_nu values using the characteristic function
    
    # loop to calculate x values for FFT
    for j in range(N):  
        km[j] = beta+j*lda  # calculating km values
        if j == 0:
            wJ = (eta/2)  # adjusting the weight for the first iteration
        else:
            wJ = eta  # using eta as the weight for subsequent iterations
        xX[j] = cmath.exp(-1j*beta*nuJ[j])*df*psi_nuJ[j]*wJ  # calculating x values for FFT
     
    yY = np.fft.fft(xX)  # performing the FFT on x values
    cT_km = np.zeros((N))  # initializing an array of zeros for cT_km values

    # loop to calculate option prices using the FFT results
    for i in range(N):
        multiplier = math.exp(-alpha*km[i])/math.pi  # calculating the multiplier for option prices
        cT_km[i] = multiplier*np.real(yY[i])  # calculating option prices
    
    return km, cT_km  # returning km values and option prices

# ----------------------------------------------------------------------------------------------------------------------------

# importing the Thread class from the threading module
import threading
# importing the sys module for standard I/O operations
import sys
# importing the time module to use sleep function
import time

# defining the DotPrinter class, inheriting from Thread
class DotPrinter(threading.Thread):
    # initializing the DotPrinter class
    def __init__(self, process_name="Minimization Process"):
        # calling the parent class constructor
        super(DotPrinter, self).__init__()
        
        # initializing running state as False
        self.running = False 
        # initializing direction for counting dots
        self.direction = 1 
        # initializing count of dots to 0
        self.dot_count = 0
        # initializing the name of the process
        self.process_name = process_name

    # defining the start method to activate the thread
    def start(self):
        # setting the running state to True
        self.running = True 
        # calling the parent class's start method to activate the thread
        super(DotPrinter, self).start()

    # defining the main method that will run in the thread
    def run(self):
        # loop to run while the running state is True
        while self.running:
            # printing dots and overwriting the line using carriage return (\r)
            print(f"\r{self.process_name} " + "." * self.dot_count, end=" " * 90)
            # flushing the output buffer to display changes immediately
            sys.stdout.flush()
            # changing direction if dot_count reaches 10
            if self.dot_count == 10:
                self.direction = -1
            # changing direction if dot_count reaches 0
            elif self.dot_count == 0:
                self.direction = 1
            # incrementing or decrementing the dot count based on direction
            self.dot_count += self.direction
            # pausing for 0.2 seconds before the next iteration
            time.sleep(0.2)

    # defining the method to stop the thread
    def stop(self):
        # setting the running state to False to exit the loop in run method
        self.running = False

# ----------------------------------------------------------------------------------------------------------------------------
