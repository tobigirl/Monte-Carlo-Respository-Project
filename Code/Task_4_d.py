import numpy as np
import math
import scipy.stats as stats
import seaborn as sns
from scipy.stats import t
import matplotlib.pyplot as plt
from numpy.random import default_rng
import time

DEFAULT_SEED = 31

rng = None

def reset_rng(seed=DEFAULT_SEED):
    global rng
    rng = default_rng(seed)

# Calculate Black Scholes formula for binary call option
def Black_Scholes_formula_for_binary_call_option(S0, K, sigma, r, T):
  d1 = (r*T - math.log(K/S0) - ((sigma**2)/2)*T) / (sigma*math.sqrt(T))
  N_d1 = stats.norm.cdf(d1)
  C_0 = math.exp(-r*T)*N_d1
  return C_0

# Applying monte_carlo for binary call option
def calculate_european_binary_call_option(S_T, K, r, T):
  
   # Calculate option payoffs
   if S_T > K:
    payoff = 1
   else:
    payoff = 0 
   # Calculate option prices
   option_price = np.exp(-r*T) * payoff

   return option_price

def Applying_Monte_Carlo(S0, r, sigma, T, N, K):
    """
    N -- Number of time points in a single path.
    M -- Number of realizations/path to generate.
    K -- Strike price of the option.
    """
    global rng

    if rng is None:
        reset_rng()

    #M = 100*N  # For study of Monte Carlo convergence
    #N = 5  # For study of Monte Carlo convergence
    # Time step
    dt = T / N
    M = 500000 # For study of discretization convergence

    # Initialize the option price accumulator
    option_price_sum = 0

    # Simulate the price process using the weak Euler scheme by SDE
    for i in range(M):
        S = S0
        for j in range(1, N+1):
          Z = rng.standard_normal()
          S = S + r*S*dt + sigma*S*np.sqrt(dt)*Z 

        # Calculate the option price 
        option_price = calculate_european_binary_call_option(S, K, r, T)

        # Accumulate the option price to monte carlo
        option_price_sum += option_price

    #Applying monte_carlo
    option_price_avg = option_price_sum / M

    return option_price_avg, M


S0 = 100   # initial stock price
K = 100   # strike price
sigma = 0.2  # volatility
r = 0.05   # risk-free interest rate
T = 1      # time to maturity

sample_size = 2
#N_range = [10,40,160,640] # For study of Monte Carlo error
N_range = [8, 16, 32, 64, 128, 256] # For study of discretization error


V0 = Black_Scholes_formula_for_binary_call_option(S0, K, sigma, r, T)
error1 = np.zeros(len(N_range))

for N in N_range:
   start_time = time.time()
   Approximate_Price, M = Applying_Monte_Carlo(S0, r, sigma, T, N, K)
   print("Approx price at T = 0: %s when the N = %s" % (Approximate_Price, N))
   print("Price at T = 0: ", V0)
   print("Absolute error: ", abs(V0-Approximate_Price))
   end_time = time.time()
   print("Execution time: %s seconds when the realizations = %s " %(end_time - start_time, M))


# Study convergence Monte Carlo technique
num_realizations = []
Monte_Carlo_error = []
mean_error_Eulur = []

for N in N_range:
    error = []
    start_time = time.time()
    for _ in range(sample_size):
        approx, M = Applying_Monte_Carlo(S0, r, sigma, T, N, K)
        error.append(V0 - approx)

    sample_mean = np.mean(error)
    mean_error_Eulur.append(sample_mean)
    #Standard deviation
    #std_error = np.std(error, ddof=1)
    #std_error1 = std_error / np.sqrt(sample_size)
    # the Variance
    sample_var = np.var(error, ddof=1)
    #error of Monte Carlo technique 
    standard_error = 1.96 * np.sqrt(sample_var/M)
    Monte_Carlo_error.append(standard_error)
    num_realizations.append(M)
    end_time = time.time()
    
  
    print(f"Num. of realizations: {M}, Num. of time Step: {N}, Mean error: {sample_mean}, \n Variance: {sample_var}, Monte Carlo error: {standard_error} \n Execution time: {end_time - start_time} seconds")

# Calculate confidence intervals (commented out for log plot)
#lower_bounds = np.log([max(el,0) for el in np.array(mean_error_Eulur) - np.array(Monte_Carlo_error)])
#upper_bounds = np.log(mean_error_Eulur) + np.array(Monte_Carlo_error)
lower_bounds = np.array(mean_error_Eulur) - np.array(Monte_Carlo_error)
upper_bounds = np.array(mean_error_Eulur) + np.array(Monte_Carlo_error)

plt.figure()
# Plot results to see as the realizations increase the 
plt.errorbar(num_realizations, mean_error_Eulur, yerr=np.array(Monte_Carlo_error), fmt='o-', capsize=5)
plt.fill_between(num_realizations, lower_bounds, upper_bounds, alpha=0.2)
plt.xlabel('Number of Realizations')
plt.ylabel('Error')
plt.title('Error of Monte Carlo Simulation with 95% Confidence Interval')

###################################
###################################
#Study the convergence of the Euler scheme
# plot to get the convergence rate as we increase the time step 
# for the numerical solution of weak Euler scheme
plt.figure()
log_N = np.log(np.array(N_range))
log_errors = np.log([max(el,0) for el in np.abs(np.array(mean_error_Eulur))-Monte_Carlo_error])
plt.errorbar(log_N, log_errors, yerr=np.array(Monte_Carlo_error), fmt='o-', capsize=5)
slope, intercept = np.polyfit(log_N, log_errors, 1)
plt.plot(log_N, slope*(log_N) + intercept, label='Convergence Rate')
plt.xlabel('Log(Time Step Size)')
plt.ylabel('Log(Error)')
plt.legend()
plt.show()
print('Convergence Rate:', abs(slope))

# Plot for convergence of Monte-Carlo error
plt.figure()
plt.title("Convergence of Monte-Carlo error")
log_M = np.log(np.array(num_realizations))
log_errors = np.log(Monte_Carlo_error)
plt.plot(log_M, log_errors, 'o', label='Errors')
slope, intercept = np.polyfit(log_M, log_errors, 1)
plt.plot(log_M, slope*(log_M) + intercept, label='Convergence Rate')
plt.xlabel('Log(Number of Realizations)')
plt.ylabel('Log(Error)')
plt.legend()
print('MC Convergence Rate:', slope)
plt.show()
