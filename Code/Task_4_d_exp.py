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

def Applying_Monte_Carlo(S0, r, sigma, T, M, K):
    
    global rng

    if rng is None:
        reset_rng()

    # Initialize the option price accumulator
    option_price_sum = 0

    # Simulate the price process using the weak Euler scheme by SDE
    for i in range(M):
        Z = rng.standard_normal()
        S = S0*math.exp((r-(1/2)*sigma**2)*T+(sigma*np.sqrt(T)*Z)) 
        # Calculate the option price 
        option_price = calculate_european_binary_call_option(S, K, r, T)

        # Accumulate the option price to monte carlo
        option_price_sum += option_price

    #Applying monte_carlo
    option_price_avg = option_price_sum / M

    return option_price_avg


S0 = 100   # initial stock price
K = 100   # strike price
sigma = 0.2  # volatility
r = 0.05   # risk-free interest rate
T = 1      # time to maturity

sample_size = 10
V0 = Black_Scholes_formula_for_binary_call_option(S0, K, sigma, r, T)
error1 = np.zeros(5)
relisations = [10**4, 10**5, 10**6, 10**7]

for M in relisations:
   start_time = time.time()
   Approximate_Price = Applying_Monte_Carlo(S0, r, sigma, T, M, K)
   print("Approx price at T = 0: %s when the M = %s" % (Approximate_Price, M))
   print("Price at T = 0: ", V0)
   print("Absolute error: ", abs(V0-Approximate_Price))
   end_time = time.time()
   print("Execution time: %s seconds " %(end_time - start_time))


# Study convergence Monte Carlo technique 
num_realizations = []
Monte_Carlo_error = []
mean_error_Eulur = []

for M in relisations:
    error = []
    start_time = time.time()
    for _ in range(sample_size):
        approx = Applying_Monte_Carlo(S0, r, sigma, T, M, K)
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
    
  
    print(f"Num. of realizations: {M}, Mean error: {sample_mean}, \n Variance: {sample_var}, Monte Carlo error: {standard_error} \n Execution time: {end_time - start_time} seconds")

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
log_N = np.log(np.array(relisations))
log_errors = np.log(np.abs(np.array(mean_error_Eulur)))
plt.plot(log_N, log_errors, 'o', label='Errors')
slope, intercept = np.polyfit(log_N, log_errors, 1)
plt.plot(log_N, slope*(log_N) + intercept, label='Convergence Rate')
plt.xlabel('Log(Time Step Size)')
plt.ylabel('Log(Error)')
plt.legend()
plt.show()
print('Convergence Rate:', abs(slope))
