import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.stats import t
from Task_4_a import Applying_Monte_Carlo
from Task_4_b import Black_Scholes_formula
import time


S0 = 110   # initial stock price
K = 100    # strike price
sigma = 0.02  # volatility
r = 0.05   # risk-free interest rate
T = 1      # time to maturity

#sample_size = 100 # To study Monte-Carlo error
sample_size = 1 # To study discretization error
#N_range = [1000,4000,16000,64000] # To study Monte-Carlo error
N_range = [10,20,30,40,50,60] # To study discretization error
V0 = Black_Scholes_formula(S0, K, sigma, r, T)

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
    
    plt.title(f"The Error using {M} realizations of gbm using weak euler \n for european call option")
    sns.histplot(error, kde=True)
plt.show()

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

# plot for convergence rate of Monte-Carlo error
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
print('Convergence Rate:', slope)

###################################
###################################
#Study the convergence of the Euler scheme
# plot to get the convergence rate as we increase the time step 
# for the numerical solution of weak Euler scheme
plt.figure()
plt.title("Convergence of discretization error")
log_N = np.log(np.array(N_range))
log_errors = np.log([max(el,0) for el in np.abs(np.array(mean_error_Eulur))-Monte_Carlo_error])
plt.errorbar(log_N, log_errors, yerr=np.array(Monte_Carlo_error), fmt='o-', capsize=5)
slope, intercept = np.polyfit(log_N, log_errors, 1)
plt.plot(log_N, slope*(log_N) + intercept, label='Convergence Rate')
plt.xlabel('Log(Time Step Size)')
plt.ylabel('Log(Error)')
plt.legend()
plt.show()
print('Convergence Rate:', slope)
