import numpy as np
import math
import scipy.stats as stats
from Task_4_a import Applying_Monte_Carlo
import time

S0 = 110   # initial stock price
K = 100    # strike price
sigma = 0.02  # volatility
r = 0.05   # risk-free interest rate
T = 1      # time to maturity

# Calculate Black Scholes formula
def Black_Scholes_formula(S0, K, sigma, r, T):
  d1 = (math.log(S0/K) + (r + (sigma**2)/2)*T) / (sigma*math.sqrt(T))
  d2 = d1 - (sigma*math.sqrt(T))
  N_d1 = stats.norm.cdf(d1)
  N_d2 = stats.norm.cdf(d2)
  C_0 = S0*N_d1 - K*math.exp(-r*T)*N_d2
  return C_0

if __name__ == "__main__":
 
 N_range = [100, 200, 400, 600, 800]
 V0 = Black_Scholes_formula(S0, K, sigma, r, T)
 error = np.zeros(len(N_range))

 for j in range(len(N_range)):
   start_time = time.time()
   Approximate_Price, M = Applying_Monte_Carlo(S0, r, sigma, T, N_range[j], K)
   print("Approx price at T = 0: %s when the N %s" % (Approximate_Price, N_range[j]))
   print("Price at T = 0: ", V0)
   error[j] = V0-Approximate_Price
   print("Absolute error: ", error[j])
   end_time = time.time()
   print("Execution time: %s seconds when the realizations = %s " %(end_time - start_time, M))

pass
