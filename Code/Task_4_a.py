import numpy as np
from numpy.random import default_rng

DEFAULT_SEED = 31

rng = None


def reset_rng(seed=DEFAULT_SEED):
    global rng
    rng = default_rng(seed)

def calculate_european_call_option_price(S_T, K, r, T):
    # Calculate option payoffs
    payoff = np.maximum(S_T - K, 0)

    # Calculate option prices
    option_price = np.exp(-r*T) * payoff

    return np.mean(option_price)

def Applying_Monte_Carlo(S0, r, sigma, T, N, K):
    """
    N -- Number of time points in a single path.
    M -- Number of realizations/path to generate.
    K -- Strike price of the option.
    """
    
    global rng

    if rng is None:
        reset_rng()
    

    #M = N # To study Monte Carlo error
    M = int(10**7/N) # To study discretization error (alternatively use 500000)
    #N = int(320000/N) # To study Monte Carlo error (alternatively use 5)
    # Time step
    dt = T/N
    # Initialize the option price accumulator
    option_price_sum = 0

    # Simulate the price process using the weak Euler scheme by SDE
    for i in range(M):
        S = S0
        #for j in range(1, N+1):
        Z = rng.standard_normal()
        #    S = S + r*S*dt + sigma*S*np.sqrt(dt)*Z
        S = S*np.exp((r-0.5*sigma**2)*T+(sigma*np.sqrt(T)*Z))

        # Calculate the option price 
        option_price = calculate_european_call_option_price(S, K, r, T)

        # Accumulate the option price to monte carlo
        option_price_sum += option_price

    #Applying monte_carlo
    option_price_avg = option_price_sum / M

    return option_price_avg, M
