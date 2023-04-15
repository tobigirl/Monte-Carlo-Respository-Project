import time
import numpy as np
from numpy.typing import NDArray
from numpy.random import default_rng
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

# Style matplotlib
plt.style.use('seaborn-v0_8-darkgrid')

# Initialize randon number generator.
rng = default_rng(31)

# Define base class for the options
class Option(ABC):
    def __init__(self, S0: float, K: float, sigma: float, r: float, T: float):
        self.S0 = S0 # Stock price
        self.K = K # Strike price
        self.sigma = sigma # Volatility
        self.r = r # Risk-free interest rate
        self.T = T # Time to maturity in years

    @abstractmethod
    def black_scholes_formula(self) -> float:
        pass

    @abstractmethod
    def discounted_payoff(self, ST: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    def forward_stock_price(self) -> float:
        return np.exp(self.r*self.T)*self.S0

    def discounted_price(self, price: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.exp(-self.r*self.T)*price

# Define class for vainilla european call options 
class EuropeanCallOption(Option):
    # Initializer.
    def __init__(self, S0, K, sigma, r, T):
        # Call base constrctor initializer
        super().__init__(S0, K, sigma, r, T)

    # Method for computing vainilla european call black scholes formula.
    def black_scholes_formula(self):
        d1 = (np.log(self.S0/self.K) + (self.r + (self.sigma**2)/2)*T) 
        d1 /= (self.sigma*np.sqrt(self.T))
        d2 = d1 - (self.sigma*np.sqrt(self.T))
        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)
        C_0 = self.S0*N_d1 - self.K*np.exp(-self.r*self.T)*N_d2
        return C_0

    # Method for computing vainilla european call discounted payoff.
    def discounted_payoff(self, ST):
        # Calculate option payoff.
        payoff = np.maximum(ST - self.K, 0)
        # Returns discounted payoff.
        return self.discounted_price(payoff)
    
    # Define function to return option type.
    def __str__(self) -> str:
        return "Vainilla european call option"

# Define class for binary european call options 
class BinaryEuropeanCallOption(Option):
    # Initializer.
    def __init__(self, S0, K, sigma, r, T):
        super().__init__(S0, K, sigma, r, T)

    # Method for computing binary cash-or-nothing european call black scholes 
    # formula.
    def black_scholes_formula(self):
        d1 = (self.r*self.T - np.log(self.K/self.S0) - ((self.sigma**2)/2)*T)
        d1 /= (self.sigma*np.sqrt(self.T))
        N_d1 = stats.norm.cdf(d1)
        C_0 = np.exp(-self.r*self.T)*N_d1
        return C_0

    # Method for computing binary cash-or-nothing european call discounted 
    # payoff.
    def discounted_payoff(self, ST):
        # Calculate option payoff.
        payoff = np.where(ST > self.K, 1, 0)
        # Returns discounted payoff.
        return self.discounted_price(payoff)

    # Method for returning option type.   
    def __str__(self) -> str:
        return "Binary cash-or-nothing european call option"

# Function for approximating price at time T using Geometric Brownian motion.
def GBM(S0: float, r: float, sigma: float, N: int, M: int, P: int) -> NDArray[np.float64]:
    # Time step.
    dt = T / N
    # Initialize the price process.
    S = np.full((M, P), S0, dtype=float)
    # Simulate the price process using the weak Euler scheme by SDE.
    for _ in range(N):
        # Initialize standard error terms.
        Z = rng.standard_normal((M,P))
        # Compute price for current time step.
        S[:, :] += + r*S[:, :]*dt + sigma*S[:, :]*np.sqrt(dt)*Z
    return S

# Function for approximating the price of an option using monte carlo method.
def monte_carlo_solver(option: Option, ST: NDArray[np.float64]):
    discounted_payoff = option.discounted_payoff(ST)
    return np.mean(discounted_payoff, axis=0)

# Function for approximating the price of an option using monte carlo method with control variates.
def control_variates_solver(option: Option, ST: NDArray[np.float64]):
    X = ST
    Y = option.discounted_payoff(X) # Get the discounted payoff of the option at time ST.
    EX = option.forward_stock_price() # Get the forward price given the current stock price S0*e^(rT).
    Y_bar = np.mean(Y, axis=0) # Compute sample mean of the discounted payoff.
    X_bar = np.mean(X, axis=0) # Compute sample mean of the forward price.
    b = np.sum((X-X_bar)*(Y-Y_bar), axis=0) / np.sum(np.power((X-X_bar),2), axis=0) # Finds optimal coefficient.
    return np.mean(Y - b*(X - EX), axis=0)  # Estimate Y(b)

S0 = 100   # Stock price
K = 100 # Strike price
sigma = 0.02  # Volatility
r = 0.05   # Risk-free interest rate
T = 1   # Time to maturity
N = 16   # Number of time steps
P = 100 # Number of samples

# Number of realizations to use.
realizations = 1024*np.array([1, 4, 16, 64, 256, 512])



methods = [
    ("Plain monte carlo", monte_carlo_solver),
    ("Monte carlo with control variates", control_variates_solver),
]

init = time.time()
eur_opt = EuropeanCallOption(S0, K, sigma, r, T)
bin_opt = BinaryEuropeanCallOption(S0, K, sigma, r, T)
options: list[Option] = [eur_opt, bin_opt]
for method_name, method_solver in methods:
    print(f"Starting Method: {method_name}")
    for option in options:
        print(f"\tOption type: {option}")

        V0 = option.black_scholes_formula()
        V0_hat = np.empty((len(realizations), P))
        
        for i, M in enumerate(realizations):
            print(f"\t\tExecuting for {M} realizations")
            start = time.time() 
            rng = default_rng(31)
            ST = GBM(S0, r, sigma, N, M, P)
            result = method_solver(option, ST)
            V0_hat[i, :] = result 
            end = time.time()
            print(f"\t\tTime elapsed: {abs(end - start):.2f}s")
            print()

        error = V0 - V0_hat
        mean_error = error.mean(axis=1)
        std_error = error.std(axis=1, ddof=1)

        np.set_printoptions(formatter={'float_kind':"{:.10f}".format})
        print()
        print(f"\tReal price: {V0:.4f}")
        print(f"\tApprox price (by num of realizations): {V0_hat.mean(axis=1)}")
        print(f"\tMean error (by num of realizations): {mean_error}")
        print(f"\tStd error (by num of realizations): {std_error}")

        alpha = 0.05
        t = [stats.t(m-1).isf(alpha/2) for m in realizations]
        top = np.round(mean_error + t*std_error/np.sqrt(realizations), 12)
        bottom = np.round(mean_error - t*std_error/np.sqrt(realizations), 12)

        plt.xticks(range(1,len(realizations)+1), labels=[f"4^{i+10}" for i in range(len(realizations))])
        
        for i in range(1,len(realizations)+1):
            plt.plot([i, i], [top[i-1], bottom[i-1]], color='black')
            plt.plot([ i - 0.1, i + 0.1], [top[i-1], top[i-1]], color='black')
            plt.plot([ i - 0.1, i + 0.1], [bottom[i-1], bottom[i-1]], color='black')
            plt.plot(i, np.round(mean_error[i-1], 12), markersize=3, marker='o', color="blue")

        plt.plot(range(1,len(realizations)+1), np.round(mean_error, 12), color="blue")

        plt.xlabel('Realizations')
        plt.ylabel('Error')
        plt.show()

    print(f"Ending Method: {method_name}")
    print("="*100)
