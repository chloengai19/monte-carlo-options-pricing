"""
Monte Carlo pricing for a European call option.

Compares a Monte Carlo estimate against the closed-form Black-Scholes 
price, then checks whether antithetic variates reduce the simulation error.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# parameters
S0 = 100        # current stock price
K = 100         # strike price
r = 0.05        # risk-free rate
sigma = 0.2     # volatility
T = 1.0         # time to maturity (years)
N = 100_000     # number of simulations


def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


bsm_price = black_scholes_call(S0, K, r, sigma, T)


def mc_call(S0, K, r, sigma, T, N):
    Z = np.random.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    std_error = np.exp(-r * T) * np.std(payoff) / np.sqrt(N)
    return price, std_error


def mc_call_antithetic(S0, K, r, sigma, T, N):
    # draw half the normals, mirror them for the other half
    half_N = N // 2
    Z = np.random.standard_normal(half_N)
    Z_anti = np.concatenate([Z, -Z])

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_anti)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    std_error = np.exp(-r * T) * np.std(payoff) / np.sqrt(N)
    return price, std_error


np.random.seed(1)
price_plain, se_plain = mc_call(S0, K, r, sigma, T, N)
price_anti, se_anti = mc_call_antithetic(S0, K, r, sigma, T, N)

print("Black-Scholes price:", bsm_price)
print("Monte Carlo price:", price_plain, " SE:", se_plain)
print("Antithetic Monte Carlo price:", price_anti, " SE:", se_anti)
print("Variance reduction factor:", (se_plain / se_anti) ** 2)