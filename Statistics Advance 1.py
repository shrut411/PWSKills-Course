# Q1. Probability Density Function (PDF)
# A PDF describes the probability of a continuous random variable falling within a range of values.
# The total area under the PDF curve equals 1.

# Q2. Types of Probability Distributions:
# - Discrete: e.g., Binomial, Poisson
# - Continuous: e.g., Normal, Exponential

# Q3. Function to calculate PDF of a normal distribution
from scipy.stats import norm, poisson
import numpy as np
import matplotlib.pyplot as plt

def normal_pdf(x, mean, std_dev):
    return norm.pdf(x, loc=mean, scale=std_dev)

# Example usage:
x_val = 2
mean = 0
std_dev = 1
pdf_result = normal_pdf(x_val, mean, std_dev)
print(f"Q3: PDF of normal distribution at x={x_val}: {pdf_result}")

# Q4. Properties of Binomial Distribution:
# - Fixed number of trials (n)
# - Two possible outcomes per trial (success/failure)
# - Constant probability of success (p)
# - Independent trials
# Example events:
# 1. Flipping a coin 10 times.
# 2. Inspecting 100 products for defects.

# Q5. Generate 1000 samples from a binomial distribution and plot histogram
binom_data = np.random.binomial(n=10, p=0.4, size=1000)
plt.hist(binom_data, bins=10, edgecolor='black')
plt.title("Q5: Binomial Distribution Histogram")
plt.xlabel("Number of Successes")
plt.ylabel("Frequency")
plt.show()

# Q6. Function to calculate CDF of a Poisson distribution
def poisson_cdf(x, mean):
    return poisson.cdf(x, mu=mean)

# Example usage:
x_val_poisson = 3
lambda_val = 4
cdf_result = poisson_cdf(x_val_poisson, lambda_val)
print(f"Q6: CDF of Poisson distribution at x={x_val_poisson}: {cdf_result}")

# Q7. Difference between Binomial and Poisson:
# - Binomial: fixed trials, known probability (p), bounded outcome
# - Poisson: models rare events in a fixed interval, uses only mean (λ), unbounded

# Q8. Generate Poisson samples and compute mean, variance
poisson_data = np.random.poisson(lam=5, size=1000)
sample_mean = np.mean(poisson_data)
sample_variance = np.var(poisson_data)
print(f"Q8: Poisson Sample Mean: {sample_mean}")
print(f"Q8: Poisson Sample Variance: {sample_variance}")

# Q9. Mean and Variance relations:
# - Binomial: Mean = n*p, Variance = n*p*(1-p)
# - Poisson: Mean = λ, Variance = λ

# Q10. In normal distribution, least frequent data:
# Least frequent data points lie in the tails (extreme ends) far from the mean.
