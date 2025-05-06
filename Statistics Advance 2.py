from scipy.stats import norm
import numpy as np

# Q1: PMF vs PDF
# - PMF (Probability Mass Function): Used for discrete variables. Gives probability of exact outcomes.
#   Example: P(X=2) in a binomial distribution.
# - PDF (Probability Density Function): Used for continuous variables. Probability at a point is zero; area under curve gives probability.
#   Example: Height distribution of people.

# Q2: CDF (Cumulative Distribution Function)
# - CDF gives the probability that a random variable X is less than or equal to a certain value.
#   Example: P(X ≤ 10)
# - CDF is used for calculating cumulative probabilities and for comparing distributions.

# Q3: Normal Distribution usage examples
# - Human height, blood pressure, IQ scores.
# - Parameters:
#   Mean (μ) → Center of the bell curve
#   Std Dev (σ) → Width of the curve (spread)

# Q4: Importance of Normal Distribution
# - Widely used in statistics due to the Central Limit Theorem.
# - Used in quality control, finance, biology, etc.
# - Real-life examples: Test scores, measurement errors, employee productivity.

# Q5: Bernoulli Distribution
# - Models a single experiment with two outcomes: success (1) or failure (0).
# - Example: Tossing a coin once.
# - Difference:
#   - Bernoulli: single trial
#   - Binomial: multiple independent Bernoulli trials

# Q6: Probability of observation > 60 with μ=50, σ=10
z = (60 - 50) / 10  # z-score = 1
p = 1 - norm.cdf(z)  # P(X > 60)
print(f"Q6: Probability of X > 60: {p:.4f}")  # Output: ~0.1587

# Q7: Uniform Distribution
# - All outcomes equally likely.
# - Example: Rolling a fair die → P(1) = P(2) = ... = P(6) = 1/6

# Q8: Z-score
# - Z = (X - μ) / σ
# - Tells how many standard deviations a value is from the mean.
# - Importance:
#   - Used in hypothesis testing
#   - Enables comparison between different distributions

# Q9: Central Limit Theorem (CLT)
# - States that the sampling distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the population's distribution.
# - Significance:
#   - Justifies use of normal distribution in many statistical methods

# Q10: Assumptions of CLT
# - Samples are independent
# - Sample size is sufficiently large (n > 30 generally)
# - Population has finite variance

