import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Q1: Estimation Statistics
# - Point Estimate: Single value used to estimate a population parameter (e.g., sample mean).
# - Interval Estimate: Range of values (confidence interval) likely to contain the parameter.

# Q2: Estimate population mean using sample mean and std
def estimate_population_mean(sample_mean, sample_std, n):
    return (sample_mean, sample_mean - 1.96 * sample_std / np.sqrt(n), sample_mean + 1.96 * sample_std / np.sqrt(n))

# Q3: Hypothesis Testing
# - Process of testing assumptions about a population.
# - Importance: Helps in decision-making using data.

# Q4: Example Hypothesis
# H0: μ_male = μ_female
# H1: μ_male > μ_female

# Q5: Hypothesis test for difference in two means
def hypothesis_test_two_means(x1, x2, s1, s2, n1, n2):
    pooled_se = np.sqrt((s1**2)/n1 + (s2**2)/n2)
    t_stat = (x1 - x2) / pooled_se
    df = min(n1, n2) - 1  # Approximate df
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return t_stat, p_val

# Q6: Null & Alternative Hypothesis
# H0: No effect or difference (μ1 = μ2)
# H1: There is an effect (μ1 ≠ μ2, or >, <)

# Q7: Steps in Hypothesis Testing
# 1. Define H0 and H1
# 2. Choose significance level (α)
# 3. Compute test statistic
# 4. Find p-value
# 5. Compare p-value with α to accept/reject H0

# Q8: p-value
# - Probability of observing data as extreme as sample if H0 is true
# - Low p-value (< α) → reject H0

# Q9: Plot Student’s t-distribution
df = 10
x = np.linspace(-4, 4, 100)
plt.plot(x, stats.t.pdf(x, df))
plt.title("Student's t-distribution (df=10)")
plt.xlabel("t")
plt.ylabel("PDF")
plt.grid(True)
plt.show()

# Q10: Two-sample t-test
def two_sample_t_test(mean1, mean2, std1, std2, n1, n2):
    se = np.sqrt((std1**2)/n1 + (std2**2)/n2)
    t = (mean1 - mean2) / se
    df = min(n1, n2) - 1  # conservative estimate
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return t, p

# Q11: Student’s t-distribution
# - Used when sample size is small (n < 30) and population std dev is unknown

# Q12: t-statistic
# t = (x̄ - μ) / (s / √n)

# Q13: Revenue estimation
sample_mean = 500
sample_std = 50
n = 50
ci = stats.t.interval(0.95, df=n-1, loc=sample_mean, scale=sample_std/np.sqrt(n))
print("Q13: 95% CI for population mean revenue:", ci)

# Q14: Drug test
mean_diff = 8
μ = 10
std = 3
n = 100
t = (mean_diff - μ) / (std / np.sqrt(n))
p = stats.t.sf(abs(t), df=n-1) * 2
print("Q14: t =", t, ", p =", p)

# Q15: Product weight test
μ = 5
sample_mean = 4.8
s = 0.5
n = 25
t = (sample_mean - μ) / (s / np.sqrt(n))
p = stats.t.cdf(t, df=n-1)  # one-tailed
print("Q15: t =", t, ", p =", p)

# Q16: Two student groups test
x1, x2 = 80, 75
s1, s2 = 10, 8
n1, n2 = 30, 40
t, p = two_sample_t_test(x1, x2, s1, s2, n1, n2)
print("Q16: t =", t, ", p =", p)

# Q17: Ads watched estimation
sample_mean = 4
s = 1.5
n = 50
ci = stats.t.interval(0.99, df=n-1, loc=sample_mean, scale=s/np.sqrt(n))
print("Q17: 99% CI for mean number of ads:", ci)
