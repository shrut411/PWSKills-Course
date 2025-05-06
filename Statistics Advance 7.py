import numpy as np
from scipy.stats import f, f_oneway

# Q1: Write a Python function that takes in two arrays of data and calculates the F-value for a variance ratio test.
# The function should return the F-value and the corresponding p-value for the test.
def variance_ratio_test(data1, data2):
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)
    F = var1 / var2 if var1 > var2 else var2 / var1  # Ensure F > 1
    df1 = len(data1) - 1
    df2 = len(data2) - 1
    p_value = 2 * (1 - f.cdf(F, df1, df2))  # Two-tailed test
    return F, p_value

# Q2: Given a significance level of 0.05 and the degrees of freedom for the numerator and denominator of an F-distribution,
# write a Python function that returns the critical F-value for a two-tailed test.
def critical_f_value(alpha, dfn, dfd):
    alpha_half = alpha / 2  # Two-tailed
    f_lower = f.ppf(alpha_half, dfn, dfd)
    f_upper = f.ppf(1 - alpha_half, dfn, dfd)
    return f_lower, f_upper  # Return both critical values (lower and upper)

# Q3: Write a Python program that generates random samples from two normal distributions with known variances
# and uses an F-test to determine if the variances are equal. The program should output the F-value, degrees of freedom, and p-value.
np.random.seed(42)  # For reproducibility
sample1_q3 = np.random.normal(0, np.sqrt(10), 12)
sample2_q3 = np.random.normal(0, np.sqrt(15), 12)
F_q3, p_q3 = variance_ratio_test(sample1_q3, sample2_q3)
df1_q3 = len(sample1_q3) - 1
df2_q3 = len(sample2_q3) - 1
print(f"Q3: F-value: {F_q3:.3f}, Degrees of Freedom: ({df1_q3}, {df2_q3}), p-value: {p_q3:.4f}")

# Q4: The variances of two populations are known to be 10 and 15. A sample of 12 observations is taken from each population.
# Conduct an F-test at the 5% significance level to determine if the variances are significantly different.
sample1_q4 = np.random.normal(0, np.sqrt(10), 12)
sample2_q4 = np.random.normal(0, np.sqrt(15), 12)
F_q4, p_q4 = variance_ratio_test(sample1_q4, sample2_q4)
alpha_q4 = 0.05
f_lower_q4, f_upper_q4 = critical_f_value(alpha_q4, 11, 11)
print(f"\nQ4: F-value: {F_q4:.3f}, p-value: {p_q4:.4f}")
print(f"Critical F-values (α=0.05): Lower: {f_lower_q4:.3f}, Upper: {f_upper_q4:.3f}")
if F_q4 < f_lower_q4 or F_q4 > f_upper_q4:
    print("Reject null hypothesis: Variances are significantly different.")
else:
    print("Fail to reject null hypothesis: No significant difference in variances.")

# Q5: A manufacturer claims that the variance of the diameter of a certain product is 0.005. A sample of 25 products is taken,
# and the sample variance is found to be 0.006. Conduct an F-test at the 1% significance level to determine if the claim is justified.
sample_var_q5 = 0.006
claimed_var_q5 = 0.005
F_q5 = sample_var_q5 / claimed_var_q5  # Sample variance / claimed variance
df1_q5 = 24  # Sample df (n-1)
df2_q5 = float('inf')  # Population variance known, but we approximate with large df
p_q5 = 2 * (1 - f.cdf(F_q5, df1_q5, df2_q5))
alpha_q5 = 0.01
f_lower_q5, f_upper_q5 = critical_f_value(alpha_q5, df1_q5, 1000)  # Approximate large df
print(f"\nQ5: F-value: {F_q5:.3f}, p-value: {p_q5:.4f}")
print(f"Critical F-values (α=0.01): Lower: {f_lower_q5:.3f}, Upper: {f_upper_q5:.3f}")
if F_q5 < f_lower_q5 or F_q5 > f_upper_q5:
    print("Reject null hypothesis: The claimed variance is not justified.")
else:
    print("Fail to reject null hypothesis: The claimed variance is justified.")

# Q6: Write a Python function that takes in the degrees of freedom for the numerator and denominator of an F-distribution
# and calculates the mean and variance of the distribution. The function should return the mean and variance as a tuple.
def f_distribution_stats(dfn, dfd):
    mean = dfd / (dfd - 2) if dfd > 2 else np.nan  # Mean exists if dfd > 2
    variance = (2 * dfd**2 * (dfn + dfd - 2)) / (dfn * (dfd - 2)**2 * (dfd - 4)) if dfd > 4 else np.nan  # Variance exists if dfd > 4
    return mean, variance

# Example for Q6
mean_q6, var_q6 = f_distribution_stats(11, 11)
print(f"\nQ6: Mean of F-distribution: {mean_q6:.3f}, Variance: {var_q6:.3f}")

# Q7: A random sample of 10 measurements is taken from a normal population with unknown variance. The sample variance is found to be 25.
# Another random sample of 15 measurements is taken from another normal population with unknown variance, and the sample variance is 20.
# Conduct an F-test at the 10% significance level to determine if the variances are significantly different.
sample_var1_q7 = 25
sample_var2_q7 = 20
F_q7 = sample_var1_q7 / sample_var2_q7 if sample_var1_q7 > sample_var2_q7 else sample_var2_q7 / sample_var1_q7
df1_q7 = 10 - 1
df2_q7 = 15 - 1
p_q7 = 2 * (1 - f.cdf(F_q7, df1_q7, df2_q7))
alpha_q7 = 0.10
f_lower_q7, f_upper_q7 = critical_f_value(alpha_q7, df1_q7, df2_q7)
print(f"\nQ7: F-value: {F_q7:.3f}, p-value: {p_q7:.4f}")
print(f"Critical F-values (α=0.10): Lower: {f_lower_q7:.3f}, Upper: {f_upper_q7:.3f}")
if F_q7 < f_lower_q7 or F_q7 > f_upper_q7:
    print("Reject null hypothesis: Variances are significantly different.")
else:
    print("Fail to reject null hypothesis: No significant difference in variances.")

# Q8: The following data represent the waiting times in minutes at two different restaurants on a Saturday night:
# Restaurant A: 24, 25, 28, 23, 22, 20, 27; Restaurant B: 31, 33, 35, 30, 32, 36.
# Conduct an F-test at the 5% significance level to determine if the variances are significantly different.
data_a_q8 = [24, 25, 28, 23, 22, 20, 27]
data_b_q8 = [31, 33, 35, 30, 32, 36]
F_q8, p_q8 = variance_ratio_test(data_a_q8, data_b_q8)
df1_q8 = len(data_a_q8) - 1
df2_q8 = len(data_b_q8) - 1
alpha_q8 = 0.05
f_lower_q8, f_upper_q8 = critical_f_value(alpha_q8, df1_q8, df2_q8)
print(f"\nQ8: F-value: {F_q8:.3f}, p-value: {p_q8:.4f}")
print(f"Critical F-values (α=0.05): Lower: {f_lower_q8:.3f}, Upper: {f_upper_q8:.3f}")
if F_q8 < f_lower_q8 or F_q8 > f_upper_q8:
    print("Reject null hypothesis: Variances are significantly different.")
else:
    print("Fail to reject null hypothesis: No significant difference in variances.")

# Q9: The following data represent the test scores of two groups of students:
# Group A: 80, 85, 90, 92, 87, 83; Group B: 75, 78, 82, 79, 81, 84.
# Conduct an F-test at the 1% significance level to determine if the variances are significantly different.
data_a_q9 = [80, 85, 90, 92, 87, 83]
data_b_q9 = [75, 78, 82, 79, 81, 84]
F_q9, p_q9 = variance_ratio_test(data_a_q9, data_b_q9)
df1_q9 = len(data_a_q9) - 1
df2_q9 = len(data_b_q9) - 1
alpha_q9 = 0.01
f_lower_q9, f_upper_q9 = critical_f_value(alpha_q9, df1_q9, df2_q9)
print(f"\nQ9: F-value: {F_q9:.3f}, p-value: {p_q9:.4f}")
print(f"Critical F-values (α=0.01): Lower: {f_lower_q9:.3f}, Upper: {f_upper_q9:.3f}")
if F_q9 < f_lower_q9 or F_q9 > f_upper_q9:
    print("Reject null hypothesis: Variances are significantly different.")
else:
    print("Fail to reject null hypothesis: No significant difference in variances.")
