# Import necessary libraries
import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt

# Q1: 95% Confidence Interval for sample with mean=50, std=5, n=30
mean = 50
std_dev = 5
n = 30
conf_level = 0.95
t_critical = stats.t.ppf((1 + conf_level)/2, df=n-1)
margin_error = t_critical * (std_dev / math.sqrt(n))
ci_95 = (mean - margin_error, mean + margin_error)
# Interpretation: We are 95% confident the true mean lies within ci_95
print("Q1:", ci_95)

# Q2: Chi-square goodness of fit test (M&M color example)
observed = [22, 18, 19, 11, 10, 20]  # sample observed
expected = [20, 20, 20, 10, 10, 20]  # expected percentages
chi_stat, chi_p = stats.chisquare(f_obs=observed, f_exp=expected)
print("Q2: Chi-square stat =", chi_stat, ", p-value =", chi_p)
# If p < 0.05, reject H0

# Q3: Chi-square test for contingency table
data = np.array([[20, 15], [10, 25], [15, 20]])
chi2, p_val, dof, expected_vals = stats.chi2_contingency(data)
print("Q3: Chi2 =", chi2, ", p-value =", p_val)
# Interpretation: If p < 0.05, significant difference between groups

# Q4: 95% CI for proportion of smokers (60 out of 500)
p_hat = 60 / 500
z = stats.norm.ppf(0.975)
moe = z * math.sqrt((p_hat * (1 - p_hat)) / 500)
ci_smoke = (p_hat - moe, p_hat + moe)
print("Q4:", ci_smoke)

# Q5: 90% CI for mean = 75, std = 12, n = 30
mean = 75
std = 12
n = 30
z90 = stats.norm.ppf(0.95)
moe90 = z90 * (std / math.sqrt(n))
ci_90 = (mean - moe90, mean + moe90)
print("Q5:", ci_90)

# Q6: Chi-square distribution plot with df=10, shade stat=15
x = np.linspace(0, 30, 300)
df = 10
plt.plot(x, stats.chi2.pdf(x, df))
plt.fill_between(x, stats.chi2.pdf(x, df), where=(x >= 15), color='red', alpha=0.5)
plt.title("Chi-square distribution df=10, shaded from 15")
plt.xlabel("Chi-square")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# Q7: 99% CI for proportion preferring Coke (520/1000)
p_hat = 520 / 1000
z = stats.norm.ppf(0.995)
moe = z * math.sqrt((p_hat * (1 - p_hat)) / 1000)
ci_99 = (p_hat - moe, p_hat + moe)
print("Q7:", ci_99)

# Q8: Chi-square goodness of fit (biased coin: 55 heads, 45 tails)
observed = [55, 45]
expected = [50, 50]
chi_stat, chi_p = stats.chisquare(f_obs=observed, f_exp=expected)
print("Q8: Chi-square stat =", chi_stat, ", p-value =", chi_p)

# Q9: Chi-square test for independence (smoker & lung cancer)
table = np.array([[60, 140], [30, 170]])
chi2, p, dof, exp = stats.chi2_contingency(table)
print("Q9: Chi2 =", chi2, ", p-value =", p)
# Interpretation: If p < 0.05, there is significant association between smoking and lung cancer

import numpy as np
from scipy.stats import chi2_contingency, t

# Q10: Chi-Square Test for Independence
# Observed frequencies
observed = np.array([[200, 150, 150],  # U.S.
                     [225, 175, 100]]) # U.K.

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(observed)

# Significance level
alpha_chi2 = 0.01

print("Q10 Results:")
print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"P-Value: {p_value:.4f}")
print(f"Degrees of Freedom: {dof}")
if p_value < alpha_chi2:
    print("Reject the null hypothesis: There is a significant association between chocolate preference and country.")
else:
    print("Fail to reject the null hypothesis: No significant association.")

# Q11: One-Sample t-Test
sample_mean = 72
hypothesized_mean = 70
sample_std = 10
n = 30
alpha_t = 0.05

# Calculate t-statistic
t_stat = (sample_mean - hypothesized_mean) / (sample_std / np.sqrt(n))

# Degrees of freedom
df_t = n - 1

# Critical t-value for two-tailed test
critical_t = t.ppf(1 - alpha_t/2, df_t)

print("\nQ11 Results:")
print(f"t-Statistic: {t_stat:.3f}")
print(f"Critical t-Value (two-tailed, α=0.05): ±{critical_t:.3f}")
if abs(t_stat) > critical_t:
    print("Reject the null hypothesis: Population mean differs from 70.")
else:
    print("Fail to reject the null hypothesis: Insufficient evidence to conclude the mean differs from 70.")
