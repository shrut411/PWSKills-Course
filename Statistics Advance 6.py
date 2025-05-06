import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

# Q1: Assumptions for ANOVA and Violations
# ANOVA requires: 1) Normality—data in each group should be normally distributed; 
# 2) Homogeneity of Variance—variances across groups should be equal; 
# 3) Independence—observations must be independent. 
# Violations: Non-normal data (e.g., skewed income) can mislead F-statistic, use Kruskal-Wallis instead. 
# Unequal variances (e.g., one group’s scores vary more) inflate F-statistic, use Welch’s ANOVA. 
# Non-independent data (e.g., same subjects tested twice) requires repeated measures ANOVA.

# Q2: Types of ANOVA and Situations
# 1) One-Way ANOVA: Compare means across one factor (e.g., test scores across 3 teaching methods).
# 2) Two-Way ANOVA: Examine two factors and their interaction (e.g., test scores by method and gender).
# 3) Repeated Measures ANOVA: Within-subject designs (e.g., test scores before and after intervention).

# Q3: Partitioning Variance in ANOVA
# Variance splits into: 1) Between-Group (SSB)—factor differences; 
# 2) Within-Group (SSW)—random variation; 3) Total (SST), where SST = SSB + SSW. 
# This is key to determine if between-group differences are significant via the F-statistic.

# Q4: Calculating SST, SSE (SSB), and SSR in One-Way ANOVA
print("Q4: One-Way ANOVA - SST, SSE, SSR")
data_q4 = {'Group': ['A']*20 + ['B']*20 + ['C']*20, 
           'Value': np.random.normal(10, 2, 20).tolist() + 
                    np.random.normal(12, 2, 20).tolist() + 
                    np.random.normal(11, 2, 20).tolist()}
df_q4 = pd.DataFrame(data_q4)
model_q4 = ols('Value ~ Group', data=df_q4).fit()
anova_table_q4 = anova_lm(model_q4)
print(anova_table_q4)
# SST (Total SS) = SSB (Between SS) + SSW (Within SS)

# Q5: Main Effects and Interaction Effects in Two-Way ANOVA
print("\nQ5: Two-Way ANOVA - Main and Interaction Effects")
data_q5 = {'Factor1': ['A']*30 + ['B']*30 + ['A']*30 + ['B']*30,
           'Factor2': ['X']*30 + ['X']*30 + ['Y']*30 + ['Y']*30,
           'Value': np.random.normal(10, 2, 30).tolist() + 
                    np.random.normal(12, 2, 30).tolist() + 
                    np.random.normal(11, 2, 30).tolist() + 
                    np.random.normal(13, 2, 30).tolist()}
df_q5 = pd.DataFrame(data_q5)
model_q5 = ols('Value ~ Factor1 + Factor2 + Factor1:Factor2', data=df_q5).fit()
anova_table_q5 = anova_lm(model_q5)
print(anova_table_q5)
# Main effects: Factor1, Factor2. Interaction: Factor1:Factor2.

# Q6: Interpreting One-Way ANOVA Results
# F-statistic = 5.23, p-value = 0.02. Since p < 0.05, reject null hypothesis. 
# There are significant differences between group means. 
# Follow up with a post-hoc test (e.g., Tukey’s HSD) to identify which groups differ.

# Q7: Handling Missing Data in Repeated Measures ANOVA
# Methods: 1) Listwise Deletion—exclude subjects with missing data, reduces power. 
# 2) Imputation—use mean or regression, but risks bias if data isn’t missing at random. 
# 3) Mixed-Effects Models—better handle missing data. 
# Consequences: Reduced power, potential bias if missingness is related to outcome.

# Q8: Common Post-Hoc Tests After ANOVA
# 1) Tukey’s HSD: All pairwise comparisons (e.g., test scores across 3 methods). 
# 2) Bonferroni Correction: Adjusts for multiple comparisons, more conservative. 
# 3) Scheffé Test: Flexible for complex comparisons, less powerful. 
# Use Tukey’s HSD for all pairwise comparisons after a significant ANOVA.

# Q9: One-Way ANOVA for Mean Weight Loss
print("\nQ9: One-Way ANOVA - Weight Loss Across Diets")
data_q9 = {'Diet': ['A']*17 + ['B']*17 + ['C']*16,
           'WeightLoss': np.random.normal(5, 1.5, 17).tolist() + 
                         np.random.normal(7, 1.5, 17).tolist() + 
                         np.random.normal(6, 1.5, 16).tolist()}
df_q9 = pd.DataFrame(data_q9)
model_q9 = ols('WeightLoss ~ Diet', data=df_q9).fit()
anova_table_q9 = anova_lm(model_q9)
print(anova_table_q9)
# If p < 0.05, there are significant differences in weight loss between diets.

# Q10: Two-Way ANOVA for Task Completion Time
print("\nQ10: Two-Way ANOVA - Task Completion Time")
data_q10 = {'Program': ['A']*10 + ['B']*10 + ['C']*10 + ['A']*10 + ['B']*10 + ['C']*10,
            'Experience': ['Novice']*30 + ['Experienced']*30,
            'Time': np.random.normal(20, 3, 10).tolist() + 
                    np.random.normal(18, 3, 10).tolist() + 
                    np.random.normal(22, 3, 10).tolist() + 
                    np.random.normal(15, 3, 10).tolist() + 
                    np.random.normal(13, 3, 10).tolist() + 
                    np.random.normal(17, 3, 10).tolist()}
df_q10 = pd.DataFrame(data_q10)
model_q10 = ols('Time ~ Program + Experience + Program:Experience', data=df_q10).fit()
anova_table_q10 = anova_lm(model_q10)
print(anova_table_q10)
# Interpret F-statistics and p-values for Program, Experience, and interaction.

# Q11: Two-Sample t-Test and Post-Hoc Test for Teaching Methods
print("\nQ11: Two-Sample t-Test - Teaching Methods")
data_q11_control = np.random.normal(75, 10, 50)
data_q11_treatment = np.random.normal(80, 10, 50)
t_stat, p_val = ttest_ind(data_q11_control, data_q11_treatment)
print(f"t-Statistic: {t_stat:.3f}, p-Value: {p_val:.4f}")
# If p < 0.05, significant difference. Post-hoc not needed for two groups, but could use effect size (e.g., Cohen’s d).

# Q12: Repeated Measures ANOVA for Store Sales
print("\nQ12: Repeated Measures ANOVA - Store Sales")
data_q12 = pd.DataFrame({
    'Store': ['A']*30 + ['B']*30 + ['C']*30,
    'Day': list(range(30))*3,
    'Sales': np.random.normal(100, 15, 30).tolist() + 
             np.random.normal(110, 15, 30).tolist() + 
             np.random.normal(105, 15, 30).tolist()
})
rm_anova = pg.rm_anova(dv='Sales', within='Store', subject='Day', data=data_q12)
print(rm_anova)
# If p < 0.05, significant differences in sales. Follow up with pairwise comparisons.
posthoc_q12 = pg.pairwise_tests(dv='Sales', within='Store', subject='Day', data=data_q12)
print("\nQ12 Post-Hoc Test:")
print(posthoc_q12)
