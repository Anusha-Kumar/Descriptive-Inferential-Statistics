## Import packages
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import weightstats as stests

## Read the dataset
#Data source: https://www.kaggle.com/datasets/sougatapramanick/happiness-index-2018-2019
data_18 = pd.read_csv("2018.csv")
print("The 2018 dataset for 'Happiness Index Scores' has",data_18.shape[0], "rows and",data_18.shape[1],"columns.")
print(data_18.head())

# ----------------------- Inferential Statistics -----------------------

# We use the notion that a null hypothesis is rejected if the p-val from a statistical test is less than an alhpa (usually 0.5) 
# and fail to reject the null hypothesis otherwise. 

# *********** Test a claim that the average score of perception of freedom of different countries is 0.7. ***********

## Single sample Z-test if we assume the population variance is known
mu0=0.7
z_stat ,p_val = stests.ztest(np.array(data_18["Freedom to make life choices"]), value=mu0)
print(f'z-statistic is {z_stat}')
print(f'p-value for the test is {p_val}')

## Single sample t-test if we assume the population variance is unknown
mu0=0.7
t_stat ,p_val = stats.ttest_1samp(np.array(data_18["Freedom to make life choices"]), mu0)
print(f't-statistic is {t_stat}')
print(f'p-value for the test is {p_val}')

# *********** Test a claim that the average score of perception of freedom is equal to the average score of healthy life expectancy of different countries. ***********

## Two sample t-test assuming unknown variance
samp1 = np.array(data_18["Freedom to make life choices"])
samp2 = np.array(data_18["Healthy life expectancy"])
t_stat ,p_val = stats.ttest_ind(samp1,samp2)
print(f't-statistic is {t_stat}')
print(f'p-value for the test is {p_val}')
