## Import packages
import numpy as np
import pandas as pd

## Read the dataset
#Data source: https://www.kaggle.com/datasets/sougatapramanick/happiness-index-2018-2019
data_18 = pd.read_csv("2018.csv")
print("The 2018 dataset for 'Happiness Index Scores' has",data_18.shape[0], "rows and",data_18.shape[1],"columns.")
print(data_18.head())

# ----------------------- Descriptive Statistics -----------------------

# Note: axis is a parameter used to refer to the axis of the dataset along which we wish to consider a measure. 
# axis =0 implies performing the measure along columns, and axis=1 implies performing along the rows.

# ******************* Measures of central tendency ******************* 

#MEAN
mean_scores = np.mean(data_18.Score) #OR np.mean(data_18["Score"])
print("Mean happiness score = ", np.round(mean_scores, decimals=3))

#MEDIAN
median_scores = np.median(data_18.Score)
print("Median happiness scores = ",median_scores)

# ******************* Measures of dispersion ******************* 

#RANGE
min_gen = np.amin(data_18.Score, axis=0)
print("Minimum value of happiness scores = ", min_gen)
max_gen = np.amax(data_18.Score, axis=0)
print("Maximum value of happiness scores = ", max_gen)
# Range can be calculated from Max-Min
print("Range of happiness scores (Max-Min) = ",max_gen-min_gen)
# Range can also be calculated from an in-built function
print("Range of happiness scores (built-in function) = ",np.ptp(data_18.Score, axis=0))

#VARIANCE AND STANDARD DEVIATION
var_scores = np.var(data_18.Score, axis=0)
print("Variance of happiness scores = ", np.round(var_scores, decimals=3))
std_scores = np.std(data_18.Score, axis=0)
print("Standard deviation of happiness scores = ",np.round(std_scores, decimals=3))
#print("Square root of variance = ", np.sqrt(var1))

#PERCENTILES
#25th percentile
perc_25 = np.percentile(data_18.Score, q=25)
print("25th percentile of happiness scores = ",np.round(perc_25, decimals=3))

# ******************* Association between variables ******************* 

#COVARIANCE
cov_happiness_lifeexp = np.cov(data_18["Score"],data_18["Healthy life expectancy"])
print("Covariance matrix between variables Happiness Score and Healthy life expectancy = ",cov_happiness_lifeexp)

#CORRELATION
corr_happiness_lifeexp = np.corrcoef(data_18["Score"],data_18["Healthy life expectancy"])
print("Correlation matrix between variables Happiness Score and Healthy life expectancy = ",corr_happiness_lifeexp)
