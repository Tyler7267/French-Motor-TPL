import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.metrics import d2_tweedie_score, mean_poisson_deviance, mean_gamma_deviance

#----------------PHASE 1: Data Preprocessing and Feature Engineering----------------#
print("Starting Phase 1: Data Preprocessing and Feature Engineering...")
combined_data = pd.read_csv('combined_data.csv')

categorical_features = ['Area', 'VehPower', 'VehBrand', 'VehGas', 'Region']
for col in categorical_features:
    combined_data[col] = combined_data[col].astype(str)


# Transforming the 'Density' variable using logarithmic transformation to handle skewness
combined_data['LogDensity'] = np.log(combined_data['Density'].astype(float))
# Binning 'DrivAge' and 'VehAge' into categorical bins
combined_data['DrivAge_Binned'] = pd.cut(
    combined_data['DrivAge'],
    bins = [17, 22, 26, 30, 40, 50, 60, 75, 100],
    labels = ['18-22', '23-26', '27-30', '31-40', '41-50', '51-60', '61-75', '76+'] 
)
combined_data['VehAge_Binned'] = pd.cut(
    combined_data['VehAge'],
    bins = [-1, 1, 4, 10, 100],
    labels = ['0-1', '2-4', '5-10', '11+']
)

features_numeric = ['BonusMalus', 'LogDensity']
features_categorical = ['Area', 'VehPower', 'VehBrand', 'VehGas', 'Region', 'DrivAge_Binned', 'VehAge_Binned']
all_features = features_numeric + features_categorical  

print("Feature engineering completed. Sample of transformed data:")
print(combined_data[all_features].head())

# Creating a stratified train-test split for the frequency model
percent_train = 0.8
print(f"Creating a stratified training and testing datasets...{percent_train} of data will be used for training.")

combined_data['HasClaim'] = (combined_data['ClaimNb'] > 0).astype(int)

train_df, test_df = train_test_split(
    combined_data,
    test_size = 1 - percent_train,
    stratify = combined_data['HasClaim'],
    random_state = 42
)

# Implementing a Buhlmann-Straub Credibility Adjustment

region_stats = train_df.groupby('Region').agg(
    Total_Claims = ('ClaimNb', 'sum'),
    Total_Exposure = ('Exposure', 'sum')
).reset_index()

region_stats['Observed_Frequency'] = region_stats['Total_Claims'] / region_stats['Total_Exposure']

mu = train_df['ClaimNb'].sum() / train_df['Exposure'].sum()
g = len(region_stats) # Number of groups (regions)

# Process Variance
sample_vars = []
for reg in region_stats['Region']:
    sub = train_df[train_df['Region'] == reg]
    if len(sub) > 1:
        var_i = np.sum(sub['Exposure'] * ((sub['ClaimNb']/sub['Exposure']) - (sub['ClaimNb'].sum()/sub['Exposure'].sum()))**2) / (len(sub) - 1)
    else:
        var_i = 0
    sample_vars.append(var_i)

s2 = np.mean(sample_vars)  # EPV

# Estimating Variance of hypothetical means
total_exposure_all = region_stats['Total_Exposure'].sum()
c_factor = (total_exposure_all - np.sum(region_stats['Total_Exposure']**2)/ total_exposure_all) / (g - 1)

# Weighted variance of group means relative to the global mean
raw_vhm = np.sum(region_stats['Total_Exposure'] * (region_stats['Observed_Frequency'] - mu)**2)
a = max(0, (raw_vhm - (g - 1) * s2) / c_factor)

# Calculating credibility Z and Credility premium
k = s2 / a if a > 0 else float('inf')

region_stats['Z'] = np.where(a > 0, region_stats['Total_Exposure'] / (region_stats['Total_Exposure'] + k), 0.0)
region_stats['Credibility_Frequency'] = region_stats['Z'] * region_stats['Observed_Frequency'] + ((1 - region_stats['Z']) * mu)

cred_map = dict(zip(region_stats['Region'], region_stats['Credibility_Frequency']))
train_df['Cred_Freq_Pred'] = train_df['Region'].map(cred_map)
test_df['Cred_Freq_Pred'] = test_df['Region'].map(cred_map).fillna(mu) # For unseen regions in the test set, we assign the global mean frequency

print("Stratified train-test split completed, concluding Phase 1.")
#----------------PHASE 2: Building the Frequency Model----------------#
print("Starting Phase 2: Building the Frequency Model..")

# Isolate predictors and target variable for the frequency model
features = ['BonusMalus', 'LogDensity', 'Area', 'VehPower', 'VehBrand', 'VehGas', 'Region', 'DrivAge_Binned', 'VehAge_Binned']

X_train = train_df[features]
y_train = train_df['ClaimNb'] 
exposure_train = train_df['Exposure']

X_test = test_df[features]
y_test = test_df['ClaimNb']
exposure_test = test_df['Exposure']

# Build preprocessing pipeline for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop = 'first', sparse_output = False), features_categorical)
    ],
    remainder = 'passthrough'
)

# Defining and training the Poisson GLM Pipeline
freq_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', PoissonRegressor(alpha = 1e-5, max_iter = 300))
])

print("Training the Poisson frequency model...")
freq_pipeline.fit(X_train, y_train, regressor__sample_weight = exposure_train)  
print("Frequency model training completed.")
print("Phase 2 completed")

#----------------PHASE 3: Frequency Model Evaluation----------------#
print("Starting Phase 3: Frequency Model Evaluation...")
# Predicting expected claim counts per unit of exposure, then multiplying by actual exposure
pred_freq_test = freq_pipeline.predict(X_test)*exposure_test

dev_test = mean_poisson_deviance(y_test, pred_freq_test)
print(f"Mean Poisson Deviance on test set: {dev_test:.4f}")



#---------------Phase 4: Creating the Severity Model----------------#
print("Starting Phase 4: Creating the Severity Model...")

sev_train_df = train_df[(train_df['ClaimNb'] > 0) & (train_df['TotalClaimAmount'] > 0)].copy()
sev_test_df = test_df[(test_df['ClaimNb'] > 0) & (test_df['TotalClaimAmount'] > 0)].copy()

# Defining features specifically relevant to cost size
sev_features = ['BonusMalus', 'Area', 'VehPower', 'VehBrand', 'VehGas', 'Region', 'DrivAge_Binned', 'VehAge_Binned']

X_train_sev = sev_train_df[sev_features]
y_train_sev = sev_train_df['TotalClaimAmount'] / sev_train_df['ClaimNb']

weights_train_sev = sev_train_df['ClaimNb']

X_test_sev = sev_test_df[sev_features]
y_test_sev = sev_test_df['TotalClaimAmount'] / sev_test_df['ClaimNb']
weights_test_sev = sev_test_df['ClaimNb']

preprocessor_sev = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop = 'first', sparse_output = False), ['Area', 'VehPower', 'VehBrand', 'VehGas', 'Region', 'DrivAge_Binned', 'VehAge_Binned'])
    ],
    remainder = 'passthrough'
)

# Create Gamma GLM Pipeline for severity modeling
sev_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_sev),
    ('regressor', GammaRegressor(alpha = 1e-3, max_iter = 500))
])

# Fitting the model using claim counts as regression weights
sev_pipeline.fit(X_train_sev, y_train_sev, regressor__sample_weight = weights_train_sev)
print("Severity model training completed. Concluding Phase 4.")

#----------------PHASE 5: Severity Model Evaluation----------------#
print("Starting Phase 5: Severity Model Evaluation...")

# Predict average severity costs
pred_sev_test = sev_pipeline.predict(X_test_sev)

# Asses deviance score
dev_sev_test = mean_gamma_deviance(y_test_sev, pred_sev_test, sample_weight = weights_test_sev)
print(f"Mean Gamma Deviance on test set: {dev_sev_test:.4f}")

print(f"Actual total claim cost (Test): ${sev_test_df['TotalClaimAmount'].sum():,.2f}")
print(f"Predicted total claim cost (Test): ${((pred_sev_test * weights_test_sev).sum()):,.2f}")


