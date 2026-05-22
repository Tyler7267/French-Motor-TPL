import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from sklearn.ensemble import HistGradientBoostingRegressor 

FREQ_NUMERIC_FEATURES = ['BonusMalus', 'LogDensity', 'Cred_Freq_Pred']
FREQ_CATEGORICAL_FEATURES = ['Area', 'VehPower', 'VehBrand', 'VehGas', 'DrivAge_Binned', 'VehAge_Binned']
FREQ_FEATURES = FREQ_NUMERIC_FEATURES + FREQ_CATEGORICAL_FEATURES

SEV_NUMERIC_FEATURES = ['BonusMalus']
SEV_CATEGORICAL_FEATURES = ['Area', 'VehPower', 'VehBrand', 'VehGas', 'Region', 'DrivAge_Binned', 'VehAge_Binned']
SEV_FEATURES = SEV_NUMERIC_FEATURES + SEV_CATEGORICAL_FEATURES


def prepare_and_split_data_freq(combined_data, percent_train = 0.8):
    "Cleans types, transforms skewed fields, bins features, and splits into train/test sets."
    print("Cleaning features and creating risk bins...")

    combined_data = combined_data.copy()
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

    combined_data['HasClaim'] = (combined_data['ClaimNb'] > 0).astype(int)
    # Creating a stratified train-test split for the frequency model
    train_df, test_df = train_test_split(
        combined_data,
        test_size = 1 - percent_train,
        stratify = combined_data['HasClaim'],
        random_state = 42
    )
    return train_df.copy(), test_df.copy()


def Buhlmann_straub(train_df, test_df):
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
    
    dev_cred = mean_poisson_deviance(test_df['ClaimNb'], test_df['Cred_Freq_Pred'] * test_df['Exposure'])
    
    return train_df, test_df, k, dev_cred


def build_preprocessor(categorical_features, numeric_features = None, scale_numeric = False):
    transformers = [('cat', OneHotEncoder(drop = 'first', sparse_output = False), categorical_features)]

    if scale_numeric and numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
        return ColumnTransformer(transformers = transformers)

    if numeric_features:
        return ColumnTransformer(
            transformers = [('cat', OneHotEncoder(drop = 'first', sparse_output = False), categorical_features)],
            remainder = 'passthrough'
        )

    return ColumnTransformer(transformers = transformers)


def split_validation_data(df, validation_fraction = 0.2, random_state = 42, stratify = None):
    if stratify is None:
        return train_test_split(df, test_size = validation_fraction, random_state = random_state)

    return train_test_split(
        df,
        test_size = validation_fraction,
        stratify = df[stratify],
        random_state = random_state
    )


def Poisson_GLM_Frequency_Model(train_df, test_df): 
    train_split, validation_split = split_validation_data(train_df, stratify = 'HasClaim')

    X_train = train_split[FREQ_FEATURES]
    y_train = train_split['ClaimNb']
    exposure_train = train_split['Exposure']

    X_validation = validation_split[FREQ_FEATURES]
    y_validation = validation_split['ClaimNb']
    exposure_validation = validation_split['Exposure']

    best_score = np.inf
    best_params = None

    for alpha in [1e-6, 1e-5, 1e-4, 1e-3]:
        for max_iter in [200, 500, 1000]:
            freq_pipeline = Pipeline(steps=[
                ('preprocessor', build_preprocessor(FREQ_CATEGORICAL_FEATURES, FREQ_NUMERIC_FEATURES, scale_numeric = True)),
                ('regressor', PoissonRegressor(alpha = alpha, max_iter = max_iter))
            ])
            freq_pipeline.fit(X_train, y_train, regressor__sample_weight = exposure_train)
            validation_predictions = freq_pipeline.predict(X_validation) * exposure_validation
            score = mean_poisson_deviance(y_validation, validation_predictions)

            if score < best_score:
                best_score = score
                best_params = (alpha, max_iter)

    alpha, max_iter = best_params
    final_pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor(FREQ_CATEGORICAL_FEATURES, FREQ_NUMERIC_FEATURES, scale_numeric = True)),
        ('regressor', PoissonRegressor(alpha = alpha, max_iter = max_iter))
    ])
    final_pipeline.fit(train_df[FREQ_FEATURES], train_df['ClaimNb'], regressor__sample_weight = train_df['Exposure'])

    test_predictions = final_pipeline.predict(test_df[FREQ_FEATURES]) * test_df['Exposure']
    dev_glm_freq = mean_poisson_deviance(test_df['ClaimNb'], test_predictions)

    return final_pipeline, dev_glm_freq


def Gamma_GLM_Severity_Model(train_df, test_df):
    sev_train_df = train_df[(train_df['ClaimNb'] > 0) & (train_df['TotalClaimAmount'] > 0)].copy()
    sev_test_df = test_df[(test_df['ClaimNb'] > 0) & (test_df['TotalClaimAmount'] > 0)].copy()

    sev_train_split, sev_validation_split = split_validation_data(sev_train_df)

    X_train = sev_train_split[SEV_FEATURES]
    y_train = sev_train_split['TotalClaimAmount'] / sev_train_split['ClaimNb']
    weights_train = sev_train_split['ClaimNb']

    X_validation = sev_validation_split[SEV_FEATURES]
    y_validation = sev_validation_split['TotalClaimAmount'] / sev_validation_split['ClaimNb']
    weights_validation = sev_validation_split['ClaimNb']

    best_score = np.inf
    best_params = None

    for alpha in [1e-4, 1e-3, 1e-2]:
        for max_iter in [500, 1000, 2000]:
            sev_pipeline = Pipeline(steps=[
                ('preprocessor', build_preprocessor(SEV_CATEGORICAL_FEATURES, SEV_NUMERIC_FEATURES, scale_numeric = True)),
                ('regressor', GammaRegressor(alpha = alpha, max_iter = max_iter))
            ])
            sev_pipeline.fit(X_train, y_train, regressor__sample_weight = weights_train)
            validation_predictions = sev_pipeline.predict(X_validation)
            score = mean_gamma_deviance(y_validation, validation_predictions, sample_weight = weights_validation)

            if score < best_score:
                best_score = score
                best_params = (alpha, max_iter)

    alpha, max_iter = best_params
    final_pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor(SEV_CATEGORICAL_FEATURES, SEV_NUMERIC_FEATURES, scale_numeric = True)),
        ('regressor', GammaRegressor(alpha = alpha, max_iter = max_iter))
    ])
    final_pipeline.fit(
        sev_train_df[SEV_FEATURES],
        sev_train_df['TotalClaimAmount'] / sev_train_df['ClaimNb'],
        regressor__sample_weight = sev_train_df['ClaimNb']
    )

    pred_sev_test = final_pipeline.predict(sev_test_df[SEV_FEATURES])
    dev_glm_sev = mean_gamma_deviance(
        sev_test_df['TotalClaimAmount'] / sev_test_df['ClaimNb'],
        pred_sev_test,
        sample_weight = sev_test_df['ClaimNb']
    )

    return final_pipeline, dev_glm_sev


def ML_Frequency_Model(train_df, test_df):
    train_split, validation_split = split_validation_data(train_df, stratify = 'HasClaim')

    X_train = train_split[FREQ_FEATURES]
    y_train = train_split['ClaimNb']
    exposure_train = train_split['Exposure']

    X_validation = validation_split[FREQ_FEATURES]
    y_validation = validation_split['ClaimNb']
    exposure_validation = validation_split['Exposure']

    best_score = np.inf
    best_params = None

    for learning_rate in [0.03, 0.05]:
        for max_depth in [3, 5]:
            for max_iter in [200, 300]:
                ml_freq_model = Pipeline(steps=[
                    ('preprocessor', build_preprocessor(FREQ_CATEGORICAL_FEATURES, FREQ_NUMERIC_FEATURES)),
                    ('regressor', HistGradientBoostingRegressor(
                        learning_rate = learning_rate,
                        max_depth = max_depth,
                        max_iter = max_iter,
                        min_samples_leaf = 10,
                        l2_regularization = 0.0,
                        random_state = 42
                    ))
                ])
                ml_freq_model.fit(X_train, y_train, regressor__sample_weight = exposure_train)
                validation_predictions = np.maximum(ml_freq_model.predict(X_validation) * exposure_validation, 1e-9)
                score = mean_poisson_deviance(y_validation, validation_predictions)

                if score < best_score:
                    best_score = score
                    best_params = (learning_rate, max_depth, max_iter)

    learning_rate, max_depth, max_iter = best_params
    final_ml_freq_model = Pipeline(steps=[
        ('preprocessor', build_preprocessor(FREQ_CATEGORICAL_FEATURES, FREQ_NUMERIC_FEATURES)),
        ('regressor', HistGradientBoostingRegressor(
            learning_rate = learning_rate,
            max_depth = max_depth,
            max_iter = max_iter,
            min_samples_leaf = 10,
            l2_regularization = 0.0,
            random_state = 42
        ))
    ])
    final_ml_freq_model.fit(train_df[FREQ_FEATURES], train_df['ClaimNb'], regressor__sample_weight = train_df['Exposure'])

    pred_ml_freq_test = np.maximum(final_ml_freq_model.predict(test_df[FREQ_FEATURES]) * test_df['Exposure'], 1e-9)
    dev_ml_freq = mean_poisson_deviance(test_df['ClaimNb'], pred_ml_freq_test)
    return final_ml_freq_model, dev_ml_freq


def ML_Severity_Model(train_df, test_df):
    sev_train_df = train_df[(train_df['ClaimNb'] > 0) & (train_df['TotalClaimAmount'] > 0)].copy()
    sev_test_df = test_df[(test_df['ClaimNb'] > 0) & (test_df['TotalClaimAmount'] > 0)].copy()

    sev_train_split, sev_validation_split = split_validation_data(sev_train_df)

    X_train = sev_train_split[SEV_FEATURES]
    y_train = sev_train_split['TotalClaimAmount'] / sev_train_split['ClaimNb']
    weights_train = sev_train_split['ClaimNb']

    X_validation = sev_validation_split[SEV_FEATURES]
    y_validation = sev_validation_split['TotalClaimAmount'] / sev_validation_split['ClaimNb']
    weights_validation = sev_validation_split['ClaimNb']

    best_score = np.inf
    best_params = None

    for learning_rate in [0.01, 0.02]:
        for max_depth in [3, 4]:
            for max_iter in [50, 100]:
                base_ml_sev_model = Pipeline(steps=[
                    ('preprocessor', build_preprocessor(SEV_CATEGORICAL_FEATURES, SEV_NUMERIC_FEATURES)),
                    ('regressor', HistGradientBoostingRegressor(
                        learning_rate = learning_rate,
                        max_depth = max_depth,
                        max_iter = max_iter,
                        min_samples_leaf = 50,
                        l2_regularization = 10.0,
                        random_state = 42
                    ))
                ])
                ml_sev_model = TransformedTargetRegressor(
                    regressor = base_ml_sev_model,
                    func = np.log,
                    inverse_func = np.exp
                )
                ml_sev_model.fit(X_train, y_train, regressor__sample_weight = weights_train)
                validation_predictions = ml_sev_model.predict(X_validation)
                score = mean_gamma_deviance(y_validation, validation_predictions, sample_weight = weights_validation)

                if score < best_score:
                    best_score = score
                    best_params = (learning_rate, max_depth, max_iter)

    learning_rate, max_depth, max_iter = best_params
    final_base_ml_sev_model = Pipeline(steps=[
        ('preprocessor', build_preprocessor(SEV_CATEGORICAL_FEATURES, SEV_NUMERIC_FEATURES)),
        ('regressor', HistGradientBoostingRegressor(
            learning_rate = learning_rate,
            max_depth = max_depth,
            max_iter = max_iter,
            min_samples_leaf = 50,
            l2_regularization = 10.0,
            random_state = 42
        ))
    ])
    final_ml_sev_model = TransformedTargetRegressor(
        regressor = final_base_ml_sev_model,
        func = np.log,
        inverse_func = np.exp
    )
    final_ml_sev_model.fit(sev_train_df[SEV_FEATURES], sev_train_df['TotalClaimAmount'] / sev_train_df['ClaimNb'], regressor__sample_weight = sev_train_df['ClaimNb'])

    pred_ml_sev_test = final_ml_sev_model.predict(sev_test_df[SEV_FEATURES])
    dev_ml_sev = mean_gamma_deviance(
        sev_test_df['TotalClaimAmount'] / sev_test_df['ClaimNb'],
        pred_ml_sev_test,
        sample_weight = sev_test_df['ClaimNb']
    )
    return final_ml_sev_model, dev_ml_sev


if __name__ == "__main__":

    combined_data = pd.read_csv('combined_data.csv')

    train_df, test_df = prepare_and_split_data_freq(combined_data)
    train_df, test_df, k, dev_cred = Buhlmann_straub(train_df, test_df)
    freq_model, dev_glm_freq = Poisson_GLM_Frequency_Model(train_df, test_df)
    sev_model, dev_glm_sev = Gamma_GLM_Severity_Model(train_df, test_df)
    ml_freq_model, dev_ml_freq = ML_Frequency_Model(train_df, test_df)
    ml_sev_model, dev_ml_sev = ML_Severity_Model(train_df, test_df)

    # Printing summary of results
    print("\n--- Model Performance Summary ---")
    print(f"Buhlmann Credibility Constant (K): {k:.2f}")
    print(f"Buhlmann Credibility Model - Mean Poisson Deviance: {dev_cred:.4f}")
    print(f"Frequency Model - Mean Poisson Deviance: {dev_glm_freq:.4f}")
    print(f"Severity Model - Mean Gamma Deviance: {dev_glm_sev:.4f}")
    print(f"ML Frequency Model - Mean Poisson Deviance: {dev_ml_freq:.4f}")
    print(f"ML Severity Model - Mean Gamma Deviance: {dev_ml_sev:.4f}")
