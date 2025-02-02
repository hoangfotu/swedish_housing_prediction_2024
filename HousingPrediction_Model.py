# LOAD LIBRARIES
from sqlalchemy.engine import create_engine
import pymysql
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.model_selection import TimeSeriesSplit
import sklearn.metrics as metrics
from xgboost import XGBRegressor
import numpy as np

# CONNECT TO SQL AND GET HOUSING DATA - HIDDEN
host = ""
username = ""
password = ""
schema = ""

# Format replaces {} in the string with the respective inputs to the function
connection_string = "mysql+pymysql://{}:{}@{}/{}".format(
    username, password, host, schema)

# This creates a connection from the string we just created
connection = create_engine(connection_string)

# Collect the all columns from the customer table
query = "SELECT id, sell_date, sell_price, rooms, postcode, object_type, \
    living_area, rent, housing_association_org_number, street_name, \
    asking_price, total_commercial_area, plot_is_leased, association_tax_liability,\
    longitude, latitude, YEAR(sell_date) AS year_sold \
    FROM Apartment LEFT JOIN HousingAssociation ON Apartment.housing_association_org_number \
    = HousingAssociation.org_number LEFT JOIN  AnnualReport ON HousingAssociation.org_number \
    = AnnualReport.org_number AND YEAR(sell_date) - 1 = AnnualReport.fiscal_year"

# Use the connection to run SQL query and store it into a dataframe
df = pd.read_sql_query(con=connection.connect(), sql=query)

# VALIDATION OF NUMERICAL FEATURES
# Numeric columns to validate
cont_columns = ['living_area', 'rent', 'total_commercial_area', 'asking_price', 'rooms']
num_columns = cont_columns + ['latitude', 'longitude']

# Dictionary to accumulate indices and values taken out
invalid_numeric_data = {}

# Check if numeric columns contain only 'int64' or 'float64' data types
valid_types = (int, float)
for col in num_columns:
    invalid_numeric_data[col] = [{'index': idx, 'value': value} for idx, value in df.loc[~df[col].apply(lambda value: isinstance(value, valid_types)), col].items() if pd.notna(value)]

# Raise TypeError with specific column name if there are non-numeric values
if any(invalid_numeric_data.values()):
    raise TypeError(f"Error: Columns contain values that are not int or float at indexes: {invalid_numeric_data}")

# List to accumulate indices and values with negative values
negative_data = {}

# Check if continuous columns contain only weakly positive values
for col in cont_columns:
    negative_data[col] = [{'index': idx, 'value': value} for idx, value in df[df[col] < 0][col].items() if pd.notna(value)]

# Raise ValueError with specific column name if there are negative values
if any(negative_data.values()):
    raise ValueError(f"Error: Continuous columns contain negative values at indexes: {negative_data}")

# List to accumulate indices and values with out of range values
invalid_lat_data = {}

# Check if continuous columns contain only in range values
invalid_lat_data['latitude'] = [{'index': idx, 'value': value} for idx, value in df.loc[(df['latitude'] < -90) | (df['latitude'] > 90), 'latitude'].items() if pd.notna(value)]

# Raise TypeError if there are incorrect values in the latitude column
if invalid_lat_data['latitude']:
    raise ValueError(f"Error: Column 'latitude' contains out-of-range values at indexes: {invalid_lat_data['latitude']}")

# List to accumulate indices and values with out of range values
invalid_long_data = {}

# Check if continuous columns contain only in range values
invalid_long_data['longitude'] = [{'index': idx, 'value': value} for idx, value in df.loc[(df['longitude'] < -180) | (df['longitude'] > 180), 'longitude'].items() if pd.notna(value)]

# Raise ValueError if there are incorrect values in the longitude column
if invalid_long_data['longitude']:
    raise ValueError(f"Error: Column 'longitude' contains out-of-range values at indexes: {invalid_long_data['longitude']}")

# List to accumulate indices and values with invalid postcode values
invalid_post_data = {}

# Check if postcode column contain only positive values
invalid_post_data['postcode'] = [{'index': idx, 'value': value} for idx, value in df.loc[(df['postcode'] <= 0), 'postcode'].items() if pd.notna(value)]

# Raise ValueError if there are incorrect values in the postcode column
if invalid_post_data['postcode']:
    raise ValueError(f"Error: Column 'postcode' contains out-of-range values at indexes: {invalid_post_data['postcode']}")

# Confirmation if there are no errors in numerical variables
print("No data range or type errors for numerical variables.")

# VALIDATION OF BINARY FEATURES
# Dictionary to accumulate indices and values taken out
invalid_bin_data = {}

# Check if plot is leased dummy column contains only 1 or 0 (ignoring NaNs)
valid_bin_values = {0, 1}
invalid_bin_data['plot_is_leased'] = [{'index': idx, 'value': value} for idx, value in df.loc[~df['plot_is_leased'].isin(valid_bin_values), 'plot_is_leased'].items() if pd.notna(value)]

# Raise TypeError if there are incorrect values in the binary column
if invalid_bin_data['plot_is_leased']:
    raise TypeError(f"Error: Column 'plot_is_leased' contains values other than 0 or 1 at indexes: {invalid_bin_data['plot_is_leased']}")

# Confirmation if there are no errors in the binary data
print("No data range or type errors for binary variables.")

# VALIDATION OF CATEGORICAL FEATURES
cat_columns = ['association_tax_liability']

# List to accumulate columns with non-numeric data types
non_category_columns = []

# Check if columns contain only non-numeric data types
invalid_types = (int, float)
for col in cat_columns:
    invalid_data = [{'index': idx, 'value': value} for idx, value in df.loc[df[col].apply(lambda v: isinstance(v, invalid_types)), col].items() if pd.notna(value)]
    if invalid_data:
        non_category_columns.append({col: invalid_data})

# Raise TypeError with specific column names if there are non-category values
if non_category_columns:
    raise TypeError(f"Error: Columns contain values that are not categorical: {non_category_columns}")

# Confirmation if there are no errors in the categorical data
print("No data type errors for category variables.")

# FILL MISSING VALUES
# HOUSING ASSOCIATION NUMBER
# Calculate the mode of housing association by each street
mode_housing = df.groupby('street_name')['housing_association_org_number'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

# Write a function to impute housing association
def impute_housing(row):
    if pd.isnull(row['housing_association_org_number']):
        return mode_housing.get(row['street_name'], row['housing_association_org_number'])
    return row['housing_association_org_number']

# Apply the imputation function to the dataframe
df['housing_association_org_number'] = df.apply(impute_housing, axis=1)

# ANNUAL REPORTS
# Sort the dataframe by group and year
df = df.sort_values(by=['housing_association_org_number', 'year_sold'])

columns_to_fill = ['association_tax_liability', 'total_commercial_area', 'plot_is_leased']  # Add all columns to be backfilled

# Frontward fill missing values within each group, then backfill for those still missing
df[columns_to_fill] = df.groupby('housing_association_org_number')[columns_to_fill].ffill()
df[columns_to_fill] = df.groupby('housing_association_org_number')[columns_to_fill].bfill()

# DROP OUTLIERS BEFORE MEAN CALCULATIONS
# Drop any objects that are not apartments
df.drop(df[df['object_type'] != 'Apartment'].index, axis=0, inplace= True)

#Outlier control of numerical features
def winsorize(data, columns):
    for col in columns:
        column_data = data[col]
        lb = np.nanpercentile(column_data, 0.01)
        ub = np.nanpercentile(column_data, 99.9)
        data.loc[data[col] < lb, col] = lb
        data.loc[data[col] > ub, col] = ub
    return data

# Replace values below the lower bound and above the upper bound with percentiles in the original dataframe
df = winsorize(df,cont_columns)

# ROOMS, LIVING AREA, RENT
# Sort the dataframe according to postcode
df = df.sort_values(by=['postcode'])

# Impute the missing values of rooms, living area and rent 
imputation_features = ['rooms', 'living_area', 'rent']

# Define the number of neighbors to look at
knn_imputer = KNNImputer(n_neighbors=10)  

# Fit and transform the imputer on the data
imputed_values = knn_imputer.fit_transform(df[imputation_features])

# Update the missing values with the imputations for each column
for i in imputation_features:
    df[i] = imputed_values[:, imputation_features.index(i)]

# IMPUTE MISSING VALUES FOR FORECASTING SET (See reasoning in the accompanying report)
# Set missing values in whether the plot is leased to its mode when the sell price is missing
df.loc[df['sell_price'].isnull(), 'plot_is_leased'] = 0

# Set missing values in the tax liability to its mode when the sell price is missing
df.loc[df['sell_price'].isnull(), 'association_tax_liability'] = df['association_tax_liability'].mode()[0]

# Extract features for imputation
imputation_features = ['rooms', 'living_area', 'postcode', 'asking_price']

# Define the number of neighbors to look at
knn_imputer = KNNImputer(n_neighbors=5)

# Fit and transform the imputer on your data
imputed_values = knn_imputer.fit_transform(df.loc[df['sell_price'].isnull(), imputation_features])

# Update the dataframe with the imputed values
df.loc[df['sell_price'].isnull(), 'asking_price'] = imputed_values[:, imputation_features.index('asking_price')]

# CREATE SEASONAL VARIABLES
# Convert the sell date to datetime format
df['date'] = pd.to_datetime(df['sell_date'])

# Extract year, quarter and month into separate columns
df['quarter'] = df['date'].dt.quarter
df['month'] = df['date'].dt.month

# RECODE
# Code commerical area to has commercial area or not
df['has_commercial_area'] = df['total_commercial_area'].apply(lambda x: 1 if (not pd.isnull(x) and x > 0) else 0)

# Make into categorical variables
df['postcode'] = df['postcode'].astype('string').str[:3].astype("category") # Recode postcode into larger local areas
df[['quarter']] = df[['quarter']].astype('category')
df[['month']] = df[['month']].astype('category')
df[['association_tax_liability']] = df[['association_tax_liability']].astype('category')

# Bin variables
df['rooms'] = pd.cut(df['rooms'], bins=[1, 2, 3, 4, 5, 6, float('inf')], labels=['1', '2', '3', '4', '5', 'Above 5'], right=False, include_lowest=True).astype('category')

# Create dummies of the categorical variables
dummies = pd.get_dummies(df[['postcode', 'rooms', 'quarter', 'month', 'association_tax_liability']], dummy_na=False, drop_first=True)
final = pd.concat([df, dummies], axis='columns')

# Put names of all dummies into lists
rooms_columns = [col for col in final.columns if col.startswith('rooms_')]
postcode_columns = [col for col in final.columns if col.startswith('postcode_')]
quarter_columns = [col for col in final.columns if col.startswith('quarter_')]
month_columns = [col for col in final.columns if col.startswith('month_')]
tax_columns = [col for col in final.columns if col.startswith('association_tax_liability_')]

# SPLIT INTO THE DIFFERENT DATASETS
# XGBoost
df_tree = final.copy()

# Forecasting data
forecasting = final[final['sell_price'].isnull()]

# DROPPING MISSING VALUES
df_tree = df_tree.dropna(subset=['sell_price', 'has_commercial_area', 'living_area', 'asking_price', 'rent', 'longitude', 'latitude', 'plot_is_leased', 'rooms','association_tax_liability', 'quarter'])

# RUN A CHALLENGER MODEL
# Features (X) and target variable (y)
features = df_tree[['living_area'] + ['plot_is_leased'] + ['asking_price'] + ['has_commercial_area'] + ['rent'] + ['longitude'] + ['latitude'] + postcode_columns + month_columns + tax_columns + rooms_columns + quarter_columns]
target = df_tree['sell_price']

# Create train/test set based on a time split 80/20
tss = TimeSeriesSplit(5)
df_sorted = df_tree.sort_values(by='date')
for train_index, test_index in tss.split(features):
    x_train, x_test = features.iloc[train_index, :], features.iloc[test_index,:]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

# Fit the model and predict values based on the best parameters in the random search
model = XGBRegressor(
    learning_rate=0.1,
    max_depth=3,
    reg_lambda=1,
    alpha=10
)

# Fit the model
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# Evaluate how the model performs
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# FORECAST SELL PRICES BASED ON THE BEST MODEL
# Take out the features to forecast
forecasting_features = forecasting[['living_area'] + ['plot_is_leased'] + ['asking_price'] + ['has_commercial_area'] + ['rent'] + ['longitude'] + ['latitude'] + postcode_columns + month_columns + tax_columns + rooms_columns + quarter_columns]

# Forecast apartment sell prices
forecast_sell_price = model.predict(forecasting_features)

# Convert into an integer
forecast_sell_price = forecast_sell_price.astype(int)

# Use the index from forecasting_features to fetch corresponding id from df_linear
id_column = df.loc[forecasting_features.index, 'id']

# Create a dataframe with id and predicted sell prices
predicted_sell_prices_df = pd.DataFrame({'id': id_column, 'predicted_sell_price': forecast_sell_price})

# Create a JSON-file with the final output
predicted_sell_prices_df.to_json("assignment2_24879.json", orient="records", indent=2)