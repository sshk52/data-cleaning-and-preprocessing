##clean\_and\_preprocess.py (titanic dataset)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load raw dataset
df_raw = pd.read_csv('Titanic-Dataset.csv')

# 1. Inspect dimensions and memory allocation
print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# 2. Check for duplicate passenger entries and drop the duplicates
print(f"Duplicate Rows: {df_raw.duplicated().sum()}")
df_raw.drop_duplicates(inplace=True)
print(f"Updated Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")  # no change in dataset as 0 duplicates

# Create working df
df = df_raw.copy()

# 3. Missing value scan across features
missing_summary = df.isnull().sum()
missing_percent = (missing_summary / len(df)) * 100
audit_df = pd.DataFrame({"Missing_Count": missing_summary, "Percentage (%)": missing_percent})
print(audit_df[audit_df["Missing_Count"] > 0])

# 3.a Handle missing values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Deck"] = df["Cabin"].str.slice(0, 1).fillna("U")

# 3.b Extract Honorific Titles & Impute Age by Title Median
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_mapping = {"Lady":"Rare", "Countess":"Rare", "Capt":"Rare", "Col":"Rare", "Don":"Rare",
                 "Dr":"Rare", "Major":"Rare", "Rev":"Rare", "Sir":"Rare", "Jonkheer":"Rare",
                 "Dona":"Rare", "Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"}
df["Title"] = df["Title"].replace(title_mapping)
df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

from google.colab import data_table
data_table.DataTable(df)

# Fixed: changed 'desribe(df)' to 'df.describe()'
print(df.describe())

# 4. Correct outliers with Tukey's 1.5x Interquartile Range (IQR) capping method
df_clean = df.copy()
Q1 = df_clean["Fare"].quantile(0.25)
Q3 = df_clean["Fare"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df_clean["Fare"] = np.where(df_clean["Fare"] > upper_bound, upper_bound, df_clean["Fare"])

# 5. Feature Engineering
df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
df_clean["IsAlone"] = (df_clean["FamilySize"] == 1).astype(int)

# 6. Drop Raw Unstructured Text Columns as unneccessary  (specification) for ml models
df_clean = df_clean.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# 7. One-Hot Encoding & Standardization to use these non int varible for a model
df_encoded = pd.get_dummies(df_clean, columns=["Sex", "Embarked", "Title", "Deck"], drop_first=True)

## scale these variable to make mean=0 and std =1 for all variables to make ml model unbiased weights to eachvariable
scaler = StandardScaler()
df_encoded[["Age", "Fare", "FamilySize"]] = scaler.fit_transform(df_encoded[["Age", "Fare", "FamilySize"]])
print("Preprocessing complete! Cleaned shape", df_encoded.shape)

from google.colab import data_table
data_table.DataTable(df_clean)
data_table.DataTable(df_encoded)







#functional form

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_titanic_data(filepath_or_url='Titanic-Dataset.csv'):
    """
    Fully cleans and preprocesses the Titanic dataset.
    Handles missing values, engineering, outlier handling, encoding, and scaling.
    """
    # 1. Load the raw dataset
    df_raw = pd.read_csv(filepath_or_url)
    df_raw.drop_duplicates(inplace=True)
    df = df_raw.copy()

    # 2. Impute Embarked and Cabin/Deck
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Deck"] = df["Cabin"].str.slice(0, 1).fillna("U")

    # 3. Handle honorific title extraction and Age imputation
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {"Lady":"Rare", "Countess":"Rare", "Capt":"Rare", "Col":"Rare", "Don":"Rare",
                     "Dr":"Rare", "Major":"Rare", "Rev":"Rare", "Sir":"Rare", "Jonkheer":"Rare",
                     "Dona":"Rare", "Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"}
    df["Title"] = df["Title"].replace(title_mapping)
    df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

    # 4. Outlier capping (Tukey's IQR method)
    df_clean = df.copy()
    Q1 = df_clean["Fare"].quantile(0.25)
    Q3 = df_clean["Fare"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df_clean["Fare"] = np.where(df_clean["Fare"] > upper_bound, upper_bound, df_clean["Fare"])

    # 5. Feature Engineering
    df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
    df_clean["IsAlone"] = (df_clean["FamilySize"] == 1).astype(int)

    # 6. Drop unneeded columns
    df_clean = df_clean.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # 7. One-Hot Encoding
    df_encoded = pd.get_dummies(df_clean, columns=["Sex", "Embarked", "Title", "Deck"], drop_first=True)

    # 8. Feature Standardization
    scaler = StandardScaler()
    df_encoded[["Age", "Fare", "FamilySize"]] = scaler.fit_transform(df_encoded[["Age", "Fare", "FamilySize"]])

    return df_clean, df_encoded
df_clean, df_processed = preprocess_titanic_data('Titanic-Dataset.csv')
display(df_processed.head())

# Example of using your new code snippet:
# df_clean, df_processed = preprocess_titanic_data(Titanic-Dataset.csv')
# display(df_processed.head())