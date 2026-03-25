import pandas as pd
import numpy as np
import sqlite3
import logging


logging.basicConfig(
    filename='logs/pipeline.log',
    level= logging.INFO,
    format = '%(asctime)s - %(levelname)s -%(message)s'
)

def load_raw_data(filepath='data/raw/diabetic_data.csv'):
    logging.info("loading raw data for cleaning")
    df = pd.read_csv(filepath)
    print(f". Raw Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def replace_question_marks(df):
    logging.info("Replacing ? with NaN")
    df = df.replace('?', np.nan)
    total_missing = df.isnull().sum()
    print(f"Replaced ? with NaN , Total Missing Values: {total_missing}")
    return df

def drop_useless_columns(df):
    logging.info("Dropping useless columns")
    cols_to_drop = [
        'weight',           # 97% missing and not used further
        'payer_code',       # not clinically relevant for the models prediction
        'medical_specialty',# too many missing and same as earlier
        'encounter_id',     # just an ID , no work further
        'patient_nbr',      # just an ID
        'examide',          # only one unique value,no work further
        'citoglipton'       # only one unique value,no work further
    ]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped Useless Columns | Remaining Columns: {df.shape[1]}")
    logging.info(f"Columns after dropping: {df.shape[1]}")
    return df

def handle_missing_values(df):
    logging.info("handling missing values")


    categorical_cols = df.select_dtypes (include="str").columns
    for col in categorical_cols:
        df[col]=df[col].fillna(df[col].mode()[0])


    numerical_cols = df.select_dtypes(include="number").columns
    for col in numerical_cols:
        df[col]=df[col].fillna(df[col].median())


    print(f"Missing value is handled , Remaining = {df.isnull().sum().sum()}")
    logging.info("missing values handled successfully")
    return df       

def clean_target_column(df):
    logging.info("Cleaning target column: readmitted")

    # Convert to binary: <30 days = 1 (bad), rest = 0
    df['readmitted'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else 0
    )
    count = df['readmitted'].value_counts()
    print(f"✅ Target Column Cleaned:")
    print(f"   Not Readmitted (0): {count[0]}")
    print(f"   Readmitted <30days (1): {count[1]}")
    logging.info("Target column cleaned successfully")
    return df


def save_cleaned_data(df):
    logging.info("Saving cleaned data")

   
    df.to_csv('data/processed/cleaned_data.csv', index=False)

    # Save to database
    conn = sqlite3.connect('data/processed/hospital.db')
    df.to_sql('diabetes_cleaned', conn, if_exists='replace', index=False)
    conn.close()

    print(f" Cleaned Data Saved!")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    logging.info("Cleaned data saved successfully")
    df.to_excel('my_analysis/cleaned_data.xlsx', index=False)
    print("✅ Cleaned data exported to my_analysis/cleaned_data.xlsx")

if __name__ == "__main__":
    df = load_raw_data()
    df = replace_question_marks(df)
    df = drop_useless_columns(df)
    df = handle_missing_values(df)
    df = clean_target_column(df)
    save_cleaned_data(df)
