import pandas as pd
import sqlite3
import logging
import os

logging.basicConfig(
    filename='logs/pipeline.log',
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'

)
def load_data(filepath):
    logging.info("loading data from: " + filepath)
    df = pd.read_csv(filepath)
    logging.info("Data loaded successfully")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    df.to_excel('my_analysis/raw_data.xlsx',index= False)
    print(" Raw data exported to my_analysis/raw_data.xlsx")
    return df

def show_basic_info(df):
    logging.info("showing the basic dataset info")
    print("\n Dataset Info")
    print(df.info())
    print("\n first 5 rows :")
    print(df.head())


def save_to_database(df, db_path='data/processed/hospital.db'):
    logging.info("Saving data to SQLite database")
    conn = sqlite3.connect(db_path)
    df.to_sql('diabetes_raw', conn, if_exists='replace', index=False)
    conn.close()
    logging.info("Data saved to database successfully")
    print("\n Data Saved to Database!")

if __name__ == "__main__":
    df = load_data('data/raw/diabetic_data.csv')
    show_basic_info(df)
    save_to_database(df)
