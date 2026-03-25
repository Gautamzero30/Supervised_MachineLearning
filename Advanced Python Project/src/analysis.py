import pandas as pd
import numpy as np
from scipy import stats
import logging


logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_cleaned_data(filepath='data/processed/cleaned_data.csv'):
    logging.info("Loading cleaned data for analysis")
    df = pd.read_csv(filepath)
    print(f" Cleaned Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def descriptive_statistics(df):
    logging.info("Running descriptive statistics")

    print("\n Basic Statistical Summary:")
    print(df.describe())

    print("\n Gender Distribution:")
    print(df['gender'].value_counts())  ##count how many times each unique value appears in a column.

    print("\n Average Time in Hospital:")
    print(f"   Mean: {df['time_in_hospital'].mean():.2f} days")
    print(f"   Median: {df['time_in_hospital'].median():.2f} days")

    print("\n  Average Medications Per Patient:")
    print(f"   Mean: {df['num_medications'].mean():.2f}")
    print(f"   Median: {df['num_medications'].median():.2f}")

    print("\n Readmission Rate:")
    rate = df['readmitted'].mean() * 100
    print(f"   {rate:.2f}% of patients readmitted within 30 days")

    logging.info("Descriptive statistics completed")


def correlation_analysis(df):
    logging.info("Running correlation analysis")

    print("\n Top Correlations with Readmission:")
    numerical_df = df.select_dtypes(include='number')
    correlations = numerical_df.corr()['readmitted'].sort_values(ascending=False)
    print(correlations)

  
    correlations.to_csv('data/processed/correlations.csv')
    print("\n Correlations saved to data/processed/correlations.csv")
    logging.info("Correlation analysis completed")
    return correlations


def hypothesis_testing(df):
    logging.info("Running hypothesis testing")

    print("\n Hypothesis Test:")
    # "There is no difference." It assumes that readmitted and non-readmitted patients take roughly the same amount of medicine.
    print("   H0: No difference in medications between readmitted vs not")
    # "There is no difference." It assumes that readmitted and non-readmitted patients take different the same amount of medicine.
    print("   H1: Readmitted patients take more medications")

    readmitted = df[df['readmitted'] == 1]['num_medications']
    not_readmitted = df[df['readmitted'] == 0]['num_medications']

 
    t_stat, p_value = stats.ttest_ind(readmitted, not_readmitted)

    print(f"\n   T-statistic: {t_stat:.4f}")
    print(f"   P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("    Result: SIGNIFICANT difference found!")
        print("   Readmitted patients DO take more medications")
    else:
        print("    Result: No significant difference found")

    logging.info(f"Hypothesis test completed. P-value: {p_value:.4f}")


def skewness_kurtosis(df):
    logging.info("Calculating skewness and kurtosis")

    print("\n Skewness and Kurtosis of Key Features:")
    cols = ['time_in_hospital', 'num_medications', 'num_lab_procedures']
    for col in cols:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        print(f"\n   {col}:")
        print(f"   Skewness: {skew:.4f}")
        print(f"   Kurtosis: {kurt:.4f}")

    logging.info("Skewness and kurtosis calculated")


def save_summary_statistics(df):
    logging.info("Saving summary statistics")
    summary = df.describe()
    summary.to_csv('data/processed/summary_statistics.csv')
    print("\n  Summary Statistics Saved!")
    logging.info("Summary statistics saved successfully")


if __name__ == "__main__":
    df = load_cleaned_data()
    descriptive_statistics(df)
    correlation_analysis(df)
    hypothesis_testing(df)
    skewness_kurtosis(df)
    save_summary_statistics(df)
