import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os

# ─── 1. Setup Logging ───────────────────────────────────────
logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ─── 2. Load Cleaned Data ────────────────────────────────────
def load_cleaned_data(filepath='data/processed/cleaned_data.csv'):
    logging.info("Loading cleaned data for visualization")
    df = pd.read_csv(filepath)
    print(f"✅ Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ─── 3. Readmission Rate by Age ──────────────────────────────
def plot_readmission_by_age(df):
    logging.info("Plotting readmission rate by age group")

    age_readmission = df.groupby('age')['readmitted'].mean() * 100
    age_readmission = age_readmission.reset_index()
    age_readmission.columns = ['Age Group', 'Readmission Rate (%)']

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=age_readmission,
        x='Age Group',
        y='Readmission Rate (%)',
        palette='Blues_d'
    )
    plt.title('Readmission Rate by Age Group', fontsize=14)
    plt.xlabel('Age Group')
    plt.ylabel('Readmission Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/readmission_by_age.png')
    plt.close()
    print("✅ Readmission by Age plot saved!")
    logging.info("Readmission by age plot saved")

# ─── 4. Correlation Heatmap ──────────────────────────────────
def plot_correlation_heatmap(df):
    logging.info("Plotting correlation heatmap")

    numerical_df = df.select_dtypes(include='number')
    corr_matrix = numerical_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='coolwarm',
        linewidths=0.5
    )
    plt.title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('reports/correlation_heatmap.png')
    plt.close()
    print("✅ Correlation Heatmap saved!")
    logging.info("Correlation heatmap saved")

# ─── 5. Medication Distribution ──────────────────────────────
def plot_medication_distribution(df):
    logging.info("Plotting medication distribution")

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x='num_medications',
        hue='readmitted',
        bins=30,
        palette='Set1'
    )
    plt.title('Medication Distribution by Readmission Status', fontsize=14)
    plt.xlabel('Number of Medications')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('reports/medication_distribution.png')
    plt.close()
    print("✅ Medication Distribution plot saved!")
    logging.info("Medication distribution plot saved")

# ─── 6. Time in Hospital Boxplot ─────────────────────────────
def plot_time_in_hospital(df):
    logging.info("Plotting time in hospital boxplot")

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x='readmitted',
        y='time_in_hospital',
        hue='readmitted',
    
        palette='Set2',
        legend=False
    )
    plt.title('Time in Hospital vs Readmission Status', fontsize=14)
    plt.xlabel('Readmitted (0=No, 1=Yes)')
    plt.ylabel('Time in Hospital (days)')
    plt.tight_layout()
    plt.savefig('reports/time_in_hospital.png')
    plt.close()
    print("✅ Time in Hospital plot saved!")
    logging.info("Time in hospital plot saved")

# ─── 7. Interactive Plotly Dashboard ─────────────────────────
def plot_interactive_dashboard(df):
    logging.info("Creating interactive Plotly dashboard")

    # Readmission by age — interactive
    age_readmission = df.groupby('age')['readmitted'].mean() * 100
    age_readmission = age_readmission.reset_index()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Readmission Rate by Age',
            'Time in Hospital Distribution',
            'Gender Distribution',
            'Inpatient Visits vs Readmission'
        )
    )

    # Plot 1 — Readmission by Age
    fig.add_trace(
        go.Bar(
            x=age_readmission['age'],
            y=age_readmission['readmitted'],
            name='Readmission Rate',
            marker_color='steelblue'
        ),
        row=1, col=1
    )

    # Plot 2 — Time in Hospital
    fig.add_trace(
        go.Histogram(
            x=df['time_in_hospital'],
            name='Time in Hospital',
            marker_color='coral'
        ),
        row=1, col=2
    )

    # Plot 3 — Gender Distribution
    gender_counts = df['gender'].value_counts()
    fig.add_trace(
        go.Bar(
            x=gender_counts.index,
            y=gender_counts.values,
            name='Gender',
            marker_color='mediumseagreen'
        ),
        row=2, col=1
    )

    # Plot 4 — Inpatient Visits vs Readmission
    fig.add_trace(
        go.Box(
            x=df['readmitted'].astype(str),
            y=df['number_inpatient'],
            name='Inpatient Visits',
            marker_color='mediumpurple'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        title_text="🏥 Patient Readmission Interactive Dashboard",
        showlegend=False
    )

    fig.write_html('reports/dashboard.html')
    print(" Interactive Dashboard saved to reports/dashboard.html!")
    logging.info("Interactive dashboard saved")

# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_cleaned_data()
    plot_readmission_by_age(df)
    plot_correlation_heatmap(df)
    plot_medication_distribution(df)
    plot_time_in_hospital(df)
    plot_interactive_dashboard(df)
    print("\n🎉 All Visualizations Complete!")
    print("   Check your reports/ folder!")
