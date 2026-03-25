import pandas as pd
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
import logging
import os
from datetime import datetime

# ─── Setup Logging ───────────────────────────────────────────
logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ─── Step 1: Load All Results ────────────────────────────────
def load_results():
    print("Step 1: Loading all results...")

    df = pd.read_csv('data/processed/cleaned_data.csv')
    summary = pd.read_csv('data/processed/summary_statistics.csv')
    correlations = pd.read_csv('data/processed/correlations.csv')

    print("   All results loaded successfully")
    logging.info("Results loaded for report generation")
    return df, summary, correlations

# ─── Step 2: Generate Excel Report ──────────────────────────
def generate_excel_report(df, summary, correlations):
    print("\nStep 2: Generating Excel report...")

    excel_path = 'reports/patient_analysis_report.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # Sheet 1 — Summary Statistics
        summary.to_excel(writer, sheet_name='Summary Statistics', index=False)

        # Sheet 2 — Correlation Results
        correlations.to_excel(writer, sheet_name='Correlations', index=False)

        # Sheet 3 — Readmission by Age
        age_readmission = df.groupby('age')['readmitted'].mean() * 100
        age_readmission = age_readmission.reset_index()
        age_readmission.columns = ['Age Group', 'Readmission Rate (%)']
        age_readmission.to_excel(
            writer,
            sheet_name='Readmission By Age',
            index=False
        )

        # Sheet 4 — Model Performance
        model_performance = pd.DataFrame({
            'Metric': [
                'Total Patients',
                'Readmission Rate',
                'Model Accuracy',
                'Training Patients',
                'Testing Patients'
            ],
            'Value': [
                len(df),
                f"{df['readmitted'].mean() * 100:.2f}%",
                '66.89%',
                '81,412',
                '20,354'
            ]
        })
        model_performance.to_excel(
            writer,
            sheet_name='Model Performance',
            index=False
        )

    print(f"   Excel report saved to {excel_path}")
    logging.info("Excel report generated successfully")

# ─── Step 3: Generate PDF Report ────────────────────────────
def generate_pdf_report(df):
    print("\nStep 3: Generating PDF report...")

    pdf_path = 'reports/patient_analysis_report.pdf'
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    # ── Title ──────────────────────────────────────────────
    title = Paragraph(
        "🏥 Patient Readmission Risk Analysis Report",
        styles['Title']
    )
    content.append(title)
    content.append(Spacer(1, 20))

    # ── Date ───────────────────────────────────────────────
    date = Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Normal']
    )
    content.append(date)
    content.append(Spacer(1, 20))

    # ── Section 1: Project Overview ────────────────────────
    content.append(Paragraph("1. Project Overview", styles['Heading1']))
    content.append(Spacer(1, 10))
    overview = Paragraph(
        "This report presents an end-to-end analysis of diabetic patient "
        "readmission risk using real hospital data from 130 US hospitals. "
        "The goal is to predict whether a patient will be readmitted within "
        "30 days of discharge using machine learning.",
        styles['Normal']
    )
    content.append(overview)
    content.append(Spacer(1, 20))

    # ── Section 2: Dataset Summary ─────────────────────────
    content.append(Paragraph("2. Dataset Summary", styles['Heading1']))
    content.append(Spacer(1, 10))

    dataset_data = [
        ['Metric', 'Value'],
        ['Total Patients', f"{len(df):,}"],
        ['Total Features', str(df.shape[1])],
        ['Readmitted Patients', f"{df['readmitted'].sum():,}"],
        ['Not Readmitted', f"{(df['readmitted'] == 0).sum():,}"],
        ['Readmission Rate', f"{df['readmitted'].mean() * 100:.2f}%"],
        ['Average Hospital Stay', f"{df['time_in_hospital'].mean():.2f} days"],
        ['Average Medications', f"{df['num_medications'].mean():.2f}"],
    ]

    table = Table(dataset_data, colWidths=[250, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightblue]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    content.append(table)
    content.append(Spacer(1, 20))

    # ── Section 3: Key Findings ────────────────────────────
    content.append(Paragraph("3. Key Findings", styles['Heading1']))
    content.append(Spacer(1, 10))

    findings = [
        "• Patients with more inpatient visits are most likely to be readmitted",
        "• Readmitted patients take significantly more medications (p-value < 0.05)",
        "• Average hospital stay is 4.40 days across all patients",
        "• Female patients slightly outnumber male patients in this dataset",
        "• Only 11.16% of patients were readmitted within 30 days",
    ]
    for finding in findings:
        content.append(Paragraph(finding, styles['Normal']))
        content.append(Spacer(1, 8))

    content.append(Spacer(1, 20))

    # ── Section 4: Model Performance ──────────────────────
    content.append(Paragraph("4. Model Performance", styles['Heading1']))
    content.append(Spacer(1, 10))

    model_data = [
        ['Metric', 'Value'],
        ['Model Used', 'Logistic Regression'],
        ['Training Patients', '81,412'],
        ['Testing Patients', '20,354'],
        ['Overall Accuracy', '66.89%'],
        ['Recall (Readmitted)', '52%'],
    ]

    model_table = Table(model_data, colWidths=[250, 200])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightblue]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    content.append(model_table)
    content.append(Spacer(1, 20))

    # ── Section 5: Charts ──────────────────────────────────
    content.append(Paragraph("5. Visualizations", styles['Heading1']))
    content.append(Spacer(1, 10))

    # Add saved charts
    charts = [
        ('reports/readmission_by_age.png', 'Readmission Rate by Age Group'),
        ('reports/confusion_matrix.png', 'Model Confusion Matrix'),
        ('reports/medication_distribution.png', 'Medication Distribution'),
    ]

    for chart_path, chart_title in charts:
        if os.path.exists(chart_path):
            content.append(Paragraph(chart_title, styles['Heading2']))
            content.append(Spacer(1, 10))
            img = Image(chart_path, width=450, height=280)
            content.append(img)
            content.append(Spacer(1, 20))

    # ── Build PDF ──────────────────────────────────────────
    doc.build(content)
    print(f"   PDF report saved to {pdf_path}")
    logging.info("PDF report generated successfully")

# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    df, summary, correlations = load_results()
    generate_excel_report(df, summary, correlations)
    generate_pdf_report(df)
    print("\n🎉 Reports Generated Successfully!")
    print("   Check your reports/ folder!")

