import logging
from src.ingestion import load_data, show_basic_info, save_to_database
from src.cleaning import (
    load_raw_data,
    replace_question_marks,
    drop_useless_columns,
    handle_missing_values,
    clean_target_column,
    save_cleaned_data
)
from src.analysis import (
    load_cleaned_data,
    descriptive_statistics,
    correlation_analysis,
    hypothesis_testing,
    skewness_kurtosis,
    save_summary_statistics
)
from src.visualization import (
    load_cleaned_data as load_data_viz,
    plot_readmission_by_age,
    plot_correlation_heatmap,
    plot_medication_distribution,
    plot_time_in_hospital,
    plot_interactive_dashboard
)
from src.model import (
    load_data as load_data_model,
    encode_data,
    separate_features_target,
    scale_features,
    split_data,
    train_model,
    evaluate_model,
    save_model
)
from src.report import (
    load_results,
    generate_excel_report,
    generate_pdf_report
)

# ─── Setup Logging ───────────────────────────────────────────
logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ─── Main Pipeline ───────────────────────────────────────────
def run_pipeline():
    print("=" * 50)
    print("🏥 PATIENT READMISSION RISK PIPELINE")
    print("=" * 50)

    # ── Phase 1: Ingestion ────────────────────────────────
    print("\n📥 PHASE 1: DATA INGESTION")
    print("-" * 30)
    df_raw = load_data('data/raw/diabetic_data.csv')
    show_basic_info(df_raw)
    save_to_database(df_raw)

    # ── Phase 2: Cleaning ─────────────────────────────────
    print("\n🧹 PHASE 2: DATA CLEANING")
    print("-" * 30)
    df = load_raw_data()
    df = replace_question_marks(df)
    df = drop_useless_columns(df)
    df = handle_missing_values(df)
    df = clean_target_column(df)
    save_cleaned_data(df)

    # ── Phase 3: Analysis ─────────────────────────────────
    print("\n📊 PHASE 3: STATISTICAL ANALYSIS")
    print("-" * 30)
    df_analysis = load_cleaned_data()
    descriptive_statistics(df_analysis)
    correlation_analysis(df_analysis)
    hypothesis_testing(df_analysis)
    skewness_kurtosis(df_analysis)
    save_summary_statistics(df_analysis)

    # ── Phase 4: Visualization ────────────────────────────
    print("\n📈 PHASE 4: VISUALIZATION")
    print("-" * 30)
    df_viz = load_data_viz()
    plot_readmission_by_age(df_viz)
    plot_correlation_heatmap(df_viz)
    plot_medication_distribution(df_viz)
    plot_time_in_hospital(df_viz)
    plot_interactive_dashboard(df_viz)

    # ── Phase 5: Model ────────────────────────────────────
    print("\n🤖 PHASE 5: MACHINE LEARNING MODEL")
    print("-" * 30)
    df_model = load_data_model()
    df_model = encode_data(df_model)
    X, y = separate_features_target(df_model)
    X = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

    # ── Phase 6: Report ───────────────────────────────────
    print("\n📄 PHASE 6: REPORT GENERATION")
    print("-" * 30)
    df_report, summary, correlations = load_results()
    generate_excel_report(df_report, summary, correlations)
    generate_pdf_report(df_report)

    # ── Done ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\n📁 Check your reports/ folder for:")
    print("   • dashboard.html")
    print("   • patient_analysis_report.xlsx")
    print("   • patient_analysis_report.pdf")
    print("   • All chart images")
    print("\n📋 Check logs/pipeline.log for full history")

# ─── Entry Point ─────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
