import pickle
import numpy as np
import pandas as pd
import logging

# ─── Setup Logging ───────────────────────────────────────────
logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ─── Step 1: Load Saved Model and Scaler ────────────────────
def load_model_and_scaler():
    print("Loading model...")
    with open('data/processed/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model loaded successfully!\n")
    logging.info("Model and scaler loaded for prediction")
    return model, scaler

# ─── Step 2: Get Patient Details From Terminal ───────────────
def get_patient_details():
    print("=" * 50)
    print("🏥 PATIENT RISK ASSESSMENT FORM")
    print("=" * 50)
    print("Please enter patient details below:")
    print("-" * 50)

    # Collect patient information
    print("\n📋 Age Groups Available:")
    print("   1: [0-10]   2: [10-20]  3: [20-30]")
    print("   4: [30-40]  5: [40-50]  6: [50-60]")
    print("   7: [60-70]  8: [70-80]  9: [80-90]")
    print("   10: [90-100]")
    age = int(input("\nEnter age group number (1-10): "))

    print("\n📋 Gender:")
    print("   1: Male   2: Female")
    gender = int(input("Enter gender number: "))

    time_in_hospital = int(input(
        "\n📋 How many days was patient in hospital? (1-14): "
    ))

    num_medications = int(input(
        "\n📋 How many medications is patient taking? (1-81): "
    ))

    num_lab_procedures = int(input(
        "\n📋 How many lab procedures were done? (1-132): "
    ))

    number_inpatient = int(input(
        "\n📋 How many previous inpatient visits? (0-21): "
    ))

    number_emergency = int(input(
        "\n📋 How many emergency visits in past year? (0-76): "
    ))

    number_diagnoses = int(input(
        "\n📋 How many diagnoses does patient have? (1-16): "
    ))

    print("\n📋 Is patient on insulin?")
    print("   1: No   2: Steady   3: Up   4: Down")
    insulin = int(input("Enter insulin status (1-4): "))

    print("\n📋 Is patient on diabetes medication?")
    print("   1: Yes   2: No")
    diabetes_med = int(input("Enter option: "))

    logging.info("Patient details collected from terminal")

    return {
        'age': age,
        'gender': gender,
        'time_in_hospital': time_in_hospital,
        'num_medications': num_medications,
        'num_lab_procedures': num_lab_procedures,
        'number_inpatient': number_inpatient,
        'number_emergency': number_emergency,
        'number_diagnoses': number_diagnoses,
        'insulin': insulin,
        'diabetesMed': diabetes_med
    }

# ─── Step 3: Prepare Input for Model ────────────────────────
def prepare_input(patient_details, scaler):
    print("\nPreparing patient data for model...")

    # Load cleaned data to get all column names
    df = pd.read_csv('data/processed/cleaned_data.csv')
    feature_columns = df.drop(columns=['readmitted']).columns.tolist()

    # Create empty row with zeros
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    # Fill in patient details
    input_data['time_in_hospital'] = patient_details['time_in_hospital']
    input_data['num_medications'] = patient_details['num_medications']
    input_data['num_lab_procedures'] = patient_details['num_lab_procedures']
    input_data['number_inpatient'] = patient_details['number_inpatient']
    input_data['number_emergency'] = patient_details['number_emergency']
    input_data['number_diagnoses'] = patient_details['number_diagnoses']

    # Scale the input
    input_scaled = scaler.transform(input_data)

    logging.info("Patient input prepared for prediction")
    return input_scaled

# ─── Step 4: Make Prediction ─────────────────────────────────
def make_prediction(model, input_scaled, patient_details):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    print("\n" + "=" * 50)
    print("🏥 PATIENT RISK ASSESSMENT RESULT")
    print("=" * 50)

    if probability >= 60:
        risk_level = "🔴 HIGH RISK"
        recommendation = "Schedule follow-up within 7 days!"
    elif probability >= 30:
        risk_level = "🟡 MEDIUM RISK"
        recommendation = "Schedule follow-up within 14 days"
    else:
        risk_level = "🟢 LOW RISK"
        recommendation = "Regular follow-up schedule is fine"

    print(f"\nRisk Score:   {probability:.1f}%")
    print(f"Risk Level:   {risk_level}")
    print(f"\n📋 Key Factors Entered:")
    print(f"   • Days in Hospital:        {patient_details['time_in_hospital']}")
    print(f"   • Medications:             {patient_details['num_medications']}")
    print(f"   • Previous Inpatient Visits: {patient_details['number_inpatient']}")
    print(f"   • Emergency Visits:        {patient_details['number_emergency']}")
    print(f"   • Number of Diagnoses:     {patient_details['number_diagnoses']}")
    print(f"\n💊 Recommendation:")
    print(f"   {recommendation}")
    print("=" * 50)

    logging.info(f"Prediction made: {risk_level}, Score: {probability:.1f}%")
    return prediction, probability

# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    model, scaler = load_model_and_scaler()
    patient_details = get_patient_details()
    input_scaled = prepare_input(patient_details, scaler)
    make_prediction(model, input_scaled, patient_details)
