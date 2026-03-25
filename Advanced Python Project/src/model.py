import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging


logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_data():
    print("Step 1: Loading cleaned data...")
    df = pd.read_csv('data/processed/cleaned_data.csv')
    print(f"   Loaded {df.shape[0]} patients with {df.shape[1]} columns")
    logging.info("Data loaded for modeling")
    return df


def encode_data(df):
    print("\nStep 2: Converting text columns to numbers...")
    
  
    text_columns = df.select_dtypes(include='object').columns
    
   
    encoder = LabelEncoder()
    for column in text_columns:
        df[column] = encoder.fit_transform(df[column])
    
    print(f"   Converted {len(text_columns)} text columns to numbers")
    logging.info("Text columns encoded")
    return df

# ─── Step 3: Separate Features and Target ───────────────────
def separate_features_target(df):
    print("\nStep 3: Separating features and target...")
    
    # X = everything except readmitted column
    X = df.drop(columns=['readmitted'])
    
    # y = only the readmitted column (what we want to predict)
    y = df['readmitted']
    
    print(f"   Features (X): {X.shape[1]} columns")
    print(f"   Target (y): readmitted column")
    logging.info("Features and target separated")
    return X, y

# ─── Step 4: Scale Numbers ───────────────────────────────────
def scale_features(X):
    print("\nStep 4: Scaling all numbers to same range...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for later use
    with open('data/processed/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("   All features scaled successfully")
    logging.info("Features scaled")
    return X_scaled

# ─── Step 5: Split Into Train and Test ──────────────────────
def split_data(X, y):
    print("\nStep 5: Splitting data into train and test...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 20% for testing
        random_state=42,     # for reproducibility
        stratify=y           # keep same ratio of 0s and 1s
    )
    
    print(f"   Training patients: {X_train.shape[0]}")
    print(f"   Testing patients:  {X_test.shape[0]}")
    logging.info(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test

# ─── Step 6: Train Model ─────────────────────────────────────
def train_model(X_train, y_train):
    print("\nStep 6: Training Logistic Regression model...")
    print("   Please wait...")
    
    model = LogisticRegression(
        max_iter=1000,        # maximum attempts to learn
        random_state=42,      # for reproducibility
        class_weight='balanced' # handle imbalanced data
    )
    model.fit(X_train, y_train)
    
    print("   Model trained successfully!")
    logging.info("Model training completed")
    return model

# ─── Step 7: Test Model ──────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    print("\nStep 7: Testing model on unseen patients...")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Check accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n   Accuracy: {accuracy * 100:.2f}%")
    print("\n   Detailed Report:")
    print(classification_report(y_test, predictions,
          target_names=['Not Readmitted', 'Readmitted']))
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Readmitted', 'Readmitted'],
        yticklabels=['Not Readmitted', 'Readmitted']
    )
    plt.title('Confusion Matrix\n(How many predictions were correct?)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    plt.close()
    print("\n   Confusion Matrix saved to reports/!")
    logging.info(f"Model accuracy: {accuracy:.4f}")
    return accuracy


def save_model(model):
    print("\nStep 8: Saving trained model...")
    with open('data/processed/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("   Model saved to data/processed/model.pkl!")
    logging.info("Model saved successfully")

if __name__ == "__main__":
    df = load_data()
    df = encode_data(df)
    X, y = separate_features_target(df)
    X = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    print("\n🎉 Model Pipeline Complete!")
