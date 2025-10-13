# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Config / filenames
RAW_FILE = "raw_dataset.csv"
CLEANED_FILE = "final_cleaned_dataset.csv"
PREDICTIONS_FILE = "predicted_risks.csv"
MODEL_FILE = "rf_model.joblib"  # saved after training

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def clean_data(df):
    # Basic cleaning steps — customize to your dataset
    df = df.copy()

    # 1) Standardize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # 2) Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 3) Handle missing values: numeric -> median, categorical -> mode
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # If 'target' exists keep it
    target_col = 'target' if 'target' in df.columns else None

    # Impute numerics
    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Impute categoricals
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown")

    # 4) Convert common categorical columns if present
    # Example: sex could be 'M'/'F' -> 0/1
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'M':1, 'F':0, 'Male':1, 'Female':0}).fillna(df['sex'])

    # 5) If any column is object but looks numeric, convert
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    return df

def prepare_features(df):
    df = df.copy()
    # If 'target' present, separate it
    y = df['target'] if 'target' in df.columns else None
    X = df.drop(columns=['target']) if 'target' in df.columns else df.copy()

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X, y, scaler

def train_model(X, y):
    # Simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def predict_and_save(model, scaler, X_original, df_cleaned):
    # X_original should be the same features (after get_dummies) used in training
    preds = model.predict(X_original)
    probas = model.predict_proba(X_original)[:,1] if hasattr(model, "predict_proba") else None

    # Map to readable risk labels
    risk_label = np.where(preds==1, "High", "Low")

    out = df_cleaned.copy()
    out['Predicted_Label'] = preds
    out['Risk'] = risk_label
    if probas is not None:
        out['Risk_Probability'] = probas

    out.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Saved predictions to {PREDICTIONS_FILE}")

def main():
    # 0. Load raw dataset
    df_raw = load_data(RAW_FILE)

    # 1. Clean dataset
    df_cleaned = clean_data(df_raw)
    df_cleaned.to_csv(CLEANED_FILE, index=False)
    print(f"Saved cleaned dataset to {CLEANED_FILE}")

    # 2. Prepare features
    X, y, scaler = prepare_features(df_cleaned)

    # 3. If target exists, train a model. Otherwise, prompt to provide a trained model
    if y is not None:
        model = train_model(X, y)
        # Save model and scaler for later use
        joblib.dump(model, MODEL_FILE)
        print(f"Saved model to {MODEL_FILE}")
    else:
        # If no target column, try to load an existing model
        try:
            model = joblib.load(MODEL_FILE)
            print(f"Loaded model from {MODEL_FILE}")
        except Exception as e:
            raise RuntimeError("No target column in raw data and no saved model found.") from e

    # 4. Predict on the full cleaned dataset (use X)
    predict_and_save(model, scaler, X, df_cleaned)

if __name__ == "__main__":
    main()
