import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Paths
# ===============================
BASE_DIR = "src/backend/feature_engineering"
ENCODER_DIR = os.path.join(BASE_DIR, "encoders")
SCALER_DIR = os.path.join(BASE_DIR, "scalers")
PROCESSED_DIR = "data/processed"

os.makedirs(ENCODER_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ===============================
# MAIN FEATURE PIPELINE
# ===============================
def prepare_features(df_real, df_synthetic=None, fit=True, encoders=None, scaler=None, power_transformer=None):
    """
    Returns:
        {
            'tree_models': unscaled features,
            'deep_learning': scaled features
        }
    """

    # -------------------------------
    # 1. Combine datasets
    # -------------------------------
    if df_synthetic is not None:
        df = pd.concat([df_real, df_synthetic], axis=0).reset_index(drop=True)
    else:
        df = df_real.copy()

    # -------------------------------
    # 2. Feature Engineering
    # -------------------------------
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['ratio_feature'] = df['feature1'] / (df['feature2'] + 1e-5)

    if 'bytes_in' in df.columns and 'bytes_out' in df.columns:
        df['total_bytes'] = df['bytes_in'] + df['bytes_out']

    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

    # -------------------------------
    # 3. Encode Categorical Variables
    # -------------------------------
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, os.path.join(ENCODER_DIR, "label_encoders.joblib"))
    else:
        for col in cat_cols:
            if col in encoders:
                df[col] = encoders[col].transform(df[col].astype(str))

    # -------------------------------
    # 🔥 Branch for model types
    # -------------------------------
    df_trees = df.copy()  # Unscaled for RF / Isolation Forest

    # -------------------------------
    # 4. Transform + Scale (Deep Learning only)
    # -------------------------------
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if fit:
        power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        df[num_cols] = power_transformer.fit_transform(df[num_cols])

        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        joblib.dump(power_transformer, os.path.join(SCALER_DIR, "power_transformer.joblib"))
        joblib.dump(scaler, os.path.join(SCALER_DIR, "scaler.joblib"))
    else:
        df[num_cols] = power_transformer.transform(df[num_cols])
        df[num_cols] = scaler.transform(df[num_cols])

    # -------------------------------
    # 5. Save processed training data
    # -------------------------------
    if fit:
        df_trees.to_csv(os.path.join(PROCESSED_DIR, "train_features_trees.csv"), index=False)
        df.to_csv(os.path.join(PROCESSED_DIR, "train_features_dl.csv"), index=False)

    outputs = {
        "tree_models": df_trees,
        "deep_learning": df
    }

    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "power_transformer": power_transformer
    } if fit else None

    return outputs, artifacts


# ===============================
# TF-IDF TEXT FEATURES (Supervised ML)
# ===============================
def prepare_tfidf(df, text_col='text_feature', fit=True, tfidf_vectorizer=None):
    if fit:
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df[text_col])
        joblib.dump(tfidf, os.path.join(ENCODER_DIR, "tfidf_vectorizer.joblib"))

        # Save vocabulary for SHAP alignment
        vocab = tfidf.get_feature_names_out()
        joblib.dump(vocab, os.path.join(ENCODER_DIR, "tfidf_vocab.joblib"))
    else:
        tfidf = tfidf_vectorizer
        tfidf_matrix = tfidf.transform(df[text_col])

    return tfidf_matrix


# ===============================
# Isolation Forest Feature Frame
# ===============================
def save_iso_features(df, features):
    iso_df = df[features].copy()
    iso_df.fillna(0, inplace=True)
    joblib.dump(features, os.path.join(ENCODER_DIR, "iso_features_list.joblib"))
    return iso_df


# ===============================
# Visualization (Optional)
# ===============================
def visualize_features(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()
