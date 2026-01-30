import pandas as pd
import joblib
from feature_prep import prepare_features

# ===============================
# Load datasets
# ===============================
train_real = pd.read_csv('data/raw/train_real.csv')
train_synthetic = pd.read_csv('data/raw/train_synthetic.csv')
val_data = pd.read_csv('data/raw/validation.csv')

# ===============================
# TRAINING PIPELINE
# ===============================
train_outputs, artifacts = prepare_features(
    df_real=train_real,
    df_synthetic=train_synthetic,
    fit=True
)

train_trees = train_outputs['tree_models']   # For RF / Isolation Forest
train_dl = train_outputs['deep_learning']    # For Deep Learning

print("Train (trees):", train_trees.shape)
print("Train (DL):", train_dl.shape)

# ===============================
# VALIDATION PIPELINE
# ===============================
encoders = joblib.load('src/backend/feature_engineering/encoders/label_encoders.joblib')
scaler = joblib.load('src/backend/feature_engineering/scalers/scaler.joblib')
power_transformer = joblib.load('src/backend/feature_engineering/scalers/power_transformer.joblib')

val_outputs, _ = prepare_features(
    df_real=val_data,
    fit=False,
    encoders=encoders,
    scaler=scaler,
    power_transformer=power_transformer
)

val_trees = val_outputs['tree_models']
val_dl = val_outputs['deep_learning']

print("Val (trees):", val_trees.shape)
print("Val (DL):", val_dl.shape)
