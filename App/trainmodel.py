import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Charger le dataset d'entraînement (KDDTrain ou équivalent)
data = pd.read_csv("KDDTrain+.csv", header=None)

# Noms des colonnes NSL-KDD
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

data.columns = columns

# X / y
X = data.drop(columns=["label", "difficulty"])
y = data["label"].apply(lambda x: 0 if x == "normal" else 1)

# Colonnes
cat_cols = ["protocol_type", "service", "flag"]
num_cols = X.columns.difference(cat_cols)

# Prétraitement
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Pipeline FINAL
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# Entraînement
pipeline.fit(X, y)

# Sauvegarde
joblib.dump(pipeline, "ids_pipeline.pkl")

print("✅ Modèle entraîné et sauvegardé")