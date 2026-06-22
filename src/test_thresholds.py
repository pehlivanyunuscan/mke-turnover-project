# src/test_thresholds.py
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

workspace_src = "/Users/yunuscanpehlivan/Desktop/mke turnover/src"
sys.path.append(workspace_src)
from pipeline import MKEDataPipeline

df = pd.read_csv("data/dataset.csv")
df_beyaz = df[df["Yaka_Tipi"] == "Beyaz"].copy().drop(columns=["Yaka_Tipi"])

pipeline = MKEDataPipeline()
pipeline.fit(df_beyaz)
df_processed = pipeline.transform(df_beyaz)

categorical_cols = ["Medeni_Hal", "Ilce", "Seyahat_Sikligi", "Departman", "Unvan", "Egitim_Seviyesi", "Vardiya_Tipi", "Vardiya_Rotasyon_Sikligi"]
cat_features = [c for c in categorical_cols if c in df_processed.columns]
df_encoded = pd.get_dummies(df_processed, columns=cat_features, drop_first=True)

X = df_encoded.drop(columns=["Istifa_Etti_Mi"])
y = df_encoded["Istifa_Etti_Mi"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))

constraints_dict = {
    "Maas_TL": -1, "Maas_Endeksi": -1, "Mesafe_KM": 1, "Aylik_Mesai_Saat": 1,
    "Yas": -1, "Toplam_Tecrube_Yil": -1, "Sirketteki_Yil": -1,
    "Ayni_Unvanda_Yil": 1, "Son_Terfi_Gecen_Yil": 1, "Mevcut_Yonetici_Yil": 1,
    "Son_Zamdan_Beri_Ay": 1, "Karier_Tikanma_Orani": 1,
    "Is_Yuku_Orani": 1, "Is_Kazasi_Gecmisi": 1
}
monotone_constraints = tuple(constraints_dict.get(col, 0) for col in X.columns)

model = XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85, gamma=1.0, min_child_weight=2,
    scale_pos_weight=ratio, monotone_constraints=monotone_constraints,
    random_state=42
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]

def print_metrics(threshold, label):
    preds = (probs >= threshold).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_test, preds).ravel()
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    print(f"\n--- 📈 {label} (Eşik: {threshold:.3f}) ---")
    print(f"Confusion Matrix: TN: {TN} | FP: {FP} | FN: {FN} | TP: {TP}")
    print(f"Accuracy: %{acc*100:.2f} | Precision: %{prec*100:.2f} | Recall: %{rec*100:.2f} | F1: {f1:.4f}")

# 1. Default Threshold (0.50)
print_metrics(0.50, "Varsayılan Eşik")

# 2. Optimal F1 Threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.nanargmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print_metrics(optimal_threshold, "Optimum F1 Eşiği")

# 3. Fixed Low Threshold (e.g. 0.15 - İK Erken Uyarı Standardı)
print_metrics(0.15, "İK Erken Uyarı Eşiği")
