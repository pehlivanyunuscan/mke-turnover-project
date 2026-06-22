# scratch/test_calibration.py
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

workspace_src = "/Users/yunuscanpehlivan/Desktop/mke turnover/src"
sys.path.append(workspace_src)
from pipeline import MKEDataPipeline

df = pd.read_csv("data/dataset.csv")

def calibrate_yaka(yaka_adi):
    print(f"\n==================== 📊 {yaka_adi.upper()} YAKA EŞİK KALİBRASYONU ====================")
    df_yaka = df[df["Yaka_Tipi"] == yaka_adi].copy().drop(columns=["Yaka_Tipi"])
    
    pipeline = MKEDataPipeline()
    pipeline.fit(df_yaka)
    df_processed = pipeline.transform(df_yaka)
    
    categorical_cols = ["Cinsiyet", "Medeni_Hal", "Ilce", "Seyahat_Sikligi", "Departman", "Unvan", "Egitim_Seviyesi", "Vardiya_Tipi", "Vardiya_Rotasyon_Sikligi"]
    cat_features = [c for c in categorical_cols if c in df_processed.columns]
    df_encoded = pd.get_dummies(df_processed, columns=cat_features, drop_first=True)
    
    X = df_encoded.drop(columns=["Istifa_Etti_Mi"])
    y = df_encoded["Istifa_Etti_Mi"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    
    constraints_dict = {
        "Maas_TL": -1, "Maas_Endeksi": -1, "Mesafe_KM": 1, "Aylik_Mesai_Saat": 1,
        "Is_Memnuniyeti": -1, "Ortam_Memnuniyeti": -1, "Is_Ozel_Hayat_Dengesi": -1,
        "Ekip_Uyum_Puani": -1, "Yas": -1, "Toplam_Tecrube_Yil": -1, "Sirketteki_Yil": -1,
        "Ayni_Unvanda_Yil": 1, "Son_Terfi_Gecen_Yil": 1, "Mevcut_Yonetici_Yil": 1,
        "Son_Zamdan_Beri_Ay": 1, "Celiski_Skoru": 1, "Karier_Tikanma_Orani": 1,
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
    
    # Predict probabilities on test
    probs = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold to maximize F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Find early warning threshold (where recall is closest to 0.80)
    # Filter thresholds where recall >= 0.80, and choose the highest threshold (which has best precision)
    eligible_indices = np.where(recalls >= 0.80)[0]
    if len(eligible_indices) > 0:
        # recalls index matches thresholds index except thresholds has length len(recalls)-1
        # so we find index in thresholds
        early_warning_idx = eligible_indices[-1] # the last index since recalls is sorted in descending order
        # Make sure index is within bounds of thresholds
        early_warning_idx = min(early_warning_idx, len(thresholds)-1)
        early_warning_threshold = thresholds[early_warning_idx]
    else:
        # Fallback to a default low value
        early_warning_threshold = 0.15
        
    print(f"Kalibre Edilen Eşikler:")
    print(f"  - Optimum F1 Eşiği (🚨 Acil Müdahale) : {optimal_threshold:.4f}")
    print(f"  - Erken Uyarı Eşiği (🟡 İzleme)       : {early_warning_threshold:.4f}")
    
    # Predict with both
    preds_opt = (probs >= optimal_threshold).astype(int)
    preds_ew = (probs >= early_warning_threshold).astype(int)
    
    print("\n[🚨 Acil Müdahale Seviyesi Metrikleri]")
    print(f"  Accuracy: %{accuracy_score(y_test, preds_opt)*100:.2f} | Precision: %{precision_score(y_test, preds_opt)*100:.2f} | Recall: %{recall_score(y_test, preds_opt)*100:.2f} | F1: {f1_score(y_test, preds_opt):.4f}")
    
    print("\n[🟡 Erken Uyarı Seviyesi Metrikleri]")
    print(f"  Accuracy: %{accuracy_score(y_test, preds_ew)*100:.2f} | Precision: %{precision_score(y_test, preds_ew, zero_division=0)*100:.2f} | Recall: %{recall_score(y_test, preds_ew)*100:.2f} | F1: {f1_score(y_test, preds_ew):.4f}")

calibrate_yaka("Beyaz")
calibrate_yaka("Mavi")
