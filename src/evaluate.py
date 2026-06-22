# src/evaluate.py
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# Append the src path in case it is run from subfolders
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import MKEDataPipeline

print("🔍 MKE Turnover Model Değerlendirme Raporu (Monotonluk Kısıtlamalı & Düzenlileştirilmiş) Hazırlanıyor...")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
dataset_path = os.path.join(project_root, "data/dataset.csv")

if not os.path.exists(dataset_path):
    print(f"❌ Hata: '{dataset_path}' dosyası bulunamadı! Lütfen önce veriyi oluşturun.")
    sys.exit(1)

df = pd.read_csv(dataset_path)

categorical_cols = [
    "Medeni_Hal", "Ilce", "Seyahat_Sikligi", 
    "Departman", "Unvan", "Egitim_Seviyesi", 
    "Vardiya_Tipi", "Vardiya_Rotasyon_Sikligi"
]

constraints_dict = {
    "Maas_TL": -1,
    "Maas_Endeksi": -1,
    "Mesafe_KM": 1,
    "Aylik_Mesai_Saat": 1,
    "Yas": -1,
    "Toplam_Tecrube_Yil": -1,
    "Sirketteki_Yil": -1,
    "Ayni_Unvanda_Yil": 1,
    "Son_Terfi_Gecen_Yil": 1,
    "Mevcut_Yonetici_Yil": 1,
    "Son_Zamdan_Beri_Ay": 1,
    "Karier_Tikanma_Orani": 1,
    "Is_Yuku_Orani": 1,
    "Is_Kazasi_Gecmisi": 1
}

def evaluate_yaka(yaka_adi):
    print(f"\n==================== 📊 {yaka_adi.upper()} YAKA MODELİ PERFORMANSI ====================")
    df_yaka = df[df["Yaka_Tipi"] == yaka_adi].copy()
    df_yaka = df_yaka.drop(columns=["Yaka_Tipi"])
    
    # Fit pipeline
    pipeline = MKEDataPipeline()
    pipeline.fit(df_yaka)
    df_processed = pipeline.transform(df_yaka)
    
    cat_features = [c for c in categorical_cols if c in df_processed.columns]
    df_encoded = pd.get_dummies(df_processed, columns=cat_features, drop_first=True)
    
    X = df_encoded.drop(columns=["Istifa_Etti_Mi"])
    y = df_encoded["Istifa_Etti_Mi"]
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Imbalance ratio
    ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    
    monotone_constraints = tuple(constraints_dict.get(col, 0) for col in X.columns)
    
    # Model (Monotonluk ve Düzenlileştirme parametreleri eklendi)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=1.0,
        min_child_weight=2,
        scale_pos_weight=ratio,
        monotone_constraints=monotone_constraints,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    print(f"✅ Doğruluk Oranı (Accuracy) : %{acc*100:.2f}")
    print(f"🎯 Kesinlik (Precision)      : %{prec*100:.2f} (Model 'İstifa Edecek' dediğinde doğruluk oranı)")
    print(f"⚡ Duyarlılık (Recall)       : %{rec*100:.2f} (Tüm istifaların model tarafından yakalanma oranı)")
    print(f"⭐ F1-Score                  : {f1:.4f}")
    print(f"🌀 ROC-AUC (Ayırt Etme Gücü)  : {auc:.4f} (1.0 mükemmel, 0.5 şans eseri)")
    print("\n📋 Detaylı Sınıflandırma Matrisi:")
    print(classification_report(y_test, preds))

evaluate_yaka("Beyaz")
evaluate_yaka("Mavi")
