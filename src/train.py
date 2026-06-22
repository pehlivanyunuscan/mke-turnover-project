# src/train.py
"""
MKE Personel Turnover Projesi - Model Eğitim, Kalibrasyon ve Kaydetme Pipeline'ı
Bu script, Beyaz ve Mavi Yaka modellerini eğitir, karar eşiklerini (threshold) optimize eder,
monotonluk kısıtlamalarını uygular ve çıktıları FastAPI/Streamlit kullanımı için kaydeder.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Modüllerin import edilebilmesi için yol tanımı
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    from pipeline import MKEDataPipeline
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
    from pipeline import MKEDataPipeline

# Klasörleri oluştur
os.makedirs("models", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. MONOTONLUK KISITLAMALARI SÖZLÜĞÜ (MONOTONE CONSTRAINTS)
# -----------------------------------------------------------------------------
# 1: Pozitif ilişki (Değişken arttıkça istifa riski artar)
# -1: Negatif ilişki (Değişken arttıkça istifa riski azalır)
# 0: Herhangi bir yönlü kısıt yoktur
CONSTRAINTS_DICT = {
    "Maas_TL": -1,
    "Maas_Endeksi": -1,
    "Mesafe_KM": 1,
    "Aylik_Mesai_Saat": 1,
    "Yas": -1,
    "Toplam_Tecrube_Yil": -1,
    "Sirketteki_Yil": -1,
    "Ayni_Unvanda_Yil": 1,        # Unvanda çakılı kalma riski artırır
    "Son_Terfi_Gecen_Yil": 1,     # Terfi alamama süresi riski artırır
    "Mevcut_Yonetici_Yil": 1,     # Yöneticiyle çalışma süresi riski artırır
    "Son_Zamdan_Beri_Ay": 1,      # Zamsız geçen ay sayısı riski artırır
    "Karier_Tikanma_Orani": 1,    # Kariyer tıkanma oranı arttıkça risk artar
    "Is_Yuku_Orani": 1,           # Yeni sorumluluk çarpımı formülü
    "Is_Kazasi_Gecmisi": 1        # İş kazası geçmişi riski artırır
}

CATEGORICAL_COLS = [
    "Medeni_Hal", "Ilce", "Seyahat_Sikligi", 
    "Departman", "Unvan", "Egitim_Seviyesi", 
    "Vardiya_Tipi", "Vardiya_Rotasyon_Sikligi"
]

def optimize_threshold(y_true, y_probs):
    """
    0.25'ten 0.55'e 0.05 adımlarla F1 skorunu maksimize eden optimal karar eşiğini arar.
    Ayrıca istifa edenlerin en az %80'ini yakalayacak Erken Uyarı karar eşiğini hesaplar.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    
    # 0.25'ten 0.55'e 0.05 adımlarla
    threshold_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    for t in threshold_grid:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = t
            
    # Erken uyarı eşiği hesabı (Recall >= %80 hedefi)
    early_warning_threshold = float(best_threshold * 0.5) # Fallback varsayılan
    for t in reversed(np.linspace(0.01, 0.99, 100)):
        preds = (y_probs >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        if rec >= 0.80:
            early_warning_threshold = t
            break
            
    # Erken uyarı eşiğinin optimal eşikten kesin olarak küçük olduğunu garanti et
    if early_warning_threshold >= best_threshold:
        early_warning_threshold = float(best_threshold * 0.7)
        
    return float(best_threshold), float(early_warning_threshold)

def train_and_evaluate_yaka(yaka_adi, df_raw, pipeline):
    """
    İlgili yaka sınıfı için veri ön işlemeyi gerçekleştirir, modeli eğitir,
    eşik değerlerini optimize eder, performans raporunu yazar ve modelleri kaydeder.
    """
    print(f"\n==================== 🤖 {yaka_adi.upper()} YAKA EĞİTİM VE KALİBRASYON HATI ====================")
    
    # 1. Yaka Filtreleme
    df_yaka = df_raw[df_raw["Yaka_Tipi"] == yaka_adi].copy().drop(columns=["Yaka_Tipi"])
    
    # 2. Pipeline Dönüşümü (Feature Engineering adımları)
    df_processed = pipeline.transform(df_yaka)
    
    # Kategorik kolonları filtreleme ve Dummy encoding uygulama
    cat_features = [c for c in CATEGORICAL_COLS if c in df_processed.columns]
    df_encoded = pd.get_dummies(df_processed, columns=cat_features, drop_first=True)
    
    # Hedef ve Girdi Değişkenleri Ayrımı
    X = df_encoded.drop(columns=["Istifa_Etti_Mi"])
    y = df_encoded["Istifa_Etti_Mi"]
    
    # 3. Train-Test Split (80/20 Oranında Stratified Bölme)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Sınıf Dengesizliği Düzeltme (scale_pos_weight hesaplama)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    imbalance_ratio = float(neg_count / pos_count) if pos_count > 0 else 1.0
    print(f"📊 Sınıf Dağılımı: Negatif={neg_count} | Pozitif={pos_count} ( scale_pos_weight = {imbalance_ratio:.3f} )")
    
    # 5. Monotonluk Kısıtlamalarını Hizalama
    monotone_constraints = tuple(CONSTRAINTS_DICT.get(col, 0) for col in X.columns)
    
    # 6. Model Kurulumu ve Eğitimi
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=1.0,
        min_child_weight=2,
        scale_pos_weight=imbalance_ratio,
        monotone_constraints=monotone_constraints,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print(f"🎯 Model eğitim süreci tamamlandı.")
    
    # 7. Karar Sınırı Arama ve Optimizasyonu
    y_probs = model.predict_proba(X_val)[:, 1]
    opt_t, ew_t = optimize_threshold(y_val, y_probs)
    print(f"⚖️ Kalibre edilen karar sınırları: Acil Müdahale Eşiği={opt_t:.4f} | Erken Uyarı Eşiği={ew_t:.4f}")
    
    # 8. Performans Raporlama (Seçilen Eşiklere Göre)
    def print_metrics(threshold, label):
        preds = (y_probs >= threshold).astype(int)
        TN, FP, FN, TP = confusion_matrix(y_val, preds).ravel()
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        
        print(f"\n   📈 [{label} Sınırı - Eşik: {threshold:.3f}] Performansı:")
        print(f"   --------------------------------------------------------")
        print(f"   Karmaşıklık Matrisi (CM)  : TN={TN} | FP={FP} (Yanlış Alarm) | FN={FN} (Kaçırılan) | TP={TP}")
        print(f"   Doğruluk (Accuracy)       : % {acc*100:.2f}")
        print(f"   Kesinlik (Precision)      : % {prec*100:.2f}")
        print(f"   Duyarlılık (Recall)       : % {rec*100:.2f}")
        print(f"   F1-Score                  : {f1:.4f}")
        
    print_metrics(0.50, "Varsayılan (0.50)")
    print_metrics(opt_t, "Optimal F1 (Acil Müdahale)")
    print_metrics(ew_t, "Erken Uyarı (Takip)")
    
    # 9. Model ve Ayarların Kaydedilmesi
    yaka_slug = "beyaz_yaka" if yaka_adi == "Beyaz" else "mavi_yaka"
    joblib.dump(model, f"models/{yaka_slug}_xgb.pkl")
    joblib.dump(X.columns.tolist(), f"models/{yaka_slug}_columns.pkl")
    joblib.dump({"optimal_threshold": opt_t, "early_warning_threshold": ew_t}, f"models/{yaka_slug}_thresholds.pkl")
    print(f"💾 {yaka_adi} Yaka modeli, kolonları ve eşik değerleri models/ klasörüne kaydedildi.")

def run_pipeline():
    # Ham veriyi okuma
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dataset_path = os.path.join(project_root_dir, "data/dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Hata: '{dataset_path}' veri seti bulunamadı. Lütfen önce veri setini üretin.")
        sys.exit(1)
        
    df = pd.read_csv(dataset_path)
    
    # Pipeline eğitimi
    pipeline = MKEDataPipeline()
    pipeline.fit(df)
    joblib.dump(pipeline.medyan_maas_sozlugu, "models/medyan_maas_sozlugu.pkl")
    print("💾 Medyan maaş sözlüğü models/ klasörüne kaydedildi.")
    
    # Beyaz Yaka Eğitimi
    train_and_evaluate_yaka("Beyaz", df, pipeline)
    
    # Mavi Yaka Eğitimi
    train_and_evaluate_yaka("Mavi", df, pipeline)
    
    print("\n🎉 Tüm eğitim, kalibrasyon ve kaydetme adımları başarıyla tamamlandı!")

if __name__ == "__main__":
    run_pipeline()