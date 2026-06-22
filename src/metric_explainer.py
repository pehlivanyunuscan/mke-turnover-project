# src/metric_explainer.py
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# Workspace path injection
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import MKEDataPipeline

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_explanation():
    clear_console()
    print("===============================================================================")
    print("🎯 MKE TURNOVER - YAPAY ZEKA METRİKLERİ VE KARAR GEREKÇELERİ KILAVUZU 🎯")
    print("===============================================================================")
    print("Bu script, eğitilen Beyaz Yaka modelinin test verileri üzerindeki tahmin başarısını")
    print("formülleriyle hesaplar ve İK diline çevirerek açıklar.\n")

    # 1. Veri Okuma
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    dataset_path = os.path.join(project_root, "data/dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Hata: '{dataset_path}' bulunamadı.")
        return
    
    df = pd.read_csv(dataset_path)
    df_yaka = df[df["Yaka_Tipi"] == "Beyaz"].copy().drop(columns=["Yaka_Tipi"])
    
    # 2. Pipeline ve Veri Dönüşümü
    pipeline = MKEDataPipeline()
    pipeline.fit(df_yaka)
    df_processed = pipeline.transform(df_yaka)
    
    categorical_cols = [
        "Medeni_Hal", "Ilce", "Seyahat_Sikligi", 
        "Departman", "Unvan", "Egitim_Seviyesi", 
        "Vardiya_Tipi", "Vardiya_Rotasyon_Sikligi"
    ]
    cat_features = [c for c in categorical_cols if c in df_processed.columns]
    df_encoded = pd.get_dummies(df_processed, columns=cat_features, drop_first=True)
    
    X = df_encoded.drop(columns=["Istifa_Etti_Mi"])
    y = df_encoded["Istifa_Etti_Mi"]
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Monotonluk kısıtlamaları tanımı (değerlendirme tutarlılığı için)
    constraints_dict = {
        "Maas_TL": -1, "Maas_Endeksi": -1, "Mesafe_KM": 1, "Aylik_Mesai_Saat": 1,
        "Yas": -1, "Toplam_Tecrube_Yil": -1, "Sirketteki_Yil": -1,
        "Ayni_Unvanda_Yil": 1, "Son_Terfi_Gecen_Yil": 1, "Mevcut_Yonetici_Yil": 1,
        "Son_Zamdan_Beri_Ay": 1, "Karier_Tikanma_Orani": 1,
        "Is_Yuku_Orani": 1, "Is_Kazasi_Gecmisi": 1
    }
    monotone_constraints = tuple(constraints_dict.get(col, 0) for col in X.columns)
    
    # Model Eğitimi
    ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, gamma=1.0, min_child_weight=2,
        scale_pos_weight=ratio, monotone_constraints=monotone_constraints,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Tahminler
    preds = model.predict(X_test)
    
    # Karmaşıklık Matrisi (Confusion Matrix) Çıkarımı
    # TN: True Negative, FP: False Positive, FN: False Negative, TP: True Positive
    TN, FP, FN, TP = confusion_matrix(y_test, preds).ravel()
    
    total = TN + FP + FN + TP
    
    print("-------------------------------------------------------------------------------")
    # 3. Temel Dağılım Tablosu (Matris)
    print("📊 TAHMİN MATRİSİ SONUÇLARI (Test Seti: {} Çalışan)".format(total))
    print("-------------------------------------------------------------------------------")
    print(f"|  Gerçek Durum \\ Tahmin |  KALACAK (Tahmin: 0)  |  İSTİFA EDECEK (Tahmin: 1) |")
    print(f"|-----------------------|-----------------------|----------------------------|")
    print(f"|  ŞİRKETTE KALDI (0)   |  TN: {TN} Kişi (Doğru)   |  FP: {FP} Kişi (Hatalı)      |")
    print(f"|  İSTİFA EDİP GİTTİ (1)|  FN: {FN} Kişi (Hatalı)   |  TP: {TP} Kişi (Doğru)       |")
    print("-------------------------------------------------------------------------------\n")
    
    # Metrik Hesaplamaları
    acc = (TP + TN) / total
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    # 4. Metrik Metrik Açıklamalar
    print("===============================================================================")
    print("🧠 METRİKLERİN DETAYLI ADIM ADIM AÇIKLAMASI")
    print("===============================================================================\n")
    
    # ACCURACY
    print("1️⃣ DOĞRULUK ORANI (ACCURACY)")
    print(f"   👉 Formül: (TP + TN) / Toplam = ({TP} + {TN}) / {total}")
    print(f"   📈 Sonuç : % {acc * 100:.2f}")
    print("   ❓ Ne İşe Yarar?: Modelin yaptığı tüm tahminlerin (hem kalanların hem gidenlerin) genel başarı oranıdır.")
    print("   ⚠️ İK Uyarısı  : İK veri setleri dengesizdir (istifa oranı düşüktür). Herkese 'kalacak' diyen işe yaramaz bir model")
    print("                    bile bu veri setinde %89 doğruluk oranı verir. Bu yüzden jürinin gözünde tek başına değersizdir!\n")
    
    # PRECISION
    print("2️⃣ KESİNLİK / DOĞRULUK PAYI (PRECISION)")
    print(f"   👉 Formül: TP / (TP + FP) = {TP} / ({TP} + {FP})")
    print(f"   📈 Sonuç : % {prec * 100:.2f}")
    print("   ❓ Ne İşe Yarar?: Modelimizin 'İSTİFA EDECEK' diye işaretlediği çalışanlardan yüzde kaçı gerçekten gitti?")
    print("   💼 İK Yorumu   : Yalancı alarm denetimidir. Bu oranın yüksek olması, İK'nın yanlış alarmlar yüzünden")
    print("                    boş yere vakit ve kaynak harcamasını (gereksiz ikna görüşmeleri vb.) engeller.\n")
    
    # RECALL
    print("3️⃣ DUYARLILIK / YAKALAMA ORANI (RECALL)")
    print(f"   👉 Formül: TP / (TP + FN) = {TP} / ({TP} + {FN})")
    print(f"   📈 Sonuç : % {rec * 100:.2f}")
    print("   ❓ Ne İşe Yarar?: Şirketten gerçekten istifa edip giden toplam kişilerin yüzde kaçını gitmeden önce yakalayabildik?")
    print("   💼 İK Yorumu   : Modelin yakalama gücüdür. İK için en hayati metriktir. Kaçan istifa sayısını (FN) minimize")
    print("                    ederek kritik pozisyondaki personellerin kaybını engellememizi sağlar.\n")
    
    # F1 SCORE
    print("4️⃣ F1-SKORU (F1-SCORE)")
    print(f"   👉 Formül: 2 * (Precision * Recall) / (Precision + Recall)")
    print(f"   📈 Sonuç : {f1:.4f}")
    print("   ❓ Ne İşe Yarar?: Kesinlik (Precision) ve Duyarlılık (Recall) arasındaki dengedir. İkisi de yüksek olmalıdır.")
    print("   💼 İK Yorumu   : Modelin genel kalitesini gösteren tekil göstergedir. Yalancı alarm ile kaçan istifa terazisinin")
    print("                    tam ortasıdır. Jüri üyeleri İK projelerinde ilk olarak bu F1 skoruna bakarlar.\n")
    
    print("===============================================================================")
    print("🎯 JÜRİYE SUNARKEN KULLANILACAK KATİL ARGÜMAN:")
    print("-------------------------------------------------------------------------------")
    print(f" 'MKE genelinde istifa oranı %11 civarındadır. Rastgele tahmin yapan bir model yerine")
    print(f" geliştirdiğimiz XGBoost modeli sayesinde, gitmekte olan personelleri %{rec * 100:.1f} başarıyla")
    print(f" önceden saptayabiliyoruz. Bu da kurumsal tecrübe kaybını doğrudan engellememizi sağlıyor.'")
    print("===============================================================================")

if __name__ == "__main__":
    run_explanation()
