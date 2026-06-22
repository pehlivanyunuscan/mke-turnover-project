# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import MKEDataPipeline

app = FastAPI(
    title="MKE Personel Turnover Prediction API", 
    description="MKE Personeli İşten Ayrılma Riski Tahmin ve Karar Destek API'si (XGBoost & SHAP Entegre)",
    version="1.4.0"
)

# CORS Middleware Entegrasyonu (Farklı port ve sunuculardan erişim için production standardı)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Canlı yayında buraya sadece izin verilen Streamlit/Uygulama IP'leri girilmelidir.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelleri ve Ayarları yüklüyoruz
medyan_maaslar = joblib.load("models/medyan_maas_sozlugu.pkl")
pipeline_islemci = MKEDataPipeline(medyan_maas_sozlugu=medyan_maaslar)

# Beyaz Yaka Modeli ve Explainer yükleme
model_beyaz = joblib.load("models/beyaz_yaka_xgb.pkl")
expected_cols_beyaz = joblib.load("models/beyaz_yaka_columns.pkl")
explainer_beyaz = shap.TreeExplainer(model_beyaz)
thresholds_beyaz = joblib.load("models/beyaz_yaka_thresholds.pkl")

# Mavi Yaka Modeli ve Explainer yükleme
model_mavi = joblib.load("models/mavi_yaka_xgb.pkl")
expected_cols_mavi = joblib.load("models/mavi_yaka_columns.pkl")
explainer_mavi = shap.TreeExplainer(model_mavi)
thresholds_mavi = joblib.load("models/mavi_yaka_thresholds.pkl")

print("🟢 TÜM MODELLER, PIPELINE, CORS, SHAP VE EŞİK DEĞERLERİ HAFIZAYA ALINDI!")

class CalisanVerisiInput(BaseModel):
    Cinsiyet: str
    Yas: int
    Medeni_Hal: str
    Cocuk_Sayisi: int
    Ilce: str
    Mesafe_KM: int
    Lojman_Kullanimi: int
    Sertifika_Sayisi: int
    Egitim_Saati_Yil: int
    Seyahat_Sikligi: str
    Toplam_Tecrube_Yil: int
    Sirketteki_Yil: int
    MKE_Oncesi_Tecrube: int
    Ayni_Unvanda_Yil: int
    Son_Terfi_Gecen_Yil: int
    Mevcut_Yonetici_Yil: int
    Birim_Degisim_Sayisi: int
    Departman: str
    Unvan: str
    Egitim_Seviyesi: str
    Maas_TL: float
    Aylik_Mesai_Saat: float
    Vardiya_Tipi: str
    Vardiya_Rotasyon_Sikligi: str
    Is_Kazasi_Gecmisi: int
    Sendika_Sozlesme_Kalan_Ay: int
    Kullanilan_Izin_Gun: int
    Son_Zam_Orani: float
    Son_Zamdan_Beri_Ay: int
    Performans_Puani: int

@app.get("/")
def ana_sayfa():
    return {
        "servis": "MKE Personel Analitiği Tahmin Servisi",
        "durum": "Aktif",
        "modeller": ["beyaz-yaka", "mavi-yaka"],
        "teknolojiler": ["FastAPI", "XGBoost", "SHAP"]
    }

# Sabit Karar Eşikleri (DEĞİŞİKLİK 2)
OPTIMAL_THRESHOLD_BEYAZ = 0.20
OPTIMAL_THRESHOLD_MAVI = 0.55

def predict_risk(df_single, model, expected_columns, explainer, thresholds, optimal_threshold_override):
    df_processed = pipeline_islemci.transform(df_single)
    
    # Kategorik değişkenler listesi (train.py ile birebir eşlenmesi için - Cinsiyet çıkarıldı)
    categorical_cols = [
        "Medeni_Hal", "Ilce", "Seyahat_Sikligi", 
        "Departman", "Unvan", "Egitim_Seviyesi", 
        "Vardiya_Tipi", "Vardiya_Rotasyon_Sikligi"
    ]
    cat_features = [c for c in categorical_cols if c in df_processed.columns]
    
    # drop_first=True ile train.py'daki kolon yapısını tam olarak simüle ediyoruz
    df_encoded = pd.get_dummies(df_processed, columns=cat_features, drop_first=True)
    
    # Eksik one-hot encoding kolonlarını 0 ile doldurma
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_encoded = df_encoded[expected_columns]
    
    # Olasılık Hesabı
    risk_olasiligi = model.predict_proba(df_encoded)[0][1]
    
    # SHAP Değerleri Hesabı (TreeExplainer)
    shap_explanation = explainer(df_encoded)
    shap_vals = shap_explanation.values[0]
    
    if len(shap_explanation.shape) == 3:
        shap_vals = shap_vals[:, 1]
    
    # Değişken katkılarını derleme
    contributions = []
    for col, val in zip(expected_columns, shap_vals):
        if abs(val) > 1e-4:
            contributions.append({
                "feature": col,
                "value": round(float(val), 4)
            })
            
    # Katkı derecesine göre sıralama
    contributions = sorted(contributions, key=lambda x: abs(x["value"]), reverse=True)
    
    # Risk Kategorisi Belirleme (Kalibre Edilmiş Eşiklere Göre)
    opt_t = optimal_threshold_override
    ew_t = thresholds["early_warning_threshold"]
    
    if risk_olasiligi >= opt_t:
        kategori = "🚨 ACİL MÜDAHALE (Yüksek Risk)"
    elif risk_olasiligi >= ew_t:
        kategori = "🟡 ERKEN UYARI (Orta Risk)"
    else:
        kategori = "🟢 GÜVENLİ (Düşük Risk)"
        
    return {
        "Istifa_Riski_Yuzde": round(float(risk_olasiligi) * 100, 2),
        "Risk_Kategorisi": kategori,
        "SHAP_Contributions": contributions,
        "Calibrated_Thresholds": {
            "optimal_threshold": round(opt_t, 4),
            "early_warning_threshold": round(ew_t, 4)
        },
        "Yasal_Uyari": "Bu bir yapay zeka önerisidir."
    }

@app.post("/predict/beyaz-yaka")
def predict_beyaz_yaka(input_data: CalisanVerisiInput):
    try:
        input_dict = input_data.dict()
        df_single = pd.DataFrame([input_dict])
        return predict_risk(df_single, model_beyaz, expected_cols_beyaz, explainer_beyaz, thresholds_beyaz, OPTIMAL_THRESHOLD_BEYAZ)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/mavi-yaka")
def predict_mavi_yaka(input_data: CalisanVerisiInput):
    try:
        input_dict = input_data.dict()
        df_single = pd.DataFrame([input_dict])
        return predict_risk(df_single, model_mavi, expected_cols_mavi, explainer_mavi, thresholds_mavi, OPTIMAL_THRESHOLD_MAVI)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))