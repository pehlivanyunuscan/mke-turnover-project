# app/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import shap
import matplotlib.pyplot as plt

OPTIMAL_THRESHOLD_BEYAZ = 0.20
OPTIMAL_THRESHOLD_MAVI = 0.55

# Page configuration - McKinsey/Bain style presentation mode
st.set_page_config(
    page_title="MKE İK Yönetim Kurulu Paneli", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Workspace path injection for pipeline loading
workspace_src = "/Users/yunuscanpehlivan/Desktop/mke turnover/src"
if workspace_src not in sys.path:
    sys.path.append(workspace_src)
from pipeline import MKEDataPipeline

# CSS for corporate Bain/McKinsey light-theme dashboard aesthetic
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Reset Streamlit defaults for custom dashboard look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #F4F6F9 !important;
        color: #1E293B !important;
    }
    
    /* Force readable text colors for Streamlit widgets */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stMarkdownContainer"] p,
    .stRadio label,
    .stSlider label,
    .stSelectbox label,
    .stNumberInput label,
    .stCheckbox label,
    .stRadio div[role="radiogroup"] p,
    div[data-testid="stRadio"] label p,
    .stWidgetForm label,
    .stApp label {
        color: #1E293B !important;
        font-weight: 500 !important;
    }
    
    /* Header Card styling */
    .dashboard-header {
        background: #1B2A4A;
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(27, 42, 74, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* KPI Cards styling */
    .kpi-container {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: left;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        border-bottom: 4px solid #1B2A4A;
    }
    
    .kpi-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1B2A4A;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    
    .kpi-subtext {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 500;
    }
    
    /* Layout sections */
    .content-card {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #1B2A4A;
        margin-bottom: 15px;
        border-left: 4px solid #F39C12;
        padding-left: 10px;
    }
    
    /* Action Table Style */
    .action-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
    }
    .action-table th {
        background-color: #1B2A4A;
        color: white;
        font-weight: 700;
        padding: 12px;
        text-align: left;
        font-size: 0.85rem;
    }
    .action-table td {
        padding: 12px;
        border-bottom: 1px solid #E2E8F0;
        font-size: 0.85rem;
    }
    .action-table tr:hover {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. VERİ YÜKLEME VE TOPLU TAHMİN (INITIALIZATION)
# -----------------------------------------------------------------------------

@st.cache_resource
def load_models_and_predict_all_v3():
    # Model ve pipeline bileşenlerini yükleme
    medyan_maaslar = joblib.load("models/medyan_maas_sozlugu.pkl")
    pipeline_islemci = MKEDataPipeline(medyan_maas_sozlugu=medyan_maaslar)
    
    model_beyaz = joblib.load("models/beyaz_yaka_xgb.pkl")
    cols_beyaz = joblib.load("models/beyaz_yaka_columns.pkl")
    
    model_mavi = joblib.load("models/mavi_yaka_xgb.pkl")
    cols_mavi = joblib.load("models/mavi_yaka_columns.pkl")
    
    # Eşik değerlerini yükleme
    thresholds_beyaz = joblib.load("models/beyaz_yaka_thresholds.pkl")
    thresholds_mavi = joblib.load("models/mavi_yaka_thresholds.pkl")
    
    # Ham veriyi okuma
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    dataset_path = os.path.join(project_root, "data/dataset.csv")
    df = pd.read_csv(dataset_path)
    
    # Beyaz Yaka Tahminleri
    df_beyaz = df[df["Yaka_Tipi"] == "Beyaz"].copy()
    df_beyaz_no_yaka = df_beyaz.drop(columns=["Yaka_Tipi"])
    df_beyaz_proc = pipeline_islemci.transform(df_beyaz_no_yaka)
    
    # Kategorik dummy encoding
    categorical_cols = ["Medeni_Hal", "Ilce", "Seyahat_Sikligi", "Departman", "Unvan", "Egitim_Seviyesi", "Vardiya_Tipi", "Vardiya_Rotasyon_Sikligi"]
    cat_beyaz = [c for c in categorical_cols if c in df_beyaz_proc.columns]
    df_beyaz_enc = pd.get_dummies(df_beyaz_proc, columns=cat_beyaz, drop_first=True)
    for col in cols_beyaz:
        if col not in df_beyaz_enc.columns:
            df_beyaz_enc[col] = 0
    df_beyaz_enc = df_beyaz_enc[cols_beyaz].reset_index(drop=True)
    
    probs_beyaz = model_beyaz.predict_proba(df_beyaz_enc)[:, 1]
    df_beyaz["Risk_Skoru"] = probs_beyaz
    df_beyaz["original_yaka_index"] = range(len(df_beyaz))
    
    # Mavi Yaka Tahminleri
    df_mavi = df[df["Yaka_Tipi"] == "Mavi"].copy()
    df_mavi_no_yaka = df_mavi.drop(columns=["Yaka_Tipi"])
    df_mavi_proc = pipeline_islemci.transform(df_mavi_no_yaka)
    
    cat_mavi = [c for c in categorical_cols if c in df_mavi_proc.columns]
    df_mavi_enc = pd.get_dummies(df_mavi_proc, columns=cat_mavi, drop_first=True)
    for col in cols_mavi:
        if col not in df_mavi_enc.columns:
            df_mavi_enc[col] = 0
    df_mavi_enc = df_mavi_enc[cols_mavi].reset_index(drop=True)
    
    probs_mavi = model_mavi.predict_proba(df_mavi_enc)[:, 1]
    df_mavi["Risk_Skoru"] = probs_mavi
    df_mavi["original_yaka_index"] = range(len(df_mavi))
    
    # Birleştirilmiş nihai veri seti
    df_all = pd.concat([df_beyaz, df_mavi]).reset_index(drop=True)
    
    # SHAP Explainer'ları
    explainer_beyaz = shap.TreeExplainer(model_beyaz)
    explainer_mavi = shap.TreeExplainer(model_mavi)
    
    return df_all, df_beyaz_enc, df_mavi_enc, explainer_beyaz, explainer_mavi, cols_beyaz, cols_mavi, thresholds_beyaz, thresholds_mavi, pipeline_islemci, model_beyaz, model_mavi

# Verileri ve tahminleri arka planda yükle
df_all, df_beyaz_enc, df_mavi_enc, explainer_beyaz, explainer_mavi, cols_beyaz, cols_mavi, thresholds_beyaz, thresholds_mavi, pipeline_islemci, model_beyaz, model_mavi = load_models_and_predict_all_v3()

# SHAP Değişken İsimleri Sözlüğü (Teknik ifadeleri iş diline çeviriyoruz)
shap_translation = {
    "Is_Memnuniyeti": "Düşük İş Memnuniyeti",
    "Is_Ozel_Hayat_Dengesi": "İş-Özel Hayat Dengesi Problemleri",
    "Performans_Puani": "Düşük Performans Değerlendirmesi",
    "Maas_Endeksi": "Düşük Maaş Politikası (Rol Ortalaması Altı)",
    "Maas_TL": "Düşük Aylık Maaş",
    "Aylik_Mesai_Saat": "Aşırı Fazla Mesai Saatleri",
    "Yas": "Genç Yaş Mobilizasyonu",
    "Toplam_Tecrube_Yil": "Kıdem ve Tecrübe Eksikliği",
    "Sirketteki_Yil": "Düşük Kurumsal Bağlılık Süresi",
    "Ayni_Unvanda_Yil": "Kariyer İlerlemesinde Tıkanma",
    "Son_Terfi_Gecen_Yil": "Terfi Alamama Süresi",
    "Mevcut_Yonetici_Yil": "Yönetici ile Uyumsuzluk Süresi",
    "Son_Zamdan_Beri_Ay": "Zam Döneminden Beri Geçen Süre",
    "Celiski_Skoru": "Aşırı Mesai & Hayat Dengesi Çelişkisi",
    "Karier_Tikanma_Orani": "Düşük Terfi / Kariyer Tıkanıklığı",
    "Is_Yuku_Orani": "Aşırı Fazla Mesai / İzin Yetersizliği",
    "Is_Kazasi_Gecmisi": "İş Kazası Geçmişi ve Güvensizlik",
    "Mesafe_KM": "Uzak İkametgah ve Ulaşım Zorluğu"
}

def clean_feature_name(name):
    # One-hot encoded ekleri temizle ve sözlükten çevir
    base_name = name
    for cat in ["Departman_", "Unvan_", "Egitim_Seviyesi_", "Medeni_Hal_", "Ilce_", "Vardiya_Tipi_", "Vardiya_Rotasyon_Sikligi_", "Seyahat_Sikligi_"]:
        if name.startswith(cat):
            base_name = name.split(cat)[0]  # Kategori adını koru veya eşle
            return f"Rol/Lokasyon Özelliği ({name.replace('_', ' ')})"
    return shap_translation.get(base_name, base_name.replace("_", " "))

# -----------------------------------------------------------------------------
# MKE DERECE-KADEME MAAŞ VE UNVAN SKALASI (GÖRSEL REFERANS İÇİN)
# -----------------------------------------------------------------------------
mke_skala = {
    "Mavi": {
        "Operator": {"min": 24000, "max": 28000},
        "Teknisyen": {"min": 28000, "max": 35000},
        "Atolye Ustasi": {"min": 35000, "max": 48000},
        "Basusta": {"min": 45000, "max": 60000}
    },
    "Beyaz": {
        "Uzman": {"min": 42000, "max": 55000},
        "Muhendis": {"min": 48000, "max": 60000},
        "Kidemli Muhendis": {"min": 60000, "max": 85000},
        "Takim Lideri": {"min": 85000, "max": 120000}
    }
}

def render_simulator():
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>👤 Çalışan Simülatörü ve Karar Destek Paneli</div>", unsafe_allow_html=True)
    st.write("Bu simülasyon ekranında, mevcut bir personelin İK koşullarını değiştirerek (örneğin maaş artışı veya fazla mesai azaltımı) ayrılma olasılığındaki anlık değişimi görebilirsiniz.")
    
    # Seçenek: Mevcut çalışan veya Yeni Profil
    mod = st.radio("Simülasyon Modu", ["Mevcut Çalışan Üzerinde Simülasyon", "Sıfırdan Yeni Çalışan Profili"], horizontal=True)
    
    if mod == "Mevcut Çalışan Üzerinde Simülasyon":
        # Çalışan listesi oluştur (Personel_ID - Departman - Unvan)
        emp_list = df_all.apply(lambda r: f"{r['Personel_ID']} - {r['Departman']} - {r['Unvan']} (Mevcut Risk: %{r['Risk_Skoru']*100:.1f})", axis=1).tolist()
        selected_emp_str = st.selectbox("Simüle Edilecek Çalışanı Seçin", emp_list)
        selected_id = selected_emp_str.split(" - ")[0]
        
        # Çalışanın mevcut verilerini çek
        emp_data = df_all[df_all["Personel_ID"] == selected_id].iloc[0].to_dict()
    else:
        # Boş bir çalışan verisi oluştur
        emp_data = {
            "Personel_ID": "YENI-01", "Yaka_Tipi": "Beyaz", "Cinsiyet": "Erkek", "Yas": 30,
            "Medeni_Hal": "Bekar", "Cocuk_Sayisi": 0, "Ilce": "Cankaya", "Mesafe_KM": 15,
            "Lojman_Kullanimi": 0, "Sertifika_Sayisi": 1, "Egitim_Saati_Yil": 20, "Seyahat_Sikligi": "Nadiren",
            "Toplam_Tecrube_Yil": 8, "Sirketteki_Yil": 3, "MKE_Oncesi_Tecrube": 5, "Ayni_Unvanda_Yil": 2,
            "Son_Terfi_Gecen_Yil": 2, "Mevcut_Yonetici_Yil": 2, "Birim_Degisim_Sayisi": 0,
            "Departman": "AR-GE Tasarim", "Unvan": "Muhendis", "Egitim_Seviyesi": "Lisans",
            "Maas_TL": 50000.0, "Aylik_Mesai_Saat": 25.0, "Vardiya_Tipi": "Yok", "Vardiya_Rotasyon_Sikligi": "Yok",
            "Is_Kazasi_Gecmisi": 0, "Sendika_Sozlesme_Kalan_Ay": 0, "Kullanilan_Izin_Gun": 18,
            "Son_Zam_Orani": 40.0, "Son_Zamdan_Beri_Ay": 6, "Performans_Puani": 3,
            "Ortam_Memnuniyeti": 3, "Is_Memnuniyeti": 3, "Ekip_Uyum_Puani": 4, "Is_Ozel_Hayat_Dengesi": 3
        }

    # Form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h4 style='color:#1B2A4A; border-bottom:2px solid #E2E8F0; padding-bottom:5px;'>📋 Pozisyon & Demografi</h4>", unsafe_allow_html=True)
        yaka_tipi = st.selectbox("Yaka Sınıfı", ["Beyaz", "Mavi"], index=0 if emp_data["Yaka_Tipi"] == "Beyaz" else 1)
        
        # Departman ve unvan listesini yaka tipine göre dinamik filtrele!
        if yaka_tipi == "Beyaz":
            depts = ["AR-GE Tasarim", "Bilgi Teknolojileri", "IK", "Satinalma", "Kalite Yonetim"]
            titles = ["Uzman", "Muhendis", "Kidemli Muhendis", "Takim Lideri"]
        else:
            depts = ["Muhimmat Fabrikasi", "Silah Fabrikasi", "Gazi Fisek Fabrikasi", "Bakim Onarim"]
            titles = ["Operator", "Teknisyen", "Atolye Ustasi", "Basusta"]
            
        dept_index = depts.index(emp_data["Departman"]) if emp_data["Departman"] in depts else 0
        title_index = titles.index(emp_data["Unvan"]) if emp_data["Unvan"] in titles else 0
        
        departman = st.selectbox("Departman", depts, index=dept_index)
        unvan = st.selectbox("Unvan", titles, index=title_index)
        
        cinsiyet = st.selectbox("Cinsiyet", ["Erkek", "Kadin"], index=0 if emp_data["Cinsiyet"] == "Erkek" else 1)
        medeni_hal = st.selectbox("Medeni Hal", ["Evli", "Bekar"], index=0 if emp_data["Medeni_Hal"] == "Evli" else 1)
        cocuk_sayisi = st.number_input("Çocuk Sayısı", min_value=0, max_value=10, value=int(emp_data["Cocuk_Sayisi"]))
        egitim_seviyesi = st.selectbox("Eğitim Seviyesi", ["Lise", "On Lisans", "Lisans", "Yuksek Lisans", "Doktora"], 
                                       index=["Lise", "On Lisans", "Lisans", "Yuksek Lisans", "Doktora"].index(emp_data["Egitim_Seviyesi"]) if emp_data["Egitim_Seviyesi"] in ["Lise", "On Lisans", "Lisans", "Yuksek Lisans", "Doktora"] else 2)

    with col2:
        st.markdown("<h4 style='color:#1B2A4A; border-bottom:2px solid #E2E8F0; padding-bottom:5px;'>💼 Kıdem & Finansal</h4>", unsafe_allow_html=True)
        yas = st.slider("Yaş", 18, 65, int(emp_data["Yas"]))
        toplam_tecrube = st.slider("Toplam Tecrübe (Yıl)", 0, 45, int(emp_data["Toplam_Tecrube_Yil"]))
        sirketteki_yil = st.slider("Kurum Kıdemi (MKE Yılı)", 0, 45, int(emp_data["Sirketteki_Yil"]))
        ayni_unvanda_yil = st.slider("Aynı Unvanda Geçen Yıl", 0, 30, int(emp_data["Ayni_Unvanda_Yil"]))
        son_terfi_gecen_yil = st.slider("Son Terfiden Beri Geçen Yıl", 0, 20, int(emp_data["Son_Terfi_Gecen_Yil"]))
        mevcut_yonetici_yil = st.slider("Mevcut Yöneticiyle Çalışma Yılı", 0, 20, int(emp_data["Mevcut_Yonetici_Yil"]))
        
        # Get the median salary for the selected role
        medyan_maas_dict = joblib.load("models/medyan_maas_sozlugu.pkl")
        medyan_maas = medyan_maas_dict.get((departman, unvan), 50000.0)
        
        # Calculate initial endeks based on current Maas_TL or default to 1.0
        initial_endeks = float(emp_data.get("Maas_TL", medyan_maas) / medyan_maas)
        
        # Slider for Maaş Endeksi (Compa-Ratio)
        maas_endeksi = st.slider(
            "Maaş Endeksi (Rol Medyan Maaş Çarpanı)", 
            min_value=0.50, 
            max_value=2.00, 
            value=float(np.clip(initial_endeks, 0.50, 2.00)), 
            step=0.05
        )
        maas_tl = float(medyan_maas * maas_endeksi)
        st.info(f"💵 Simüle Edilen Maaş: {int(maas_tl):,} TL (Rol Medyanı: {int(medyan_maas):,} TL)")
        
        son_zam_orani = st.slider("Son Zam Oranı (%)", 0.0, 100.0, float(emp_data["Son_Zam_Orani"]))
        son_zam_ay = st.slider("Son Zamdan Beri Geçen Ay", 0, 24, int(emp_data["Son_Zamdan_Beri_Ay"]))

    with col3:
        st.markdown("<h4 style='color:#1B2A4A; border-bottom:2px solid #E2E8F0; padding-bottom:5px;'>⚡ Çalışma Şartları & Performans</h4>", unsafe_allow_html=True)
        aylik_mesai = st.slider("Aylık Fazla Mesai (Saat)", 0, 100, int(emp_data["Aylik_Mesai_Saat"]))
        kullanilan_izin = st.slider("Kullanılan İzin Günü (Yıllık)", 0, 30, int(emp_data["Kullanilan_Izin_Gun"]))
        mesafe_km = st.slider("Ev-İş Mesafesi (KM)", 1, 100, int(emp_data["Mesafe_KM"]))
        lojman = st.selectbox("Lojman Kullanımı", ["Lojman Yok", "Lojmanda Kalıyor"], index=int(emp_data["Lojman_Kullanimi"]))
        lojman_val = 1 if lojman == "Lojmanda Kalıyor" else 0
        
        # Memnuniyet Skorları (Modelden kaldırılmıştır)
        performans = st.slider("Performans Puanı (1-5)", 1, 5, int(emp_data["Performans_Puani"]))
        ortam_m, is_m, ekip_m, denge_m = 3, 3, 3, 3
        
        # Ekstra Detaylar
        sertifika = st.number_input("Sertifika Sayısı", min_value=0, max_value=20, value=int(emp_data.get("Sertifika_Sayisi", 0)))
        egitim_saat = st.number_input("Yıllık Eğitim Saati", min_value=0, max_value=200, value=int(emp_data.get("Egitim_Saati_Yil", 20)))
        birim_degisim = st.number_input("Birim Değişim Sayısı", min_value=0, max_value=10, value=int(emp_data.get("Birim_Degisim_Sayisi", 0)))
        
        if yaka_tipi == "Beyaz":
            seyahat = st.selectbox("Seyahat Sıklığı", ["Seyahat Yok", "Nadiren", "Siklikla"], 
                                   index=["Seyahat Yok", "Nadiren", "Siklikla"].index(emp_data.get("Seyahat_Sikligi", "Seyahat Yok")) if emp_data.get("Seyahat_Sikligi") in ["Seyahat Yok", "Nadiren", "Siklikla"] else 0)
            vardiya_tipi, vardiya_rotasyon, is_kazasi_val, sendika = "Yok", "Yok", 0, 0
        else:
            seyahat = "Seyahat Yok"
            vardiya_tipi = st.selectbox("Vardiya Tipi", ["Iki Vardiya", "Uc Vardiya", "Yok"], index=0 if emp_data.get("Vardiya_Tipi", "Iki Vardiya") == "Iki Vardiya" else 1 if emp_data.get("Vardiya_Tipi", "Uc Vardiya") == "Uc Vardiya" else 2)
            vardiya_rotasyon = st.selectbox("Rotasyon Sıklığı", ["Haftalik", "Aylik", "Yok"], index=0 if emp_data.get("Vardiya_Rotasyon_Sikligi", "Haftalik") == "Haftalik" else 1 if emp_data.get("Vardiya_Rotasyon_Sikligi", "Aylik") == "Aylik" else 2)
            is_kazasi = st.selectbox("İş Kazası Geçmişi", ["Yok", "Var"], index=int(emp_data.get("Is_Kazasi_Gecmisi", 0)))
            is_kazasi_val = 1 if is_kazasi == "Var" else 0
            sendika = st.slider("Sendika Sözleşmesine Kalan (Ay)", 0, 24, int(emp_data.get("Sendika_Sozlesme_Kalan_Ay", 0)))

    # Business validation logic
    validation_errors = []
    if yaka_tipi == "Beyaz":
        max_possible_exp = yas - 22
        if toplam_tecrube > max_possible_exp:
            validation_errors.append(f"Çelişkili Kıdem: 22 yaşından önce üniversite mezuniyeti varsayılmadığından, {yas} yaşındaki bir beyaz yakanın toplam tecrübesi en fazla {max(0, max_possible_exp)} yıl olabilir (Girilen: {toplam_tecrube} yıl).")
    else:
        max_possible_exp = yas - 18
        if toplam_tecrube > max_possible_exp:
            validation_errors.append(f"Çelişkili Kıdem: 18 yaşından önce tam zamanlı çalışma varsayılmadığından, {yas} yaşındaki bir mavi yakanın toplam tecrübesi en fazla {max(0, max_possible_exp)} yıl olabilir (Girilen: {toplam_tecrube} yıl).")

    if sirketteki_yil > toplam_tecrube:
        validation_errors.append(f"MKE yılı ({sirketteki_yil} yıl) toplam tecrübe yılından ({toplam_tecrube} yıl) büyük olamaz.")
    if ayni_unvanda_yil > sirketteki_yil:
        validation_errors.append(f"Aynı unvanda geçirilen süre ({ayni_unvanda_yil} yıl) MKE kıdem süresinden ({sirketteki_yil} yıl) büyük olamaz.")
    if son_terfi_gecen_yil > ayni_unvanda_yil:
        validation_errors.append(f"Son terfi süresi ({son_terfi_gecen_yil} yıl) aynı unvanda geçen yıldan ({ayni_unvanda_yil} yıl) büyük olamaz.")

    # Lojman check
    if lojman_val == 1 and mesafe_km > 5:
        validation_errors.append("MKE Lojmanlarında kalan çalışanların ev-iş mesafesi en fazla 5 KM olmalıdır.")

    if len(validation_errors) > 0:
        st.error("⚠️ Giriş Verilerinde Çelişki Tespit Edildi:")
        for err in validation_errors:
            st.write(f"- {err}")
        st.warning("Lütfen hesaplama yapabilmek için çelişkili değerleri düzeltin.")
        disabled = True
    else:
        disabled = False
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not disabled:
        # Construct row dictionary
        sim_input = {
            "Cinsiyet": cinsiyet, "Yas": yas, "Medeni_Hal": medeni_hal, "Cocuk_Sayisi": cocuk_sayisi,
            "Ilce": "MKE Lojmanlari" if lojman_val == 1 else "Cankaya",
            "Mesafe_KM": mesafe_km, "Lojman_Kullanimi": lojman_val, "Sertifika_Sayisi": sertifika,
            "Egitim_Saati_Yil": egitim_saat, "Seyahat_Sikligi": seyahat,
            "Toplam_Tecrube_Yil": toplam_tecrube, "Sirketteki_Yil": sirketteki_yil, 
            "MKE_Oncesi_Tecrube": toplam_tecrube - sirketteki_yil, "Ayni_Unvanda_Yil": ayni_unvanda_yil,
            "Son_Terfi_Gecen_Yil": son_terfi_gecen_yil, "Mevcut_Yonetici_Yil": mevcut_yonetici_yil,
            "Birim_Degisim_Sayisi": birim_degisim, "Departman": departman, "Unvan": unvan,
            "Egitim_Seviyesi": egitim_seviyesi, "Maas_TL": maas_tl, "Aylik_Mesai_Saat": aylik_mesai,
            "Vardiya_Tipi": vardiya_tipi, "Vardiya_Rotasyon_Sikligi": vardiya_rotasyon, "Is_Kazasi_Gecmisi": is_kazasi_val,
            "Sendika_Sozlesme_Kalan_Ay": sendika, "Kullanilan_Izin_Gun": kullanilan_izin, 
            "Son_Zam_Orani": son_zam_orani, "Son_Zamdan_Beri_Ay": son_zam_ay, "Performans_Puani": performans,
            "Ortam_Memnuniyeti": ortam_m, "Is_Memnuniyeti": is_m, "Ekip_Uyum_Puani": ekip_m, "Is_Ozel_Hayat_Dengesi": denge_m
        }
        
        # Calculate Risk and SHAP
        df_single = pd.DataFrame([sim_input])
        df_processed = pipeline_islemci.transform(df_single)
        
        categorical_cols_list = ["Medeni_Hal", "Ilce", "Seyahat_Sikligi", "Departman", "Unvan", "Egitim_Seviyesi", "Vardiya_Tipi", "Vardiya_Rotasyon_Sikligi"]
        cat_features = [c for c in categorical_cols_list if c in df_processed.columns]
        df_encoded = pd.get_dummies(df_processed, columns=cat_features, drop_first=True)
        
        cols = cols_beyaz if yaka_tipi == "Beyaz" else cols_mavi
        model = model_beyaz if yaka_tipi == "Beyaz" else model_mavi
        explainer = explainer_beyaz if yaka_tipi == "Beyaz" else explainer_mavi
        thresholds = thresholds_beyaz if yaka_tipi == "Beyaz" else thresholds_mavi
        
        for col in cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[cols]
        
        prob_val = model.predict_proba(df_encoded)[0][1]
        
        # Determine simulated risk category
        opt = OPTIMAL_THRESHOLD_BEYAZ if yaka_tipi == "Beyaz" else OPTIMAL_THRESHOLD_MAVI
        ew = thresholds["early_warning_threshold"]
        
        if prob_val >= opt:
            sim_kategori = "🚨 ACİL MÜDAHALE (Yüksek Risk)"
            sim_color = "#C0392B"
        elif prob_val >= ew:
            sim_kategori = "🟡 ERKEN UYARI (Orta Risk)"
            sim_color = "#F39C12"
        else:
            sim_kategori = "🟢 GÜVENLİ (Düşük Risk)"
            sim_color = "#1A7F5A"
            
        # Display Results
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🎯 Simülasyon Hesaplama Sonuçları</div>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown(f"""
            <div style="background:{sim_color}10; border:2px solid {sim_color}; padding:20px; border-radius:12px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                <span style="font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; color:#64748B; font-weight:700;">Simüle Edilen Ayrılma Riski</span>
                <span style="font-size:3.5rem; font-weight:800; color:{sim_color}; line-height:1.1; margin:10px 0;">%{prob_val*100:.1f}</span>
                <span style="font-weight:700; color:{sim_color}; font-size:1.0rem;">{sim_kategori}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison metrics if using existing worker
            if mod == "Mevcut Çalışan Üzerinde Simülasyon":
                original_risk = emp_data["Risk_Skoru"]
                diff = prob_val - original_risk
                diff_formatted = f"% {diff*100:+.1f}"
                diff_color = "#C0392B" if diff > 0 else "#1A7F5A" if diff < 0 else "#64748B"
                
                st.markdown(f"""
                <div style="margin-top:15px; background:white; border:1px solid #E2E8F0; padding:12px; border-radius:8px; text-align:center;">
                    <div style="font-size:0.8rem; color:#64748B;">Mevcut Referans Riski: <strong>%{original_risk*100:.1f}</strong></div>
                    <div style="font-size:1.0rem; font-weight:700; color:{diff_color}; margin-top:5px;">Değişim: {diff_formatted}</div>
                </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            st.markdown("<div style='text-align:center; font-weight:700; color:#1B2A4A; margin-bottom:10px;'>Ayrılma Riskine Etki Eden Kişisel Faktörler (SHAP Karar Analizi)</div>", unsafe_allow_html=True)
            
            # SHAP Values
            shap_explanation = explainer(df_encoded)
            shap_vals = shap_explanation.values[0]
            if len(shap_explanation.shape) == 3:
                shap_vals = shap_vals[:, 1]
                
            contributions = []
            for col_name, val in zip(cols, shap_vals):
                if abs(val) > 0.001:
                    contributions.append((clean_feature_name(col_name), val))
            
            contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:8]
            
            if len(contributions) > 0:
                feats_sim = [c[0] for c in contributions][::-1]
                vals_sim = [c[1] for c in contributions][::-1]
                colors = ['#C0392B' if v > 0 else '#2980B9' for v in vals_sim]
                
                fig, ax = plt.subplots(figsize=(6, 3.2))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                bars = ax.barh(feats_sim, vals_sim, color=colors, height=0.45)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#64748B')
                ax.spines['bottom'].set_color('#64748B')
                ax.tick_params(colors='#1E293B', labelsize=8)
                ax.xaxis.grid(True, linestyle='--', alpha=0.15, color='#1E293B')
                ax.axvline(x=0, color='#64748B', linestyle='-', linewidth=0.8)
                
                for bar in bars:
                    width = bar.get_width()
                    va_val = 'center'
                    if width > 0:
                        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'+{width:.3f}', 
                                va=va_val, ha='left', color='#C0392B', fontweight='bold', fontsize=7.5)
                    else:
                        ax.text(width - 0.005, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                                va=va_val, ha='right', color='#2980B9', fontweight='bold', fontsize=7.5)
                                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("Belirgin bir risk faktörü bulunamadı.")
                
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SAYFA SEÇİCİ VE YÖNLENDİRME (DİNAMİK SAYFA SEÇİMİ)
# -----------------------------------------------------------------------------
view = st.radio(
    "Aktif Ekranı Seçin:", 
    ["📊 Kurumsal Yönetici Paneli", "👤 Kişi Bazlı Risk Simülatörü (What-If)"], 
    horizontal=True
)

if view == "👤 Kişi Bazlı Risk Simülatörü (What-If)":
    # 2. Bölüm Başlığı (Simülatör için özel)
    st.markdown(f"""
    <div class="dashboard-header">
        <div style="text-align:left;">
            <h2 style="margin:0; font-weight:800; font-size:1.8rem; color:white;">👤 Çalışan İstifa Riski Simülatörü</h2>
            <p style="margin:5px 0 0 0; font-size:0.95rem; color:#94A3B8;">İK Karar Destek ve 'What-If' Simülasyon Ekranı</p>
        </div>
        <div style="text-align:right; background:rgba(255,255,255,0.08); padding:10px 20px; border-radius:8px; border:1px solid rgba(255,255,255,0.1);">
            <small style="color:#94A3B8; text-transform:uppercase; font-weight:700;">Simülasyon Modeli</small>
            <h3 style="margin:0; font-weight:800; color:white;">XGBoost & SHAP</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    render_simulator()
    
    # Yasal Bilgilendirme ve KVKK Uyum Notu (Simülatör için de gösterelim)
    yasal_not = """
    <div style="text-align:center; padding:15px; margin-top:20px; color:#64748B; font-size:0.75rem; border-top:1px solid #E2E8F0;">
        ⚖️ <strong>Yasal Uyum Notu:</strong> Bu yönetim kurulu ekranı tamamen veri odaklı karar destek amacıyla geliştirilmiştir. Ayrılma nedenleri SHAP (Shapley Additive Explanations) işbirlikli oyun teorisi algoritmasıyla hesaplanmış olup, nihai aksiyon kararları insan kaynakları direktörlüğü koordinasyonunda değerlendirilmelidir. KVKK uyumluluğu kapsamında çalışan hassas bilgileri maskelenmiştir.
    </div>
    """
    st.markdown(yasal_not.replace('\n', ' '), unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------------------------
# 2. YAKA FİLTRESİ
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-bottom:15px;'>", unsafe_allow_html=True)
yaka_secimi = st.radio(
    "İnceleme Kapsamı:",
    ["Tüm Çalışanlar", "Beyaz Yaka", "Mavi Yaka"],
    horizontal=True
)
st.markdown("</div>", unsafe_allow_html=True)

df_filtered = df_all.copy()
if yaka_secimi == "Beyaz Yaka":
    df_filtered = df_all[df_all["Yaka_Tipi"] == "Beyaz"].copy()
elif yaka_secimi == "Mavi Yaka":
    df_filtered = df_all[df_all["Yaka_Tipi"] == "Mavi"].copy()

# -----------------------------------------------------------------------------
# 3. BAŞLIK VE KPI SATIRI (ROW 1)
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="dashboard-header">
    <div style="text-align:left;">
        <h2 style="margin:0; font-weight:800; font-size:1.8rem; color:white;">📊 MKE A.Ş. İK Yönetim Kurulu Sunum Paneli ({yaka_secimi})</h2>
        <p style="margin:5px 0 0 0; font-size:0.95rem; color:#94A3B8;">Kurumsal Personel Kayıp Riski ve Müdahale Stratejileri Ekranı</p>
    </div>
    <div style="text-align:right; background:rgba(255,255,255,0.08); padding:10px 20px; border-radius:8px; border:1px solid rgba(255,255,255,0.1);">
        <small style="color:#94A3B8; text-transform:uppercase; font-weight:700;">Toplam Personel</small>
        <h3 style="margin:0; font-weight:800; color:white;">{len(df_filtered):,}</h3>
    </div>
</div>
""", unsafe_allow_html=True)

# Kalibre Edilmiş Risk Kategorilerinin Belirlenmesi
def assign_risk_category(row):
    if row["Yaka_Tipi"] == "Beyaz":
        opt = OPTIMAL_THRESHOLD_BEYAZ
        ew = thresholds_beyaz["early_warning_threshold"]
    else:
        opt = OPTIMAL_THRESHOLD_MAVI
        ew = thresholds_mavi["early_warning_threshold"]
    
    if row["Risk_Skoru"] >= opt:
        return "🚨 ACİL MÜDAHALE (Yüksek Risk)"
    elif row["Risk_Skoru"] >= ew:
        return "🟡 ERKEN UYARI (Orta Risk)"
    else:
        return "🟢 GÜVENLİ (Düşük Risk)"

df_filtered["Risk_Kategorisi"] = df_filtered.apply(assign_risk_category, axis=1)

# KPI Hesaplamaları
acil_sayi = df_filtered[df_filtered["Risk_Kategorisi"].str.contains("ACİL")].shape[0]
erken_uyari_sayi = df_filtered[df_filtered["Risk_Kategorisi"].str.contains("ERKEN")].shape[0]
guvenli_sayi = df_filtered[df_filtered["Risk_Kategorisi"].str.contains("GÜVENLİ")].shape[0]
ortalama_risk = df_filtered["Risk_Skoru"].mean() * 100

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""<div class="kpi-container" style="border-bottom-color: #C0392B;">
<div class="kpi-title">🚨 Acil Müdahale</div>
<div class="kpi-value">{acil_sayi} Kişi</div>
<div class="kpi-subtext">Kritik Eşik Üzerindeki Yüksek Risk</div>
</div>""", unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""<div class="kpi-container" style="border-bottom-color: #F39C12;">
<div class="kpi-title">🟡 Erken Uyarı</div>
<div class="kpi-value">{erken_uyari_sayi} Kişi</div>
<div class="kpi-subtext">Takip Listesindeki Orta Risk</div>
</div>""", unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""<div class="kpi-container" style="border-bottom-color: #1A7F5A;">
<div class="kpi-title">🟢 Güvenli Bölge</div>
<div class="kpi-value">{guvenli_sayi} Kişi</div>
<div class="kpi-subtext">Normal Değerler Altındaki Personel</div>
</div>""", unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""<div class="kpi-container" style="border-bottom-color: #1B2A4A;">
<div class="kpi-title">Kurum Risk Ortalaması</div>
<div class="kpi-value">%{ortalama_risk:.1f}</div>
<div class="kpi-subtext">Tüm Kurum Genel İstifa İndeksi</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. ANA GRAFİKLER (ROW 2)
# -----------------------------------------------------------------------------
if yaka_secimi == "Tüm Çalışanlar":
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Hangi departman en yüksek kayıp riskine sahip?</div>", unsafe_allow_html=True)
        
        # Departman bazlı ortalama risk skorları
        dept_risk = df_filtered.groupby("Departman")["Risk_Skoru"].mean().sort_values(ascending=True) * 100
        
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        bars = ax.barh(dept_risk.index, dept_risk.values, color='#1B2A4A', edgecolor='none', height=0.55)
        bars[-1].set_color('#C0392B') 
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#64748B')
        ax.spines['bottom'].set_color('#64748B')
        ax.tick_params(colors='#1E293B', labelsize=10)
        ax.xaxis.grid(True, linestyle='--', alpha=0.15, color='#1E293B')
        ax.set_axisbelow(True)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'%{width:.1f}', 
                    va='center', ha='left', color='#1E293B', fontweight='bold', fontsize=9)
                    
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Yaka sınıflarının risk dağılımı nasıl karşılaştırılıyor?</div>", unsafe_allow_html=True)
        
        # Yaka bazlı risk ortalamaları
        yaka_risk = df_filtered.groupby("Yaka_Tipi")["Risk_Skoru"].mean() * 100
        
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        categories = ['Beyaz Yaka', 'Mavi Yaka']
        values = [yaka_risk.get('Beyaz', 0), yaka_risk.get('Mavi', 0)]
        
        bars = ax.bar(categories, values, color=['#34495E', '#1B2A4A'], width=0.45)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#64748B')
        ax.spines['bottom'].set_color('#64748B')
        ax.tick_params(colors='#1E293B', labelsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.15, color='#1E293B')
        ax.set_axisbelow(True)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'%{yval:.1f}', 
                    va='bottom', ha='center', color='#1E293B', fontweight='bold', fontsize=10)
                    
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Specific yaka selected, render department risk chart full width!
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Hangi departman en yüksek kayıp riskine sahip?</div>", unsafe_allow_html=True)
    
    dept_risk = df_filtered.groupby("Departman")["Risk_Skoru"].mean().sort_values(ascending=True) * 100
    
    fig, ax = plt.subplots(figsize=(14, 3.8))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    bars = ax.barh(dept_risk.index, dept_risk.values, color='#1B2A4A', edgecolor='none', height=0.55)
    bars[-1].set_color('#C0392B') 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#64748B')
    ax.spines['bottom'].set_color('#64748B')
    ax.tick_params(colors='#1E293B', labelsize=10)
    ax.xaxis.grid(True, linestyle='--', alpha=0.15, color='#1E293B')
    ax.set_axisbelow(True)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'%{width:.1f}', 
                va='center', ha='left', color='#1E293B', fontweight='bold', fontsize=9)
                
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. DETAYLAR VE ETKENLER (ROW 3 - 2 KOLON)
# -----------------------------------------------------------------------------
col_det1, col_det2 = st.columns(2)

with col_det1:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>İstifayı tetikleyen en kritik 5 faktör hangisi? (Global SHAP)</div>", unsafe_allow_html=True)
    
    def render_global_importances(model, cols):
        importances = dict(zip(cols, model.feature_importances_))
        translated_importances = {}
        for k, v in importances.items():
            tr_k = clean_feature_name(k)
            translated_importances[tr_k] = translated_importances.get(tr_k, 0) + v
            
        sorted_importances = sorted(translated_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        
        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        feats = [item[0] for item in sorted_importances][::-1]
        imp_vals = [item[1] for item in sorted_importances][::-1]
        
        bars = ax.barh(feats, imp_vals, color='#F39C12', edgecolor='none', height=0.55)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#64748B')
        ax.spines['bottom'].set_color('#64748B')
        ax.tick_params(colors='#1E293B', labelsize=10)
        ax.xaxis.grid(True, linestyle='--', alpha=0.15, color='#1E293B')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        st.pyplot(fig)

    if yaka_secimi == "Beyaz Yaka":
        render_global_importances(model_beyaz, cols_beyaz)
    elif yaka_secimi == "Mavi Yaka":
        render_global_importances(model_mavi, cols_mavi)
    else:
        tab_glob_b, tab_glob_m = st.tabs(["Beyaz Yaka Etkenleri", "Mavi Yaka Etkenleri"])
        with tab_glob_b:
            render_global_importances(model_beyaz, cols_beyaz)
        with tab_glob_m:
            render_global_importances(model_mavi, cols_mavi)
            
    st.markdown("</div>", unsafe_allow_html=True)

with col_det2:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Kurum kıdemi (MKE Yılı) riski nasıl etkiliyor?</div>", unsafe_allow_html=True)
    
    kidem_risk = df_filtered.groupby("Sirketteki_Yil")["Risk_Skoru"].mean() * 100
    kidem_risk = kidem_risk.sort_index().head(12)
    
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    ax.plot(kidem_risk.index, kidem_risk.values, color='#1B2A4A', marker='o', linewidth=2.5, markersize=6)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#64748B')
    ax.spines['bottom'].set_color('#64748B')
    ax.tick_params(colors='#1E293B', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.15, color='#1E293B')
    ax.set_xlabel("MKE Hizmet Yılı (Kıdem)", color='#1E293B', fontweight='bold', fontsize=9)
    ax.set_ylabel("Ortalama Risk Oranı (%)", color='#1E293B', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. AKSİYON TABLOSU (ROW 5)
# -----------------------------------------------------------------------------
st.markdown("<div class='content-card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>🚨 Acil Müdahale Edilmesi Gereken En Yüksek Riskli 10 Çalışan</div>", unsafe_allow_html=True)
if yaka_secimi == "Tüm Çalışanlar":
    yaka_desc = "kurum genelinde"
else:
    yaka_desc = f"seçilen yaka grubunda ({yaka_secimi.lower()})"
st.write(f"Aşağıdaki liste, {yaka_desc} işten ayrılma riski en yüksek olan ve İK tarafından öncelikli görüşme yapılması gereken 10 personeli listeler. Ana sebepler SHAP analizi ile kişiye özel hesaplanmıştır.")

# En yüksek riskli 10 çalışanı seçelim
top_10 = df_filtered.sort_values(by="Risk_Skoru", ascending=False).head(10).copy()

# Bu 10 çalışanın bireysel risk sebeplerini bulalım (SHAP ile)
top_reasons = []
for idx, row in top_10.iterrows():
    yaka = row["Yaka_Tipi"]
    yaka_idx = int(row["original_yaka_index"])
    
    # İlgili yaka verisini bulup encoded halini çekme
    if yaka == "Beyaz":
        row_encoded = df_beyaz_enc.iloc[yaka_idx]
        shap_vals = explainer_beyaz(pd.DataFrame([row_encoded]))[0].values
        cols = cols_beyaz
    else:
        row_encoded = df_mavi_enc.iloc[yaka_idx]
        shap_vals = explainer_mavi(pd.DataFrame([row_encoded]))[0].values
        cols = cols_mavi
        
    if len(shap_vals.shape) == 2:
        shap_vals = shap_vals[:, 1]
        
    max_idx = np.argmax(shap_vals)
    best_feat = cols[max_idx]
    top_reasons.append(clean_feature_name(best_feat))

top_10["Ana_Sebep"] = top_reasons

# Tabloyu oluşturma
table_html = """
<table class="action-table">
    <thead>
        <tr>
            <th>Personel ID</th>
            <th>Yaka Sınıfı</th>
            <th>Departman / Fabrika</th>
            <th>Mevcut Unvan</th>
            <th style="text-align:right;">İstifa Riski</th>
            <th>Risk Kategorisi</th>
            <th>Birincil Ayrılma Tetikleyicisi (SHAP)</th>
        </tr>
    </thead>
    <tbody>
"""

for idx, row in top_10.iterrows():
    risk_formatted = f"% {row['Risk_Skoru']*100:.1f}"
    
    # Risk kategorisine göre renk ve badge
    if "ACİL" in row['Risk_Kategorisi']:
        score_color = "#C0392B"
        badge_text = "🚨 ACİL MÜDAHALE"
    elif "ERKEN" in row['Risk_Kategorisi']:
        score_color = "#F39C12"
        badge_text = "🟡 ERKEN UYARI"
    else:
        score_color = "#1A7F5A"
        badge_text = "🟢 GÜVENLİ"
    
    table_html += f"""<tr>
<td><strong>{row['Personel_ID']}</strong></td>
<td>{row['Yaka_Tipi']} Yaka</td>
<td>{row['Departman']}</td>
<td>{row['Unvan']}</td>
<td style="text-align:right; font-weight:800; color:{score_color};">{risk_formatted}</td>
<td style="font-weight:700; color:{score_color};">{badge_text}</td>
<td style="font-weight:600; color:#1B2A4A;">🔴 {row['Ana_Sebep']}</td>
</tr>"""

table_html += "</tbody></table>"

# Markdown parser'ın girintileri kod bloğu olarak algılamasını engellemek için satır sonlarını temizleyip basıyoruz
st.markdown(table_html.replace('\n', ' '), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Yasal Bilgilendirme ve KVKK Uyum Notu
yasal_not = """
<div style="text-align:center; padding:15px; margin-top:20px; color:#64748B; font-size:0.75rem; border-top:1px solid #E2E8F0;">
    ⚖️ <strong>Yasal Uyum Notu:</strong> Bu yönetim kurulu ekranı tamamen veri odaklı karar destek amacıyla geliştirilmiştir. Ayrılma nedenleri SHAP (Shapley Additive Explanations) işbirlikli oyun teorisi algoritmasıyla hesaplanmış olup, nihai aksiyon kararları insan kaynakları direktörlüğü koordinasyonunda değerlendirilmelidir. KVKK uyumluluğu kapsamında çalışan hassas bilgileri maskelenmiştir.
</div>
"""
st.markdown(yasal_not.replace('\n', ' '), unsafe_allow_html=True)