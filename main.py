import pandas as pd
import numpy as np

# Tekrarlanabilirlik için seed sabitliyoruz
np.random.seed(42)
n_samples = 6000

# --- 1. MKE DERECE-KADEME MAAŞ VE UNVAN SKALASI (GÜNCELLENDİ) ---
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

rows = []
for i in range(n_samples):
    p_id = f"MKE-{10000 + i}"
    yaka = np.random.choice(["Mavi", "Beyaz"], p=[0.65, 0.35])
    cinsiyet = np.random.choice(["Erkek", "Kadin"], p=[0.85, 0.15] if yaka == "Mavi" else [0.75, 0.25])
    
    # 1. DÜZELTME: Yaş ve Kıdem İlişkisini Gerçekçi Yapıyoruz
    yas = np.random.randint(22, 60)
    if yaka == "Beyaz":
        max_tecrube = max(1, yas - 23) # Mezuniyet yaşını 23 kabul ettik
    else:
        max_tecrube = max(1, yas - 18) # Meslek lisesi mezuniyetini 18 kabul ettik
        
    toplam_tecrube = np.random.randint(1, max_tecrube + 1)
    sirketteki_yil = np.random.randint(1, toplam_tecrube + 1)
    mke_oncesi_tecrube = toplam_tecrube - sirketteki_yil
    
    # Unvan ve Departman Seçimi (Bakis Uzman Çıkarıldı)
    if yaka == "Mavi":
        departman = np.random.choice(["Muhimmat Fabrikasi", "Silah Fabrikasi", "Gazi Fisek Fabrikasi", "Bakim Onarim"])
        if toplam_tecrube < 4: unvan = "Operator"
        elif toplam_tecrube < 8: unvan = np.random.choice(["Operator", "Teknisyen"], p=[0.4, 0.6])
        elif toplam_tecrube < 14: unvan = np.random.choice(["Teknisyen", "Atolye Ustasi"], p=[0.3, 0.7])
        else: unvan = np.random.choice(["Atolye Ustasi", "Basusta"], p=[0.4, 0.6])
        egitim = np.random.choice(["Lise", "On Lisans"], p=[0.70, 0.30])
    else:
        departman = np.random.choice(["AR-GE Tasarim", "Bilgi Teknolojileri", "IK", "Satinalma", "Kalite Yonetim"])
        if toplam_tecrube < 4: unvan = np.random.choice(["Muhendis", "Uzman"], p=[0.6, 0.4])
        elif toplam_tecrube < 9: unvan = np.random.choice(["Muhendis", "Uzman", "Kidemli Muhendis"], p=[0.2, 0.2, 0.6])
        elif toplam_tecrube < 14: unvan = "Kidemli Muhendis"
        else: unvan = np.random.choice(["Kidemli Muhendis", "Takim Lideri"], p=[0.4, 0.6])
        egitim = np.random.choice(["Lisans", "Yuksek Lisans", "Doktora"], p=[0.60, 0.33, 0.07])

    # Maaş Skalası Kuralı
    min_m = mke_skala[yaka][unvan]["min"]
    max_m = mke_skala[yaka][unvan]["max"]
    maas_tl = int(min_m + (max_m - min_m) * (sirketteki_yil / max(1, toplam_tecrube)) + np.random.randint(-2000, 2000))
    
    ayni_unvanda_yil = np.random.randint(0, sirketteki_yil) if sirketteki_yil > 1 else 0
    son_terfi_gecen_yil = np.random.randint(0, ayni_unvanda_yil + 1) if ayni_unvanda_yil > 0 else 0
    mevcut_yonetici_yil = np.random.randint(1, min(6, sirketteki_yil + 2))
    birim_degisim_sayisi = np.random.randint(0, max(1, int(sirketteki_yil / 4)))
    
    medeni_hal = np.random.choice(["Evli", "Bekar"], p=[0.70, 0.30] if yas > 28 else [0.20, 0.80])
    cocuk_sayisi = np.random.choice([1, 2, 3], p=[0.45, 0.45, 0.10]) if medeni_hal == "Evli" else 0
    lojman = np.random.choice([0, 1], p=[0.65, 0.35] if yaka == "Mavi" else [0.85, 0.15])
    
    # 2. DÜZELTME: Lojman ve İlçe Çelişkisini Çözüyoruz
    if lojman == 1:
        mesafe_km = np.random.randint(1, 4)
        ilce = "MKE Lojmanlari"
    else:
        mesafe_km = np.random.randint(8, 55)
        ilce = np.random.choice(["Cankaya", "Etimesgut", "Yenimahalle", "Mamaj", "Kalebasi"])

    # Vardiya ve Mesai Koşulları
    if yaka == "Mavi":
        vardiya_tipi = np.random.choice(["Iki Vardiya", "Uc Vardiya"], p=[0.50, 0.50])
        vardiya_rotasyon = np.random.choice(["Haftalik", "Aylik"], p=[0.70, 0.30])
        aylik_mesai = np.random.randint(20, 65) if vardiya_tipi == "Uc Vardiya" else np.random.randint(10, 40)
        is_kazasi = np.random.choice([0, 1], p=[0.88, 0.12] if vardiya_tipi == "Uc Vardiya" else [0.95, 0.05])
        revir_ziyaret = np.random.randint(3, 11) if vardiya_tipi == "Uc Vardiya" else np.random.randint(0, 5)
        sendika_kalan_ay = np.random.randint(1, 25)
        seyahat = "Seyahat Yok"
    else:
        vardiya_tipi, vardiya_rotasyon = "Yok", "Yok"
        aylik_mesai = np.random.randint(5, 45)
        is_kazasi, revir_ziyaret, sendika_kalan_ay = 0, np.random.randint(0, 3), 0
        seyahat = np.random.choice(["Seyahat Yok", "Nadiren", "Siklikla"], p=[0.65, 0.25, 0.10])

    sertifika = np.random.randint(0, 3) if yaka == "Mavi" else np.random.randint(1, 7)
    egitim_saat = np.random.randint(4, 20) if yaka == "Mavi" else np.random.randint(15, 50)
    kullanilan_izin = np.random.randint(14, 26)
    
    performans = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.55, 0.18, 0.07])
    son_zam_orani = np.random.uniform(30.0, 55.0)
    son_zam_ay = np.random.randint(1, 13)

    # Anket Davranışları
    gercekten_mutsuz_mu = False
    if yaka == "Beyaz" and ayni_unvanda_yil > 4 and performans >= 4: gercekten_mutsuz_mu = True
    if yaka == "Mavi" and vardiya_tipi == "Uc Vardiya" and mesafe_km > 35 and lojman == 0: gercekten_mutsuz_mu = True
    if aylik_mesai > 50: gercekten_mutsuz_mu = True

    if gercekten_mutsuz_mu:
        if np.random.rand() < 0.35: # Sinsi profil
            ortam_m, is_m, ekip_m, denge_m = 5, 5, 5, 5
        else:
            ortam_m, is_m, ekip_m, denge_m = np.random.choice([1,2]), np.random.choice([1,2]), np.random.choice([2,3]), np.random.choice([1,2])
    else:
        ortam_m, is_m, ekip_m, denge_m = np.random.choice([3,4,5], p=[0.2, 0.5, 0.3]), np.random.choice([3,4,5], p=[0.1, 0.6, 0.3]), np.random.choice([4,5], p=[0.4, 0.6]), np.random.choice([3,4,5], p=[0.3, 0.5, 0.2])

    if mesafe_km > 40 and not gercekten_mutsuz_mu:
        denge_m = max(1, denge_m - 2)

    # 3. DÜZELTME: İstifa Mantığını Yıldız Çalışanlara ve Hakiki Risklere Göre Kuruyoruz
    score = 0
    if gercekten_mutsuz_mu: score += 4
    if maas_tl < (min_m + (max_m - min_m) * 0.25): score += 3 # Unvan grubuna göre bariz az kazanıyorsa
    if son_zam_ay > 9: score += 1.5
    if yaka == "Mavi" and is_kazasi == 1: score += 2
    if yaka == "Mavi" and sendika_kalan_ay <= 3: score -= 3.5 # Sendika zammı kilidi
    
    # Yıldız Çalışan Tuzağı: Performansı yüksek (4 veya 5) ama 3 yıldır yerinde sayıyor ve mutsuzsa
    if performans >= 4 and son_terfi_gecen_yil >= 3: score += 3

    # Yaka Tipi İstifa Oranları Ayarlaması: Mavi Yaka istifa oranı Beyaz Yaka'dan daha düşük olmalıdır.
    if yaka == "Mavi":
        score -= 2.0
    else:
        score += 1.2

    prob = 1 / (1 + np.exp(-(score - 4.5)))
    istifa = np.random.choice([1, 0], p=[prob, 1 - prob])

    rows.append([p_id, yaka, cinsiyet, yas, medeni_hal, cocuk_sayisi, ilce, mesafe_km, lojman, sertifika, egitim_saat, seyahat,
                 toplam_tecrube, sirketteki_yil, mke_oncesi_tecrube, ayni_unvanda_yil, son_terfi_gecen_yil, mevcut_yonetici_yil,
                 birim_degisim_sayisi, departman, unvan, egitim, maas_tl, aylik_mesai, vardiya_tipi, vardiya_rotasyon,
                 is_kazasi, sendika_kalan_ay, kullanilan_izin, son_zam_orani, son_zam_ay, performans, ortam_m, is_m, ekip_m, denge_m, istifa])

columns = ["Personel_ID", "Yaka_Tipi", "Cinsiyet", "Yas", "Medeni_Hal", "Cocuk_Sayisi", "Ilce", "Mesafe_KM", "Lojman_Kullanimi", "Sertifika_Sayisi", "Egitim_Saati_Yil", "Seyahat_Sikligi",
           "Toplam_Tecrube_Yil", "Sirketteki_Yil", "MKE_Oncesi_Tecrube", "Ayni_Unvanda_Yil", "Son_Terfi_Gecen_Yil", "Mevcut_Yonetici_Yil",
           "Birim_Degisim_Sayisi", "Departman", "Unvan", "Egitim_Seviyesi", "Maas_TL", "Aylik_Mesai_Saat", "Vardiya_Tipi", "Vardiya_Rotasyon_Sikligi",
           "Is_Kazasi_Gecmisi", "Sendika_Sozlesme_Kalan_Ay", "Kullanilan_Izin_Gun", "Son_Zam_Orani", "Son_Zamdan_Beri_Ay", "Performans_Puani", "Ortam_Memnuniyeti", "Is_Memnuniyeti", "Ekip_Uyum_Puani", "Is_Ozel_Hayat_Dengesi", "Istifa_Etti_Mi"]

df_perfect = pd.DataFrame(rows, columns=columns)
df_perfect.to_csv("data/dataset.csv", index=False)
print("Kusursuz Veri Seti Üretildi! İstifa Oranı: ", df_perfect["Istifa_Etti_Mi"].mean())