# src/pipeline.py
import pandas as pd
import numpy as np

class MKEDataPipeline:
    def __init__(self, medyan_maas_sozlugu=None):
        self.medyan_maas_sozlugu = medyan_maas_sozlugu

    def fit(self, df):
        # Senin veri setindeki 'Departman', 'Unvan' ve 'Maas_TL' kolonlarına göre medyanları hesaplıyor
        self.medyan_maas_sozlugu = df.groupby(["Departman", "Unvan"])["Maas_TL"].median().to_dict()
        return self

    def transform(self, df_input):
        df = df_input.copy()
        
        # 1. MAAŞ ENDEKSİ (COMPA-RATIO)
        if self.medyan_maas_sozlugu:
            df["Grup_Medyan_Maas"] = df.set_index(["Departman", "Unvan"]).index.map(self.medyan_maas_sozlugu)
            df["Maas_Endeksi"] = df["Maas_TL"] / df["Grup_Medyan_Maas"]
            df["Maas_Endeksi"] = df["Maas_Endeksi"].fillna(1.0)
        else:
            df["Maas_Endeksi"] = 1.0
            
        # 2. KARİYER TIKANMA ORANI
        # Senin veri setindeki 'Ayni_Unvanda_Yil' ve 'Sirketteki_Yil' kolonlarını oranlıyoruz
        df["Karier_Tikanma_Orani"] = df["Ayni_Unvanda_Yil"] / df["Sirketteki_Yil"].replace(0, 1)
        df["Karier_Tikanma_Orani"] = df["Karier_Tikanma_Orani"].fillna(0)
        
        # 3. TÜKENMİŞLİK / İŞ YÜKÜ ENDEKSİ (WORKLOAD / BURNOUT INDEX)
        # Sorumluluk Çarpanı: Çalışanın sertifika sayısı sorumluluk seviyesini ve iş yükünü artırır.
        # İş Yükü Oranı = (Aylık Fazla Mesai * Sorumluluk Çarpanı) / (Kullanılan İzin + 1)
        sorumluluk_carpani = 1 + 0.2 * df["Sertifika_Sayisi"]
        df["Is_Yuku_Orani"] = (df["Aylik_Mesai_Saat"] * sorumluluk_carpani) / (df["Kullanilan_Izin_Gun"] + 1)
        
        # VERI SIZINTISI (LEAKAGE) VE GEREKSİZ KOLON TEMİZLİĞİ
        # Modelin kafasını karıştıracak id'leri, KVKK/etik ihlali oluşturacak korumalı öznitelikleri (Cinsiyet)
        # ve ara hesaplama kolonlarını uçuruyoruz.
        drop_cols = [
            "Personel_ID", "Grup_Medyan_Maas", "Anket_Standart_Sapma", 
            "Bitis_Tarihi", "Istifa_Nedeni", "Cinsiyet",
            "Ortam_Memnuniyeti", "Is_Memnuniyeti", "Ekip_Uyum_Puani", "Is_Ozel_Hayat_Dengesi"
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        return df