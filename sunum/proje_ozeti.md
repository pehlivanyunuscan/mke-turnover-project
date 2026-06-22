# MKE A.Ş. Çalışan Bağlılığı ve İstifa Riski Tahminleme Projesi
## Sunum Hazırlığı İçin Kapsamlı Proje Özet Dokümanı

Bu doküman, projeyi sunacak ekip arkadaşlarınıza genel iş mantığını, teknik altyapıyı ve veri mühendisliği detaylarını aktarmak üzere hazırlanmıştır. Buradaki özet maddeleri, formülleri ve modelleri kullanarak sunum slaytlarını kolayca oluşturabilirler.

---

### 1. PROJENİN AMACI VE ELE ALDIĞI İŞ PROBLEMİ
* **Proaktif İK Yaklaşımı:** Personel istifa dilekçesini vermeden (reaktif süreç başlamadan) aylar önce risk altındaki kilit personeli tespit edip bağlılığı artıracak aksiyonlar almak.
* **Kurumsal Hafızanın Korunması:** Özellikle savunma sanayii ve mühendislik rolleri için tecrübe/bilgi birikimi kaybını (attrition/turnover) en aza indirmek.
* **İK Yerine Koyma Maliyeti (ROI):** Nitelikli bir personelin ayrılması durumunda oluşan yeni işe alım, eğitim (onboarding) ve projelerdeki gecikme maliyetlerinin önüne geçmek.

---

### 2. İŞ GERÇEKLİĞİ VE MODEL HİZALAMASI (ALIGNMENT)
* **Mavi-Beyaz Yaka Ayrımı:** Mavi yaka fabrika/operasyon ekipleri ile beyaz yaka mühendis/uzman kadrolarının istifa motivasyonları tamamen farklı olduğundan sistem iki bağımsız yapay zeka modeliyle çalışmaktadır.
* **İstifa Oranı Kalibrasyonu:** Pazar gerçeklerine uygun olarak mavi yaka istifa oranı **%7.5**, beyaz yaka istifa oranı ise **%17.5** olacak şekilde modeller kalibre edilmiştir.
* **Memnuniyet Anketi Bağımsızlığı:** Yılda bir yapılan, çalışanların çekincelerle gerçek dışı/maskelenmiş cevaplar verebildiği memnuniyet anketleri model eğitiminden tamamen çıkarılmıştır. Tahminler nesnel operasyonel verilere dayanır.
* **Çocuk Sayısı Etkisinin Kırılması:** Çocuk sayısı değişkeninin tahminleri mantıksız düzeyde domine etmesi engellenmiş, gerçekçi iş yeri faktörleri (maaş adaleti, fazla mesai, unvan süresi) öne çıkarılmıştır.

---

### 3. GELİŞMİŞ DEĞİŞKEN MÜHENDİSLİĞİ (FEATURE ENGINEERING)
Modelin tahmin gücünü artıran ve ham verilerden türetilen 3 temel indeks:

* **A. Compa-Ratio (Maaş Endeksi):**
  * **Açıklama:** Çalışanın aldığı maaşın, kendi departmanında aynı unvana sahip çalışanların medyan maaşına oranı.
  * **İş Mantığı:** Kurum içi ücret adaletini ölçer. Oranın $1.0$'ın altında olması çalışanın akranlarına göre az kazandığını ve riskli olduğunu gösterir.
  * **Formül:** 
    $$\text{Maaş Endeksi} = \frac{\text{Çalışan Maaşı}}{\text{Departman \& Unvan Grubu Medyan Maaşı}}$$

* **B. Kariyer Tıkanma Oranı (Career Stagnation):**
  * **Açıklama:** Personelin mevcut unvanda kaldığı sürenin MKE kıdemine oranı.
  * **İş Mantığı:** Çalışanın terfi alamadan yerinde sayıp saymadığını ölçer. Oran $1.0$'a yaklaştıkça kariyer tıkanması ve istifa eğilimi artar.
  * **Formül:** 
    $$\text{Kariyer Tıkanma Oranı} = \frac{\text{Aynı Unvanda Geçen Yıl}}{\text{Şirketteki Toplam Yıl}}$$

* **C. İş Yükü ve Tükenmişlik Oranı (Workload / Burnout Index):**
  * **Açıklama:** Fazla mesainin ve alınan sertifikalardan gelen ek sorumlulukların, kullanılan izin günlerine oranı.
  * **İş Mantığı:** Çok mesai yapan, sertifikaları sebebiyle sorumluluk yüklenen ama izin kullanmayan personelin tükenme (burnout) eşiğini ölçer.
  * **Formül:** 
    $$\text{İş Yükü Oranı} = \frac{\text{Aylık Fazla Mesai Saat} \times (1 + 0.2 \times \text{Sertifika Sayısı})}{\text{Kullanılan İzin Günü} + 1}$$

---

### 4. MAKİNE ÖĞRENMESİ VE GÜVENİLİR YAPAY ZEKA (TRUSTWORTHY AI)
* **Model Algoritması:** Hızlı, yüksek performanslı ve karar ağaçları mantığına dayanan güçlü tahmin motoru **XGBoost Classifier**.
* **Monotonluk Kısıtlamaları (Monotone Constraints):** Yapay zekanın iş mantığıyla çelişen kararlar vermesini önlemek için yerleştirilen matematiksel sınırlar:
  * *Maaş / Zam artışı* risk skorunu **asla yükseltemez** (Negatif Kısıt).
  * *Fazla mesai / Ev-iş mesafesi artışı* risk skorunu **asla düşüremez** (Pozitif Kısıt).
* **Karar Eşikleri (Threshold Calibration):**
  * Beyaz Yaka Optimal Karar Eşiği: **0.55**
  * Mavi Yaka Optimal Karar Eşiği: **0.50**
  * **Erken Uyarı Eşiği (Early Warning):** İstifa edecek personellerin en az %80'ini (Recall hedefi) önceden yakalayabilmek için optimize edilmiş daha hassas eşik değeridir.

---

### 5. AÇIKLANABİLİR YAPAY ZEKA: SHAP ENTEGRASYONU
* **Kara Kutu Probleminin Çözümü:** Yapay zekanın sadece "istifa riski %85" demekle kalmayıp, o çalışanın ayrılma riskini tetikleyen **ilk 3 nedeni** yüzdesel katkılarıyla sunması.
* **İK İş Dili Çevirisi:** SHAP kütüphanesinden çıkan teknik değişken isimleri İK uzmanlarının anlayacağı dillere çevrilmiştir:
  * Örn: `Ayni_Unvanda_Yil` $\rightarrow$ "Kariyer İlerlemesinde Tıkanma"
  * Örn: `Aylik_Mesai_Saat` $\rightarrow$ "Aşırı Fazla Mesai Saatleri"
  * Örn: `Maas_Endeksi` $\rightarrow$ "Düşük Maaş Politikası (Rol Ortalaması Altı)"

---

### 6. İK YÖNETİCİ PANELİ VE WHAT-IF SİMÜLATÖRÜ YETENEKLERİ
* **Dinamik Filtreleme:** Tüm Kurum, Sadece Beyaz Yaka veya Sadece Mavi Yaka bazlı anlık kırılımlar.
* **Aksiyon Tablosu:** İstifa riski en yüksek 10 çalışanı, departmanları, unvanları ve **kişiye özel ayrılma tetikleyicileriyle** listeler.
* **Geri Kazanım Tavsiyeleri (Yeni):** Her bir çalışanın birincil risk sebebine (SHAP çıktısı) göre otomatik atanan özel eylem reçetesi. Bu tavsiyeler hem Kurumsal Paneldeki en yüksek riskli 10 çalışan tablosuna ("Önerilen Geri Kazanım Aksiyonu" sütunu olarak) hem de Simülatör sonuç ekranına ("İK Aksiyon ve Geri Kazanım Tavsiyesi" başlığıyla) entegre edilmiştir.
* **What-If (Karar Destek) Simülatörü:** Yüksek riskli bir çalışanın şartları (maaşı, mesaisi, lojman kullanımı vb.) iyileştirildiğinde risk oranındaki anlık düşüşü gösteren simülasyon arayüzü.
* **İş Kuralları Doğrulaması (Validation):** Simülatörde mantıksız veri girişlerini (Örn: Lojmanda kaldığı halde işe uzaklığının 40 km girilmesi, yaşından daha fazla tecrübe girilmesi gibi) engelleyen 5 aşamalı İK doğrulama mekanizması.

---

### 7. ÖNERİLEN STRATEJİK İK AKSİYONLARI
* **Ücret Adaleti:** Maaş Endeksi (Compa-Ratio) $0.80$'in altında kalan yüksek performanslı personele hakkaniyet zammı yapılması.
* **Tükenmişlik Yönetimi:** İş Yükü Oranı $5.0$'in üzerinde olan çalışanlara zorunlu yıllık izin kullandırılması ve fazla mesailerin sınırlandırılması.
* **Rotasyon & Kariyer:** Kariyer Tıkanma Oranı yüksek personele yönelik mentorluk veya departman içi birim değişikliği fırsatları sunulması.
