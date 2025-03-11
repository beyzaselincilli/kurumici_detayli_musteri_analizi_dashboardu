# Kurumsal Müşteri Analitik Platformu

Bu proje, kurumsal müşteri verilerini analiz eden, müşteri segmentasyonu yapan ve gelecekteki satışları tahmin eden kapsamlı bir veri bilimi uygulamasıdır.

## Özellikler

- 🔍 Müşteri Segmentasyonu (K-means clustering)
- 📈 Satış Tahmin Modeli (XGBoost)
- 📊 İnteraktif Görselleştirmeler
- 📱 Kullanıcı Dostu Web Arayüzü
- 📝 Otomatik Raporlama
- 📊 MLflow ile Model Takibi

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. MLflow sunucusunu başlatın:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

3. Streamlit uygulamasını çalıştırın:
```bash
cd src
streamlit run app.py
```

## Veri Formatı

Uygulamanın beklediği CSV dosyası formatı:

- customer_id: Müşteri ID
- recency: Son alışverişten bu yana geçen gün sayısı
- frequency: Toplam alışveriş sayısı
- monetary: Toplam harcama tutarı
- sales: Aylık ortalama satış tutarı
- [diğer özellikler]: Demografik ve davranışsal özellikler

## Proje Yapısı

```
customer_analytics/
├── data/               # Veri dosyaları
├── notebooks/          # Jupyter notebooks
├── src/               # Kaynak kodlar
│   ├── customer_analysis.py
│   └── app.py
├── models/            # Eğitilmiş modeller
├── reports/           # Otomatik oluşturulan raporlar
└── requirements.txt   # Bağımlılıklar
```

## Teknik Detaylar

### Müşteri Segmentasyonu
- RFM (Recency, Frequency, Monetary) analizi
- K-means clustering algoritması
- Optimal küme sayısı belirleme

### Satış Tahmin Modeli
- XGBoost regresyon modeli
- Özellik önem analizi
- Cross-validation
- Model performans metrikleri

### Web Arayüzü
- Streamlit dashboard
- İnteraktif görselleştirmeler
- Gerçek zamanlı model eğitimi
- Özelleştirilebilir parametreler

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun 