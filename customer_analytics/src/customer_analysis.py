import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import logging
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os

class CustomerAnalytics:
    def __init__(self):
        self.logger = self._setup_logger()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kmeans_model = None
        self.sales_model = None
        self.model_path = "../models/sales_prediction_model.joblib"
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_data(self, filepath):
        """Veri setini yükle ve temel temizleme işlemlerini gerçekleştir"""
        try:
            self.data = pd.read_csv(filepath)
            self.logger.info(f"Veri başarıyla yüklendi. Boyut: {self.data.shape}")
            return self.data
        except Exception as e:
            self.logger.error(f"Veri yükleme hatası: {str(e)}")
            raise

    def preprocess_data(self):
        """Veri ön işleme adımları"""
        try:
            # Eksik değerleri doldur
            self.data = self.data.fillna(self.data.mean())
            
            # Kategorik değişkenleri dönüştür
            cat_columns = self.data.select_dtypes(include=['object']).columns
            self.data = pd.get_dummies(self.data, columns=cat_columns)
            
            self.logger.info("Veri ön işleme tamamlandı")
            return self.data
        except Exception as e:
            self.logger.error(f"Veri ön işleme hatası: {str(e)}")
            raise

    def perform_customer_segmentation(self, n_clusters=4):
        """Müşteri segmentasyonu gerçekleştir"""
        try:
            # Segmentasyon için özellikleri seç
            features_for_clustering = ['recency', 'frequency', 'monetary']
            X = self.data[features_for_clustering]
            
            # Verileri ölçeklendir
            X_scaled = self.scaler.fit_transform(X)
            
            # K-means modelini eğit
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
            self.data['segment'] = self.kmeans_model.fit_predict(X_scaled)
            
            self.logger.info(f"Müşteri segmentasyonu {n_clusters} küme ile tamamlandı")
            return self.data['segment']
        except Exception as e:
            self.logger.error(f"Segmentasyon hatası: {str(e)}")
            raise

    def preprocess_features(self, data, is_training=True):
        """Veri ön işleme ve özellik mühendisliği"""
        df = data.copy()
        
        # Kategorik değişkenleri dönüştür
        categorical_cols = ['segment', 'industry', 'company_size', 'location', 
                          'preferred_contact', 'payment_method']
        
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Boolean değişkenleri dönüştür
        boolean_cols = ['has_online_account', 'has_mobile_app', 'has_integration', 'is_active']
        for col in boolean_cols:
            df[col] = df[col].astype(int)
        
        return df

    def train_sales_prediction_model(self, data):
        """Satış tahmin modelini eğit"""
        try:
            # Veriyi hazırla
            df = self.preprocess_features(data, is_training=True)
            
            # Hedef değişken ve özellikler
            target = 'annual_revenue'
            features = [col for col in df.columns if col not in [target, 'customer_id', 'company_name', 
                                                               'customer_lifetime_value', 'avg_monthly_revenue',
                                                               'engagement_score', 'churn_risk']]
            
            X = df[features]
            y = df[target]
            
            # Eğitim ve test setlerini ayır
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Model parametreleri
            params = {
                'n_estimators': 200,
                'max_depth': 7,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
            # MLflow ile model eğitimini takip et
            with mlflow.start_run():
                self.sales_model = XGBRegressor(**params)
                self.sales_model.fit(X_train, y_train)
                
                # Model performansını değerlendir
                train_pred = self.sales_model.predict(X_train)
                test_pred = self.sales_model.predict(X_test)
                
                metrics = {
                    'train_r2': r2_score(y_train, train_pred),
                    'test_r2': r2_score(y_test, test_pred),
                    'train_mae': mean_absolute_error(y_train, train_pred),
                    'test_mae': mean_absolute_error(y_test, test_pred),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
                }
                
                # Metrikleri kaydet
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                # Modeli kaydet
                joblib.dump((self.sales_model, self.label_encoders), self.model_path)
                
                self.logger.info(f"Model eğitimi tamamlandı. Test R2 skoru: {metrics['test_r2']:.4f}")
                return metrics
                
        except Exception as e:
            self.logger.error(f"Model eğitim hatası: {str(e)}")
            raise

    def predict_future_sales(self, data, months_ahead=12):
        """Gelecek dönem satış tahmini yap"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError("Model dosyası bulunamadı. Önce modeli eğitmeniz gerekiyor.")
            
            # Modeli yükle
            if self.sales_model is None:
                self.sales_model, self.label_encoders = joblib.load(self.model_path)
            
            # Veriyi hazırla
            df = self.preprocess_features(data, is_training=False)
            
            # Tahmin için özellikleri seç
            features = [col for col in df.columns if col not in ['customer_id', 'company_name', 'annual_revenue',
                                                               'customer_lifetime_value', 'avg_monthly_revenue',
                                                               'engagement_score', 'churn_risk']]
            
            # Tahminleri yap
            current_revenue = self.sales_model.predict(df[features])
            
            # Aylık büyüme faktörlerini hesapla (segment bazında)
            growth_factors = {
                'Premium': 1.02,  # %2 aylık büyüme
                'Gold': 1.015,    # %1.5 aylık büyüme
                'Silver': 1.01,   # %1 aylık büyüme
                'Bronze': 1.005   # %0.5 aylık büyüme
            }
            
            # Gelecek tahminleri hesapla
            future_predictions = []
            for month in range(1, months_ahead + 1):
                month_predictions = []
                for i, row in df.iterrows():
                    segment = data.iloc[i]['segment']
                    base_prediction = current_revenue[i]
                    growth = growth_factors[segment] ** month
                    month_predictions.append(base_prediction * growth)
                future_predictions.append(month_predictions)
            
            # Sonuçları DataFrame'e dönüştür
            future_df = pd.DataFrame(future_predictions).T
            future_df.columns = [f"Month_{i+1}" for i in range(months_ahead)]
            future_df.index = data.index
            
            return future_df
            
        except Exception as e:
            self.logger.error(f"Tahmin hatası: {str(e)}")
            raise

    def analyze_customer_lifecycle(self, data):
        """Müşteri yaşam döngüsü analizi"""
        try:
            df = data.copy()
            
            # Yaşam döngüsü segmentleri
            df['lifecycle_stage'] = pd.cut(
                df['relationship_years'],
                bins=[0, 1, 3, 5, float('inf')],
                labels=['Yeni', 'Gelişen', 'Olgun', 'Sadık']
            )
            
            # Segment bazında metrikler
            lifecycle_metrics = df.groupby('lifecycle_stage').agg({
                'annual_revenue': 'mean',
                'satisfaction_score': 'mean',
                'order_frequency': 'mean',
                'customer_lifetime_value': 'mean',
                'engagement_score': 'mean',
                'customer_id': 'count'
            }).round(2)
            
            lifecycle_metrics.columns = ['Ortalama Gelir', 'Memnuniyet', 'Sipariş Sıklığı',
                                      'Müşteri Değeri', 'Etkileşim Skoru', 'Müşteri Sayısı']
            
            return lifecycle_metrics
            
        except Exception as e:
            self.logger.error(f"Yaşam döngüsü analizi hatası: {str(e)}")
            raise

    def generate_alerts(self, data):
        """Otomatik uyarı sistemi"""
        try:
            alerts = []
            
            # Yüksek riskli müşteriler
            high_risk = data[
                (data['churn_risk'] == 'Yüksek') & 
                (data['customer_lifetime_value'] > data['customer_lifetime_value'].median())
            ]
            if not high_risk.empty:
                alerts.append({
                    'type': 'Yüksek Riskli Değerli Müşteriler',
                    'severity': 'Kritik',
                    'count': len(high_risk),
                    'details': high_risk[['company_name', 'customer_lifetime_value', 'satisfaction_score']]
                })
            
            # Düşük memnuniyet
            low_satisfaction = data[
                (data['satisfaction_score'] < 6) & 
                (data['segment'].isin(['Premium', 'Gold']))
            ]
            if not low_satisfaction.empty:
                alerts.append({
                    'type': 'Düşük Memnuniyetli Premium Müşteriler',
                    'severity': 'Yüksek',
                    'count': len(low_satisfaction),
                    'details': low_satisfaction[['company_name', 'segment', 'satisfaction_score']]
                })
            
            # Azalan sipariş sıklığı
            low_engagement = data[
                (data['engagement_score'] < 0.4) &
                (data['is_active'] == True)
            ]
            if not low_engagement.empty:
                alerts.append({
                    'type': 'Düşük Etkileşimli Aktif Müşteriler',
                    'severity': 'Orta',
                    'count': len(low_engagement),
                    'details': low_engagement[['company_name', 'engagement_score', 'last_order_days']]
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Uyarı oluşturma hatası: {str(e)}")
            raise

    def create_visualizations(self):
        """Temel görselleştirmeleri oluştur"""
        try:
            # Segment bazında müşteri dağılımı
            fig1 = px.pie(self.data, names='segment', title='Müşteri Segmentleri Dağılımı')
            
            # Segment bazında ortalama harcama
            segment_stats = self.data.groupby('segment')['monetary'].mean().reset_index()
            fig2 = px.bar(segment_stats, x='segment', y='monetary',
                         title='Segment Bazında Ortalama Müşteri Harcaması')
            
            # Görselleştirmeleri kaydet
            fig1.write_html("reports/segment_distribution.html")
            fig2.write_html("reports/segment_monetary_analysis.html")
            
            self.logger.info("Görselleştirmeler oluşturuldu ve kaydedildi")
        except Exception as e:
            self.logger.error(f"Görselleştirme hatası: {str(e)}")
            raise

if __name__ == "__main__":
    analyzer = CustomerAnalytics()
    # Örnek kullanım
    analyzer.load_data("data/customer_data.csv")
    analyzer.preprocess_data()
    analyzer.perform_customer_segmentation()
    analyzer.train_sales_prediction_model(analyzer.data)
    analyzer.create_visualizations() 