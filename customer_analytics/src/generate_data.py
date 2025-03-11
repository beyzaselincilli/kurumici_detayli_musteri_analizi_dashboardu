import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Sabit değerler ve dağılımlar
LOCATIONS = {
    'Istanbul': 0.30, 'Ankara': 0.15, 'Izmir': 0.12, 'Antalya': 0.08, 'Bursa': 0.07,
    'Adana': 0.05, 'Konya': 0.05, 'Gaziantep': 0.05, 'Mersin': 0.04, 'Kayseri': 0.03,
    'Eskişehir': 0.02, 'Trabzon': 0.02, 'Samsun': 0.02
}

CUSTOMER_SEGMENTS = {
    'Premium': 0.15,
    'Gold': 0.25,
    'Silver': 0.35,
    'Bronze': 0.25
}

INDUSTRY_SECTORS = {
    'Perakende': 0.25, 'Teknoloji': 0.15, 'Hizmet': 0.15, 'Üretim': 0.12,
    'Sağlık': 0.08, 'Eğitim': 0.07, 'Finans': 0.06, 'İnşaat': 0.05,
    'Turizm': 0.04, 'Lojistik': 0.03
}

COMPANY_SIZES = {
    'Büyük Ölçekli': 0.15,
    'Orta Ölçekli': 0.35,
    'Küçük Ölçekli': 0.50
}

CONTACT_CHANNELS = ['E-posta', 'Telefon', 'Web Site', 'Mobil Uygulama', 'Mağaza']
PAYMENT_METHODS = ['Kurumsal Kredi Kartı', 'Havale/EFT', 'Çek', 'Vadeli Ödeme']

def generate_company_name():
    prefixes = ['Türk', 'Anadolu', 'Star', 'Global', 'Tech', 'Net', 'Pro', 'Mega', 'Sistem', 'Grup']
    suffixes = ['A.Ş.', 'Ltd. Şti.', 'Holding', 'Teknoloji', 'Ticaret', 'Sanayi']
    return f"{random.choice(prefixes)} {random.choice(suffixes)}"

def generate_customer_data(n_customers=10000):
    np.random.seed(42)
    
    # Temel müşteri bilgileri
    customer_ids = range(1, n_customers + 1)
    company_names = [generate_company_name() for _ in range(n_customers)]
    
    # Lokasyon ve segment dağılımları
    locations = np.random.choice(list(LOCATIONS.keys()), n_customers, p=list(LOCATIONS.values()))
    segments = np.random.choice(list(CUSTOMER_SEGMENTS.keys()), n_customers, p=list(CUSTOMER_SEGMENTS.values()))
    industries = np.random.choice(list(INDUSTRY_SECTORS.keys()), n_customers, p=list(INDUSTRY_SECTORS.values()))
    company_sizes = np.random.choice(list(COMPANY_SIZES.keys()), n_customers, p=list(COMPANY_SIZES.values()))
    
    # Finansal metrikler
    base_revenue = np.random.lognormal(10, 1.5, n_customers)
    revenue_multipliers = {
        'Premium': 2.5, 'Gold': 1.8, 'Silver': 1.2, 'Bronze': 0.8
    }
    annual_revenue = np.array([base_revenue[i] * revenue_multipliers[segments[i]] for i in range(n_customers)])
    
    # Satış ve sipariş metrikleri
    order_frequency = np.random.negative_binomial(10, 0.3, n_customers) + 1
    avg_order_value = annual_revenue / order_frequency
    last_order_days = np.random.exponential(30, n_customers).astype(int) + 1
    
    # Müşteri etkileşim metrikleri
    satisfaction_scores = np.clip(np.random.normal(8.5, 1.5, n_customers), 0, 10)
    nps_scores = np.random.choice(range(0, 11), n_customers, p=[0.02, 0.02, 0.03, 0.03, 0.05, 0.1, 0.15, 0.2, 0.2, 0.1, 0.1])
    
    # Ek metrikler
    contract_values = np.random.lognormal(8, 1, n_customers) * 1000
    employee_counts = np.random.exponential(100, n_customers).astype(int) + 5
    relationship_years = np.random.exponential(5, n_customers).astype(int) + 1
    
    # Veri çerçevesi oluştur
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'company_name': company_names,
        'segment': segments,
        'industry': industries,
        'company_size': company_sizes,
        'location': locations,
        'annual_revenue': annual_revenue.astype(int),
        'order_frequency': order_frequency,
        'avg_order_value': avg_order_value.astype(int),
        'last_order_days': last_order_days,
        'total_orders': order_frequency,
        'contract_value': contract_values.astype(int),
        'employee_count': employee_counts,
        'relationship_years': relationship_years,
        'satisfaction_score': satisfaction_scores.round(2),
        'nps_score': nps_scores,
        'preferred_contact': np.random.choice(CONTACT_CHANNELS, n_customers),
        'payment_method': np.random.choice(PAYMENT_METHODS, n_customers),
        'has_online_account': np.random.choice([True, False], n_customers, p=[0.85, 0.15]),
        'has_mobile_app': np.random.choice([True, False], n_customers, p=[0.75, 0.25]),
        'has_integration': np.random.choice([True, False], n_customers, p=[0.60, 0.40]),
        'is_active': np.random.choice([True, False], n_customers, p=[0.90, 0.10])
    })
    
    # Hesaplanmış metrikler
    df['customer_lifetime_value'] = df['annual_revenue'] * (df['relationship_years'] * 0.8)
    df['avg_monthly_revenue'] = df['annual_revenue'] / 12
    df['engagement_score'] = (
        (df['satisfaction_score'] / 10 * 0.3) +
        (df['order_frequency'] / df['order_frequency'].max() * 0.3) +
        (df['relationship_years'] / df['relationship_years'].max() * 0.2) +
        (df['has_online_account'].astype(int) * 0.1) +
        (df['has_mobile_app'].astype(int) * 0.1)
    ).round(2)
    
    # Risk skorları
    df['churn_risk'] = np.where(
        (df['last_order_days'] > 60) | (df['satisfaction_score'] < 6) | (df['order_frequency'] < 5),
        'Yüksek',
        np.where(
            (df['last_order_days'] > 30) | (df['satisfaction_score'] < 7.5) | (df['order_frequency'] < 10),
            'Orta',
            'Düşük'
        )
    )
    
    return df

if __name__ == "__main__":
    # Veri setini oluştur
    print("Veri seti oluşturuluyor...")
    df = generate_customer_data(10000)
    
    # CSV olarak kaydet
    output_path = "../data/enterprise_customer_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nVeri seti oluşturuldu ve {output_path} konumuna kaydedildi.")
    print(f"Toplam kayıt sayısı: {len(df)}")
    
    print("\nVeri seti özeti:")
    print("\nSayısal değişkenler:")
    print(df.describe().round(2))
    
    print("\nKategorik değişken dağılımları:")
    categorical_cols = ['segment', 'industry', 'company_size', 'location', 'churn_risk']
    for col in categorical_cols:
        print(f"\n{col} dağılımı:")
        print(df[col].value_counts(normalize=True).round(3) * 100) 