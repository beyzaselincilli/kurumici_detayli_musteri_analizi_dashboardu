import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from customer_analysis import CustomerAnalytics
import mlflow
from datetime import datetime, timedelta

st.set_page_config(page_title="Müşteri Analitik Platformu", layout="wide")

def load_data():
    try:
        return pd.read_csv("../data/enterprise_customer_data.csv")
    except:
        st.error("Veri dosyası bulunamadı. Lütfen önce veri setini oluşturun.")
        return None

def main():
    st.title("Müşteri Analitik Platformu 📊")
    
    # Veriyi yükle
    data = load_data()
    if data is None:
        st.stop()
    
    # Analiz seçenekleri
    analysis_type = st.sidebar.selectbox(
        "Analiz Türü",
        ["Genel Bakış", "Satış Tahminleri", "Müşteri Yaşam Döngüsü", "Risk ve Uyarılar"]
    )
    
    if analysis_type == "Genel Bakış":
        show_overview(data)
    elif analysis_type == "Satış Tahminleri":
        show_sales_predictions(data)
    elif analysis_type == "Müşteri Yaşam Döngüsü":
        show_lifecycle_analysis(data)
    else:
        show_alerts(data)

def show_overview(data):
    # Filtreleme seçenekleri
    st.sidebar.header("Filtreleme Seçenekleri")
    
    selected_segments = st.sidebar.multiselect(
        "Müşteri Segmenti",
        options=sorted(data['segment'].unique()),
        default=sorted(data['segment'].unique())
    )
    
    selected_industries = st.sidebar.multiselect(
        "Sektör",
        options=sorted(data['industry'].unique()),
        default=sorted(data['industry'].unique())
    )
    
    selected_sizes = st.sidebar.multiselect(
        "Şirket Büyüklüğü",
        options=sorted(data['company_size'].unique()),
        default=sorted(data['company_size'].unique())
    )
    
    # Filtrelenmiş veri
    filtered_data = data[
        (data['segment'].isin(selected_segments)) &
        (data['industry'].isin(selected_industries)) &
        (data['company_size'].isin(selected_sizes))
    ]
    
    # Ana metrikler
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam Müşteri", f"{len(filtered_data):,}")
    with col2:
        st.metric("Toplam Gelir", f"₺{filtered_data['annual_revenue'].sum():,.0f}")
    with col3:
        st.metric("Ortalama Müşteri Değeri", f"₺{filtered_data['customer_lifetime_value'].mean():,.0f}")
    with col4:
        active_rate = (filtered_data['is_active'].sum() / len(filtered_data)) * 100
        st.metric("Aktif Müşteri Oranı", f"%{active_rate:.1f}")
    
    # Segment Analizi
    st.header("Segment Analizi")
    col1, col2 = st.columns(2)
    
    with col1:
        segment_revenue = filtered_data.groupby('segment')['annual_revenue'].sum().reset_index()
        fig1 = px.pie(segment_revenue, values='annual_revenue', names='segment',
                      title='Segmentlere Göre Gelir Dağılımı')
        st.plotly_chart(fig1)
    
    with col2:
        segment_stats = filtered_data.groupby('segment').agg({
            'customer_lifetime_value': 'mean',
            'satisfaction_score': 'mean'
        }).reset_index()
        
        fig2 = px.bar(segment_stats, x='segment', y=['customer_lifetime_value', 'satisfaction_score'],
                      title='Segment Bazında Müşteri Değeri ve Memnuniyet',
                      barmode='group')
        st.plotly_chart(fig2)

def show_sales_predictions(data):
    st.header("Satış Tahmin Analizi 📈")
    
    # Model eğitimi
    analyzer = CustomerAnalytics()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Satış Tahmin Modelini Eğit"):
            with st.spinner("Model eğitiliyor..."):
                metrics = analyzer.train_sales_prediction_model(data)
                st.success("Model eğitimi tamamlandı!")
                st.write("Model Performans Metrikleri:")
                metrics_df = pd.DataFrame({
                    'Metrik': ['R² (Test)', 'MAE (Test)', 'RMSE (Test)'],
                    'Değer': [
                        f"{metrics['test_r2']:.4f}",
                        f"₺{metrics['test_mae']:,.0f}",
                        f"₺{metrics['test_rmse']:,.0f}"
                    ]
                })
                st.table(metrics_df)
    
    with col2:
        months = st.slider("Tahmin Dönemi (Ay)", min_value=3, max_value=24, value=12)
    
    # Gelecek dönem tahminleri
    if st.button("Gelecek Dönem Tahminlerini Göster"):
        with st.spinner("Tahminler hesaplanıyor..."):
            predictions = analyzer.predict_future_sales(data, months_ahead=months)
            
            # Tahminleri görselleştir
            st.subheader("Gelecek Dönem Satış Tahminleri")
            
            # Segment bazında tahminler
            segment_predictions = pd.DataFrame()
            for segment in data['segment'].unique():
                segment_mask = data['segment'] == segment
                segment_predictions[segment] = predictions[segment_mask].mean()
            
            fig = go.Figure()
            for segment in segment_predictions.columns:
                fig.add_trace(go.Scatter(
                    x=segment_predictions.index,
                    y=segment_predictions[segment],
                    name=segment,
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title="Segment Bazında Gelecek Dönem Satış Tahminleri",
                xaxis_title="Ay",
                yaxis_title="Tahmini Gelir (₺)",
                hovermode='x unified'
            )
            st.plotly_chart(fig)
            
            # Toplam tahminler
            total_predictions = predictions.mean()
            current_revenue = data['annual_revenue'].sum() / 12  # Aylık ortalama gelir
            
            growth_rates = (total_predictions - current_revenue) / current_revenue * 100
            
            st.subheader("Büyüme Tahminleri")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "6 Aylık Büyüme Tahmini",
                    f"%{growth_rates['Month_6']:.1f}",
                    f"₺{total_predictions['Month_6']:,.0f}"
                )
            
            with col2:
                st.metric(
                    "12 Aylık Büyüme Tahmini",
                    f"%{growth_rates['Month_12']:.1f}",
                    f"₺{total_predictions['Month_12']:,.0f}"
                )

def show_lifecycle_analysis(data):
    st.header("Müşteri Yaşam Döngüsü Analizi 🔄")
    
    analyzer = CustomerAnalytics()
    lifecycle_metrics = analyzer.analyze_customer_lifecycle(data)
    
    # Yaşam döngüsü metrikleri
    st.subheader("Yaşam Döngüsü Segment Metrikleri")
    st.dataframe(lifecycle_metrics.style.format({
        'Ortalama Gelir': '₺{:,.0f}',
        'Müşteri Değeri': '₺{:,.0f}',
        'Memnuniyet': '{:.2f}',
        'Sipariş Sıklığı': '{:.1f}',
        'Etkileşim Skoru': '{:.2f}',
        'Müşteri Sayısı': '{:,.0f}'
    }))
    
    # Görselleştirmeler
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(
            lifecycle_metrics.reset_index(),
            x='lifecycle_stage',
            y=['Ortalama Gelir', 'Müşteri Değeri'],
            title='Yaşam Döngüsü Bazında Gelir ve Müşteri Değeri',
            barmode='group'
        )
        st.plotly_chart(fig1)
    
    with col2:
        fig2 = px.bar(
            lifecycle_metrics.reset_index(),
            x='lifecycle_stage',
            y=['Memnuniyet', 'Etkileşim Skoru'],
            title='Yaşam Döngüsü Bazında Memnuniyet ve Etkileşim',
            barmode='group'
        )
        st.plotly_chart(fig2)
    
    # Müşteri geçiş analizi
    st.subheader("Müşteri Yaşam Döngüsü Dağılımı")
    lifecycle_dist = data['relationship_years'].apply(
        lambda x: 'Yeni' if x <= 1 else ('Gelişen' if x <= 3 else ('Olgun' if x <= 5 else 'Sadık'))
    ).value_counts()
    
    fig3 = px.pie(
        values=lifecycle_dist.values,
        names=lifecycle_dist.index,
        title='Müşteri Yaşam Döngüsü Dağılımı'
    )
    st.plotly_chart(fig3)

def show_alerts(data):
    st.header("Risk ve Uyarı Sistemi ⚠️")
    
    analyzer = CustomerAnalytics()
    alerts = analyzer.generate_alerts(data)
    
    # Uyarıları göster
    for alert in alerts:
        severity_color = {
            'Kritik': 'red',
            'Yüksek': 'orange',
            'Orta': 'yellow'
        }
        
        st.warning(
            f"**{alert['type']}**  \n"
            f"Öncelik: {alert['severity']}  \n"
            f"Etkilenen Müşteri Sayısı: {alert['count']}"
        )
        
        with st.expander("Detayları Görüntüle"):
            st.dataframe(alert['details'])
    
    # Risk dağılımı
    st.subheader("Risk Dağılımı")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_dist = data['churn_risk'].value_counts()
        fig1 = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title='Churn Risk Dağılımı'
        )
        st.plotly_chart(fig1)
    
    with col2:
        risk_by_segment = pd.crosstab(data['segment'], data['churn_risk'])
        fig2 = px.bar(
            risk_by_segment.reset_index(),
            x='segment',
            y=['Düşük', 'Orta', 'Yüksek'],
            title='Segment Bazında Risk Dağılımı',
            barmode='stack'
        )
        st.plotly_chart(fig2)
    
    # Risk faktörleri
    st.subheader("Risk Faktörleri Analizi")
    risk_factors = pd.DataFrame({
        'Faktör': [
            'Düşük Memnuniyet (<6)',
            'Düşük Etkileşim (<0.4)',
            'Uzun Süredir Sipariş Yok (>60 gün)',
            'Azalan Sipariş Sıklığı'
        ],
        'Etkilenen Müşteri': [
            len(data[data['satisfaction_score'] < 6]),
            len(data[data['engagement_score'] < 0.4]),
            len(data[data['last_order_days'] > 60]),
            len(data[data['order_frequency'] < data['order_frequency'].median()])
        ]
    })
    
    fig3 = px.bar(
        risk_factors,
        x='Faktör',
        y='Etkilenen Müşteri',
        title='Risk Faktörlerine Göre Etkilenen Müşteri Sayısı'
    )
    st.plotly_chart(fig3)

if __name__ == "__main__":
    main() 