import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import numpy as np
from itertools import product
import warnings

warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(
    page_title="Müşteri Segmentasyonu ve Analizi",
    page_icon="📊",
    layout="wide"
)


# SARIMAX model optimization function
def optimize_sarimax(data, p_range, d_range, q_range, P_range, D_range, Q_range, s=12):
    best_score = float('inf')
    best_params = None

    pdq = list(product(p_range, d_range, q_range))
    seasonal_pdq = list(product(P_range, D_range, Q_range, [s]))

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(data,
                                order=param,
                                seasonal_order=param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit(disp=False)
                aic = results.aic
                if aic < best_score:
                    best_score = aic
                    best_params = (param, param_seasonal)
            except:
                continue
    return best_params


# Veri yükleme fonksiyonu
@st.cache_data
def load_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "marketing_campaign.csv")
        df = pd.read_csv(csv_path, sep="\t")

        # Eksik değerleri kontrol et
        st.write("Eksik değer kontrolü:")
        st.write(df.isnull().sum())

        # Eksik değerleri temizle
        df = df.dropna()
        st.write(f"Temizlenmiş veri seti boyutu: {df.shape}")

        # Veri ön işleme
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format='%d-%m-%Y')
        df["Age"] = 2025 - df["Year_Birth"]
        df = df[df["Age"] < 100]

        # Eğitim kategorileri
        df["Education"] = df["Education"].replace({
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate"
        })

        # Medeni durum
        df["Living_With"] = df["Marital_Status"].replace({
            "Married": "Partner",
            "Together": "Partner",
            "Absurd": "Alone",
            "Widow": "Alone",
            "YOLO": "Alone",
            "Divorced": "Alone",
            "Single": "Alone"
        })

        # Toplam harcama
        df['TotalSpending'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df[
            'MntSweetProducts'] + df['MntGoldProds']

        # Çocuk sayısı
        df["Children"] = df["Kidhome"] + df["Teenhome"]

        # Hane halkı büyüklüğü
        df['HouseholdSize'] = df["Living_With"].replace({"Alone": 1, "Partner": 2}) + df['Children']

        # Gereksiz sütunları sil
        df = df.drop(["Z_CostContact", "Z_Revenue", "Marital_Status", "Year_Birth"], axis=1)

        # Toplam kampanya kabulü
        df['TotalAcceptedCmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df[
            'AcceptedCmp5']

        # Toplam satın alma sayısı
        df['NumTotalPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df[
            'NumDealsPurchases']

        # RFM Analizi
        rfm = pd.DataFrame()
        rfm["recency"] = df["Recency"]
        rfm["frequency"] = df["NumTotalPurchases"]
        rfm["monetary"] = df["TotalSpending"]

        # RFM skorları
        r_labels = range(4, 0, -1)
        f_labels = range(1, 5)
        m_labels = range(1, 5)

        rfm['R'] = pd.qcut(rfm['recency'], q=4, labels=r_labels)
        rfm['F'] = pd.qcut(rfm['frequency'], q=4, labels=f_labels)
        rfm['M'] = pd.qcut(rfm['monetary'], q=4, labels=m_labels)

        rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

        # Müşteri segmentleri
        seg_map = {
            r'[1-2][1-2]': 'hibernating',
            r'[1-2][3-4]': 'at_risk',
            r'[1-2]5': 'cant_loose',
            r'3[1-2]': 'about_to_sleep',
            r'33': 'need_attention',
            r'[3-4][4-5]': 'loyal_customers',
            r'41': 'promising',
            r'51': 'new_customers',
            r'[4-5][2-3]': 'potential_loyalists',
            r'5[4-5]': 'champions'
        }

        rfm['segment'] = rfm['R'].astype(str) + rfm['F'].astype(str)
        rfm['segment'] = rfm['segment'].replace(seg_map, regex=True)
        df['segment'] = rfm['segment']

        # Kümeleme analizi
        X = df[['Age', 'Income', 'TotalSpending', 'NumTotalPurchases']]

        # Eksik değer kontrolü
        if X.isnull().any().any():
            st.warning("Kümeleme için kullanılacak özelliklerde hala eksik değerler var!")
            st.write("Eksik değerlerin dağılımı:")
            st.write(X.isnull().sum())
            return None, None, None, None, None, None, None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=5, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        # 6 Aylık Tahmin Analizi
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
        df['Month'] = df['Dt_Customer'].dt.to_period('M')
        monthly_data = df.groupby('Month').agg({
            'TotalSpending': 'sum',
            'NumTotalPurchases': 'sum',
            'TotalAcceptedCmp': 'sum'
        }).reset_index()

        monthly_data['Month'] = monthly_data['Month'].astype(str)
        monthly_data['Month'] = pd.to_datetime(monthly_data['Month'])
        monthly_data.set_index('Month', inplace=True)

        # SARIMAX parameter ranges for optimization
        p_range = range(0, 3)
        d_range = range(0, 2)
        q_range = range(0, 3)
        P_range = range(0, 2)
        D_range = range(0, 2)
        Q_range = range(0, 2)

        # Optimize SARIMAX models for each metric
        spending_params = optimize_sarimax(monthly_data['TotalSpending'],
                                           p_range, d_range, q_range,
                                           P_range, D_range, Q_range)
        purchases_params = optimize_sarimax(monthly_data['NumTotalPurchases'],
                                            p_range, d_range, q_range,
                                            P_range, D_range, Q_range)
        campaigns_params = optimize_sarimax(monthly_data['TotalAcceptedCmp'],
                                            p_range, d_range, q_range,
                                            P_range, D_range, Q_range)

        # Fit SARIMAX models with optimized parameters
        spending_model = SARIMAX(monthly_data['TotalSpending'],
                                 order=spending_params[0],
                                 seasonal_order=spending_params[1]).fit(disp=False)
        purchases_model = SARIMAX(monthly_data['NumTotalPurchases'],
                                  order=purchases_params[0],
                                  seasonal_order=purchases_params[1]).fit(disp=False)
        campaigns_model = SARIMAX(monthly_data['TotalAcceptedCmp'],
                                  order=campaigns_params[0],
                                  seasonal_order=campaigns_params[1]).fit(disp=False)

        # Generate forecasts
        forecast_spending = spending_model.get_forecast(steps=6)
        forecast_purchases = purchases_model.get_forecast(steps=6)
        forecast_campaigns = campaigns_model.get_forecast(steps=6)

        # Create forecast DataFrames with confidence intervals
        last_date = monthly_data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=7, freq='M')[1:]

        forecast_spending = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_spending.predicted_mean,
            'yhat_lower': forecast_spending.conf_int()['lower TotalSpending'],
            'yhat_upper': forecast_spending.conf_int()['upper TotalSpending']
        })

        forecast_purchases = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_purchases.predicted_mean,
            'yhat_lower': forecast_purchases.conf_int()['lower NumTotalPurchases'],
            'yhat_upper': forecast_purchases.conf_int()['upper NumTotalPurchases']
        })

        forecast_campaigns = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_campaigns.predicted_mean,
            'yhat_lower': forecast_campaigns.conf_int()['lower TotalAcceptedCmp'],
            'yhat_upper': forecast_campaigns.conf_int()['upper TotalAcceptedCmp']
        })

        # Öneri sistemi için veri hazırlama
        # Ürün kategorileri ve harcamaları
        product_columns = ['MntWines', 'MntFruits', 'MntMeatProducts',
                           'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        product_names = ['Şarap', 'Meyve', 'Et Ürünleri',
                         'Balık Ürünleri', 'Tatlılar', 'Altın Ürünler']

        # Kullanıcı-ürün matrisi oluşturma
        user_product_matrix = df[product_columns].values

        # KNN modeli oluşturma
        knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn_model.fit(user_product_matrix)

        return df, forecast_spending, forecast_purchases, forecast_campaigns, user_product_matrix, product_names, knn_model, monthly_data
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
        return None, None, None, None, None, None, None, None


# Veriyi yükle
df, forecast_spending, forecast_purchases, forecast_campaigns, user_product_matrix, product_names, knn_model, monthly_data = load_data()

if df is not None:
    # Başlık
    st.title("Müşteri Segmentasyonu ve Analizi")

    # Kenar çubuğu filtreleri
    st.sidebar.header("Filtreler")

    # Segment seçimi
    segments = df['segment'].unique()
    selected_segments = st.sidebar.multiselect(
        "Müşteri Segmentleri",
        options=segments,
        default=segments
    )

    # Yaş aralığı
    min_age, max_age = st.sidebar.slider(
        "Yaş Aralığı",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )

    # Gelir aralığı
    min_income, max_income = st.sidebar.slider(
        "Gelir Aralığı",
        min_value=int(df['Income'].min()),
        max_value=int(df['Income'].max()),
        value=(int(df['Income'].min()), int(df['Income'].max()))
    )

    # Filtreleme
    filtered_df = df[
        (df['segment'].isin(selected_segments)) &
        (df['Age'] >= min_age) &
        (df['Age'] <= max_age) &
        (df['Income'] >= min_income) &
        (df['Income'] <= max_income)
        ]

    # Metrikler
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam Müşteri", len(filtered_df))
    with col2:
        st.metric("Ortalama Yaş", round(filtered_df['Age'].mean(), 1))
    with col3:
        st.metric("Ortalama Gelir", f"${round(filtered_df['Income'].mean(), 2):,}")
    with col4:
        st.metric("Ortalama Harcama", f"${round(filtered_df['TotalSpending'].mean(), 2):,}")

    # Sekmeler
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Genel Bakış",
        "Müşteri Segmentasyonu",
        "Harcama Analizi",
        "Kampanya Performansı",
        "6 Aylık Tahmin",
        "Öneri Sistemi ve Pazarlama Analizi"
    ])

    with tab1:
        st.header("Genel Bakış")

        # Yaş ve gelir dağılımı
        fig1 = px.scatter(
            filtered_df,
            x="Age",
            y="Income",
            color="segment",
            title="Yaş ve Gelir Dağılımı",
            labels={"Age": "Yaş", "Income": "Gelir", "segment": "Segment"}
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Segment dağılımı
        fig2 = px.pie(
            filtered_df,
            names="segment",
            title="Segment Dağılımı"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.header("Müşteri Segmentasyonu")

        # Segment analizi
        segment_stats = filtered_df.groupby("segment").agg({
            'Age': 'mean',
            'Income': 'mean',
            'TotalSpending': 'mean',
            'NumTotalPurchases': 'mean',
            'TotalAcceptedCmp': 'mean'
        }).round(2)

        st.dataframe(segment_stats)

        # Kümeleme görselleştirmesi
        fig3 = px.scatter(
            filtered_df,
            x="TotalSpending",
            y="Income",
            color="cluster",
            title="Müşteri Kümeleri",
            labels={"TotalSpending": "Toplam Harcama", "Income": "Gelir", "cluster": "Küme"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.header("Harcama Analizi")

        # Harcama dağılımı
        fig4 = px.box(
            filtered_df,
            y="TotalSpending",
            x="segment",
            title="Segmentlere Göre Harcama Dağılımı",
            labels={"TotalSpending": "Toplam Harcama", "segment": "Segment"}
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Web vs fiziksel alışveriş
        web_vs_store = filtered_df.groupby("segment").agg({
            'NumWebPurchases': 'mean',
            'NumStorePurchases': 'mean'
        }).round(2)

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            x=web_vs_store.index,
            y=web_vs_store['NumWebPurchases'],
            name='Web Alışveriş'
        ))
        fig5.add_trace(go.Bar(
            x=web_vs_store.index,
            y=web_vs_store['NumStorePurchases'],
            name='Fiziksel Alışveriş'
        ))
        fig5.update_layout(
            title="Web vs Fiziksel Alışveriş",
            xaxis_title="Segment",
            yaxis_title="Ortalama Satın Alma Sayısı",
            barmode='group'
        )
        st.plotly_chart(fig5, use_container_width=True)

    with tab4:
        st.header("Kampanya Performansı")

        # Kampanya yanıt oranları
        campaign_response = filtered_df.groupby("segment")["TotalAcceptedCmp"].mean().sort_values(ascending=False)

        fig6 = px.bar(
            x=campaign_response.index,
            y=campaign_response.values,
            title="Segmentlere Göre Kampanya Yanıt Oranları",
            labels={"x": "Segment", "y": "Ortalama Kampanya Yanıtı"}
        )
        st.plotly_chart(fig6, use_container_width=True)

        # ROI önerileri
        st.subheader("ROI Artırma Önerileri")
        roi_recommendations = {
            'champions': '1. Yüksek değerli ürünler ve özel indirimler sunun',
            'loyal_customers': '2. Sadakat programı ve özel etkinlikler düzenleyin',
            'potential_loyalists': '3. Kişiselleştirilmiş öneriler ve hediye kartları sunun',
            'new_customers': '4. Hoş geldin kampanyaları ve indirimler sunun',
            'promising': '5. Hedefli kampanyalar ve özel teklifler sunun',
            'need_attention': '6. Kişiselleştirilmiş iletişim ve özel teklifler sunun',
            'about_to_sleep': '7. Yeniden etkinleştirme kampanyaları düzenleyin',
            'at_risk': '8. Özel indirimler ve sadakat programları sunun',
            'cant_loose': '9. Acil eylem planı ve özel teklifler sunun',
            'hibernating': '10. Yeniden etkinleştirme kampanyaları ve özel teklifler sunun'
        }

        for segment, recommendation in roi_recommendations.items():
            if segment in selected_segments:
                st.info(f"**{segment}**: {recommendation}")

    with tab5:
        st.header("6 Aylık Tahmin Analizi")

        # Harcama tahmini
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data['TotalSpending'],
            name='Gerçek',
            line=dict(color='blue')
        ))
        fig7.add_trace(go.Scatter(
            x=forecast_spending['ds'],
            y=forecast_spending['yhat'],
            name='Tahmin',
            line=dict(color='red')
        ))
        fig7.add_trace(go.Scatter(
            x=forecast_spending['ds'],
            y=forecast_spending['yhat_upper'],
            name='Üst Sınır',
            line=dict(color='rgba(255,0,0,0.2)')
        ))
        fig7.add_trace(go.Scatter(
            x=forecast_spending['ds'],
            y=forecast_spending['yhat_lower'],
            name='Alt Sınır',
            line=dict(color='rgba(255,0,0,0.2)'),
            fill='tonexty'
        ))
        fig7.update_layout(
            title='6 Aylık Toplam Harcama Tahmini',
            xaxis_title='Tarih',
            yaxis_title='Toplam Harcama'
        )
        st.plotly_chart(fig7, use_container_width=True)

        # Satın alma tahmini
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data['NumTotalPurchases'],
            name='Gerçek',
            line=dict(color='blue')
        ))
        fig8.add_trace(go.Scatter(
            x=forecast_purchases['ds'],
            y=forecast_purchases['yhat'],
            name='Tahmin',
            line=dict(color='red')
        ))
        fig8.add_trace(go.Scatter(
            x=forecast_purchases['ds'],
            y=forecast_purchases['yhat_upper'],
            name='Üst Sınır',
            line=dict(color='rgba(255,0,0,0.2)')
        ))
        fig8.add_trace(go.Scatter(
            x=forecast_purchases['ds'],
            y=forecast_purchases['yhat_lower'],
            name='Alt Sınır',
            line=dict(color='rgba(255,0,0,0.2)'),
            fill='tonexty'
        ))
        fig8.update_layout(
            title='6 Aylık Toplam Satın Alma Tahmini',
            xaxis_title='Tarih',
            yaxis_title='Toplam Satın Alma'
        )
        st.plotly_chart(fig8, use_container_width=True)

        # Kampanya kabul tahmini
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data['TotalAcceptedCmp'],
            name='Gerçek',
            line=dict(color='blue')
        ))
        fig9.add_trace(go.Scatter(
            x=forecast_campaigns['ds'],
            y=forecast_campaigns['yhat'],
            name='Tahmin',
            line=dict(color='red')
        ))
        fig9.add_trace(go.Scatter(
            x=forecast_campaigns['ds'],
            y=forecast_campaigns['yhat_upper'],
            name='Üst Sınır',
            line=dict(color='rgba(255,0,0,0.2)')
        ))
        fig9.add_trace(go.Scatter(
            x=forecast_campaigns['ds'],
            y=forecast_campaigns['yhat_lower'],
            name='Alt Sınır',
            line=dict(color='rgba(255,0,0,0.2)'),
            fill='tonexty'
        ))
        fig9.update_layout(
            title='6 Aylık Kampanya Kabul Tahmini',
            xaxis_title='Tarih',
            yaxis_title='Toplam Kampanya Kabul'
        )
        st.plotly_chart(fig9, use_container_width=True)

        # Tahmin sonuçlarını tablo olarak göster
        st.subheader("Tahmin Sonuçları")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Toplam Harcama Tahmini")
            st.dataframe(forecast_spending[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

        with col2:
            st.write("Toplam Satın Alma Tahmini")
            st.dataframe(forecast_purchases[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

        with col3:
            st.write("Kampanya Kabul Tahmini")
            st.dataframe(forecast_campaigns[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

    with tab6:
        st.header("Öneri Sistemi ve Pazarlama Analizi")

        # Öneri sistemi
        st.subheader("Kişiselleştirilmiş Ürün Önerileri")

        # Yaş aralığı seçimi
        age_ranges = {
            "18-25": (18, 25),
            "26-35": (26, 35),
            "36-45": (36, 45),
            "46-55": (46, 55),
            "56-65": (56, 65),
            "65+": (66, 100)
        }

        selected_age_range = st.selectbox(
            "Yaş Aralığı Seçin",
            options=list(age_ranges.keys())
        )

        # Seçilen yaş aralığına göre müşterileri filtrele
        min_age, max_age = age_ranges[selected_age_range]
        filtered_customers = df[
            (df['Age'] >= min_age) &
            (df['Age'] <= max_age)
            ]

        if len(filtered_customers) > 0:
            # Müşteri seçimi
            customer_id = st.selectbox(
                "Müşteri Seçin",
                options=filtered_customers.index,
                format_func=lambda x: f"Müşteri {x} - Yaş: {filtered_customers.loc[x, 'Age']}"
            )

            # Benzer müşterileri bul
            customer_data = user_product_matrix[customer_id].reshape(1, -1)
            distances, indices = knn_model.kneighbors(customer_data)
            similar_customers = indices[0][1:]  # İlk müşteri kendisi olduğu için atlıyoruz

            # Benzer müşterilerin satın aldığı ürünleri bul
            similar_products = user_product_matrix[similar_customers].mean(axis=0)
            top_products = np.argsort(-similar_products)

            # Önerileri göster
            st.write("Önerilen Ürünler:")
            for i, product_idx in enumerate(top_products[:3]):
                st.write(f"{i + 1}. {product_names[product_idx]}")

            # Müşteri bilgilerini göster
            st.subheader("Müşteri Bilgileri")
            customer_info = filtered_customers.loc[customer_id]
            st.write(f"Yaş: {customer_info['Age']}")
            st.write(f"Gelir: {customer_info['Income']}")
            st.write(f"Eğitim: {customer_info['Education']}")
            st.write(f"Medeni Durum: {customer_info['Living_With']}")
            st.write(f"Toplam Harcama: {customer_info['TotalSpending']}")
        else:
            st.warning("Seçilen yaş aralığında müşteri bulunamadı.")

        # Pazarlama etkisi analizi
        st.subheader("Pazarlama Etkisi Analizi")

        # Kampanya kabul oranları
        campaign_acceptance = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                                  'AcceptedCmp4', 'AcceptedCmp5']].mean()

        fig10 = px.bar(
            x=['Kampanya 1', 'Kampanya 2', 'Kampanya 3', 'Kampanya 4', 'Kampanya 5'],
            y=campaign_acceptance.values,
            title='Kampanya Kabul Oranları',
            labels={'x': 'Kampanya', 'y': 'Kabul Oranı'}
        )
        st.plotly_chart(fig10, use_container_width=True)

        # Kampanya etkisi analizi
        st.subheader("Kampanya Etkisi Analizi")

        # Her kampanya için ortalama harcama
        campaign_spending = []
        for i in range(1, 6):
            campaign_col = f'AcceptedCmp{i}'
            avg_spending = df[df[campaign_col] == 1]['TotalSpending'].mean()
            campaign_spending.append(avg_spending)

        fig11 = px.bar(
            x=['Kampanya 1', 'Kampanya 2', 'Kampanya 3', 'Kampanya 4', 'Kampanya 5'],
            y=campaign_spending,
            title='Kampanyalara Göre Ortalama Harcama',
            labels={'x': 'Kampanya', 'y': 'Ortalama Harcama'}
        )
        st.plotly_chart(fig11, use_container_width=True)

        # ROI analizi
        st.subheader("ROI Analizi")

        # Her kampanya için ROI hesaplama
        roi_data = []
        for i in range(1, 6):
            campaign_col = f'AcceptedCmp{i}'
            total_spending = df[df[campaign_col] == 1]['TotalSpending'].sum()
            num_accepted = df[campaign_col].sum()
            roi = total_spending / num_accepted if num_accepted > 0 else 0
            roi_data.append(roi)

        fig12 = px.bar(
            x=['Kampanya 1', 'Kampanya 2', 'Kampanya 3', 'Kampanya 4', 'Kampanya 5'],
            y=roi_data,
            title='Kampanyalara Göre ROI',
            labels={'x': 'Kampanya', 'y': 'ROI'}
        )
        st.plotly_chart(fig12, use_container_width=True)

        # Öneriler
        st.subheader("Pazarlama Önerileri")

        # En başarılı kampanya
        best_campaign = np.argmax(roi_data) + 1
        st.write(f"En yüksek ROI'ye sahip kampanya: Kampanya {best_campaign}")

        # Segment bazlı öneriler
        segment_roi = df.groupby('segment').agg({
            'TotalSpending': 'mean',
            'TotalAcceptedCmp': 'mean'
        })
        segment_roi['ROI'] = segment_roi['TotalSpending'] / segment_roi['TotalAcceptedCmp']

        st.write("Segment Bazlı ROI:")
        st.dataframe(segment_roi)
else:
    st.error("Veri yüklenemedi. Lütfen veri dosyasının doğru konumda olduğundan emin olun.")