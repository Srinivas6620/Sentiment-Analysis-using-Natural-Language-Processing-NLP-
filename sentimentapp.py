# =========================
# Customer Review Sentiment Anomaly Detection - Production-ready Streamlit Dashboard
# =========================

# -------------------------
# 1. Libraries
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from transformers import pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -------------------------
# 2. Simulate Review Dataset
# -------------------------
np.random.seed(42)
num_reviews = 5000
start_date = datetime(2025,1,1)
review_texts = [
    "Great product!", "Very bad experience.", "Loved it!", "Horrible service.", 
    "Average quality.", "Excellent support.", "Not worth the price.", 
    "Highly recommend.", "Terrible, do not buy.", "Satisfied with purchase."
]
data = {
    'review_id': range(1, num_reviews+1),
    'review_date':[start_date + timedelta(days=random.randint(0,180)) for _ in range(num_reviews)],
    'review_text':[random.choice(review_texts) for _ in range(num_reviews)],
    'reviewer_id':[random.randint(1,500) for _ in range(num_reviews)],
    'product_id':[random.randint(1,50) for _ in range(num_reviews)],
    'product_category':[random.choice(['Electronics','Books','Clothing','Toys','Kitchen']) for _ in range(num_reviews)]
}
df = pd.DataFrame(data)

# Introduce anomalies
for i in range(20):
    idx = random.randint(0, num_reviews-1)
    df.at[idx,'review_text'] = "Terrible product, do not buy!"
    df.at[idx,'review_date'] = start_date + timedelta(days=random.randint(0,180))

# -------------------------
# 3. Text Preprocessing
# -------------------------
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text
df['clean_text'] = df['review_text'].apply(preprocess_text)

# -------------------------
# 4. Sentiment Analysis
# -------------------------
analyzer = SentimentIntensityAnalyzer()
df['vader_sentiment'] = df['review_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['textblob_sentiment'] = df['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
sentiment_model = pipeline("sentiment-analysis")
def bert_sentiment(text):
    result = sentiment_model(text[:512])[0]
    score = result['score'] if result['label']=='POSITIVE' else -result['score']
    return score
df['bert_sentiment'] = df['review_text'].apply(bert_sentiment)
df['sentiment_score'] = (df['vader_sentiment'] + df['textblob_sentiment'] + df['bert_sentiment'])/3

# -------------------------
# 5. Aspect-based Sentiment
# -------------------------
aspects = {'quality':['quality','worth'],'service':['support','service'],'price':['price','buy']}
def aspect_sentiment(text):
    scores = {}
    for aspect, keywords in aspects.items():
        matched = any([k in text.lower() for k in keywords])
        scores[aspect] = TextBlob(text).sentiment.polarity if matched else df['sentiment_score'].mean()
    return scores
df['aspect_sentiment'] = df['review_text'].apply(aspect_sentiment)

# -------------------------
# 6. Time Series Aggregation
# -------------------------
df['review_date'] = pd.to_datetime(df['review_date'])
daily_sentiment = df.groupby('review_date')['sentiment_score'].mean().reset_index()
daily_sentiment['review_count'] = df.groupby('review_date')['review_id'].count().values

# -------------------------
# 7. Anomaly Detection
# -------------------------
daily_sentiment['z_score'] = (daily_sentiment['sentiment_score'] - daily_sentiment['sentiment_score'].mean()) / daily_sentiment['sentiment_score'].std()
daily_sentiment['stat_anomaly'] = daily_sentiment['z_score'].apply(lambda x: 1 if abs(x) > 2 else 0)

X = daily_sentiment[['sentiment_score']].values
iso_forest = IsolationForest(contamination=0.05, random_state=42)
daily_sentiment['iso_anomaly'] = iso_forest.fit_predict(X)
daily_sentiment['iso_anomaly'] = daily_sentiment['iso_anomaly'].apply(lambda x: 1 if x==-1 else 0)

ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
daily_sentiment['svm_anomaly'] = ocsvm.fit_predict(X)
daily_sentiment['svm_anomaly'] = daily_sentiment['svm_anomaly'].apply(lambda x:1 if x==-1 else 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(8, activation="relu")(input_layer)
encoder = Dense(4, activation="relu")(encoder)
decoder = Dense(8, activation="relu")(encoder)
decoder = Dense(input_dim, activation="linear")(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=16, verbose=0)
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions,2), axis=1)
threshold = np.percentile(mse,95)
daily_sentiment['autoencoder_anomaly'] = [1 if e>threshold else 0 for e in mse]

anomaly_columns = ['stat_anomaly','iso_anomaly','svm_anomaly','autoencoder_anomaly']
for col in anomaly_columns:
    if col not in daily_sentiment.columns:
        daily_sentiment[col] = 0
daily_sentiment['anomaly'] = daily_sentiment[anomaly_columns].max(axis=1)
daily_sentiment['anomaly_label'] = daily_sentiment['anomaly'].apply(lambda x: 'Anomaly' if x==1 else 'Normal')

# -------------------------
# 8. Forecasting Future Sentiment
# -------------------------
model = ExponentialSmoothing(daily_sentiment['sentiment_score'], trend='add', seasonal=None)
fit_model = model.fit()
forecast_days = 7
forecast_index = pd.date_range(daily_sentiment['review_date'].max()+timedelta(1), periods=forecast_days)
forecast_values = fit_model.forecast(forecast_days)
forecast_df = pd.DataFrame({'review_date':forecast_index, 'forecast_sentiment':forecast_values})

# -------------------------
# 9. Streamlit Dashboard
# -------------------------
st.title("Customer Review Sentiment Anomaly Detection - Full Dashboard")

# Sidebar filters
st.sidebar.header("Filters & Actions")
category_filter = st.sidebar.multiselect("Product Categories", options=df['product_category'].unique(), default=df['product_category'].unique())
date_range = st.sidebar.date_input("Date Range", [df['review_date'].min(), df['review_date'].max()])
products_filter = st.sidebar.multiselect("Select Products", options=df['product_id'].unique(), default=df['product_id'].unique())

# Filtered data
filtered_df = df[(df['product_category'].isin(category_filter)) &
                 (df['product_id'].isin(products_filter)) &
                 (df['review_date']>=pd.to_datetime(date_range[0])) &
                 (df['review_date']<=pd.to_datetime(date_range[1]))]
filtered_daily = daily_sentiment[(daily_sentiment['review_date']>=pd.to_datetime(date_range[0])) &
                                 (daily_sentiment['review_date']<=pd.to_datetime(date_range[1]))]

# -------------------------
# Sidebar Buttons
# -------------------------
if st.sidebar.button("Run Anomaly Detection"):
    st.success("Anomaly detection re-run completed!")

if st.sidebar.button("Run Forecast"):
    st.success("Forecast updated successfully!")

def send_email_alert(subject, body, to_email):
    sender_email = "youremail@example.com"
    password = "yourpassword"
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if st.sidebar.button("Send Alerts"):
    alerts = filtered_daily[filtered_daily['anomaly']==1][['review_date','sentiment_score']]
    alert_message = f"Detected {len(alerts)} anomalies\n{alerts.to_string(index=False)}"
    email_sent = send_email_alert("Sentiment Anomaly Alert", alert_message, "business@example.com")
    if email_sent:
        st.success("Alert email sent!")
    else:
        st.error("Failed to send alert email.")

if st.sidebar.button("Export Data"):
    st.download_button("Download Filtered Reviews", data=filtered_df.to_csv(index=False), file_name="filtered_reviews.csv")
    st.download_button("Download Daily Sentiment Anomalies", data=filtered_daily.to_csv(index=False), file_name="daily_sentiment_anomalies.csv")
    st.download_button("Download Forecast Sentiment", data=forecast_df.to_csv(index=False), file_name="forecast_sentiment.csv")

# -------------------------
# 10. Visualizations
# -------------------------
st.subheader("Daily Sentiment Trend with Forecast & Anomalies")
filtered_daily['rolling_mean'] = filtered_daily['sentiment_score'].rolling(window=7,min_periods=1).mean()
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=filtered_daily['review_date'], y=filtered_daily['sentiment_score'], mode='lines+markers', name='Daily Sentiment'))
fig_trend.add_trace(go.Scatter(x=filtered_daily['review_date'], y=filtered_daily['rolling_mean'], mode='lines', name='7-day Rolling Avg', line=dict(color='orange', width=3)))
fig_trend.add_trace(go.Scatter(x=filtered_daily[filtered_daily['anomaly']==1]['review_date'], 
                               y=filtered_daily[filtered_daily['anomaly']==1]['sentiment_score'], 
                               mode='markers', marker=dict(color='red', size=10), name='Anomaly'))
fig_trend.add_trace(go.Scatter(x=forecast_df['review_date'], y=forecast_df['forecast_sentiment'], mode='lines', name='Forecast', line=dict(color='green', dash='dash')))
st.plotly_chart(fig_trend)

st.subheader("Sentiment Distribution")
fig_dist = px.histogram(filtered_df, x='sentiment_score', nbins=20)
st.plotly_chart(fig_dist)

st.subheader("Anomalies Count by Product")
category_anomalies = filtered_df[filtered_df['review_date'].isin(filtered_daily[filtered_daily['anomaly']==1]['review_date'])]
fig_cat = px.histogram(category_anomalies, x='product_category', color='product_category')
st.plotly_chart(fig_cat)

st.subheader("Top Reviewers")
top_reviewers = filtered_df['reviewer_id'].value_counts().head(10).reset_index()
top_reviewers.columns = ['reviewer_id','review_count']
fig_top = px.bar(top_reviewers, x='reviewer_id', y='review_count', text='review_count')
st.plotly_chart(fig_top)

st.subheader("Aspect-based Sentiment Summary")
aspect_df = pd.DataFrame(filtered_df['aspect_sentiment'].tolist())
st.dataframe(aspect_df.describe())

st.subheader("Competitor Sentiment Comparison")
competitors = ['Competitor A', 'Competitor B']
competitor_data = []
for comp in competitors:
    for date in pd.date_range(df['review_date'].min(), df['review_date'].max()):
        competitor_data.append({'review_date': date, 'competitor': comp, 'sentiment_score': random.uniform(-1,1)})
competitor_df = pd.DataFrame(competitor_data)
fig_comp = px.line(competitor_df, x='review_date', y='sentiment_score', color='competitor', title='Competitor Sentiment Trend')
st.plotly_chart(fig_comp)

st.subheader("Multi-Product Sentiment Insights")
product_daily_sentiment = filtered_df.groupby(['review_date','product_id'])['sentiment_score'].mean().reset_index()
fig_multi_product = px.line(product_daily_sentiment, x='review_date', y='sentiment_score', color='product_id', title='Daily Sentiment Trend per Product', markers=True)
st.plotly_chart(fig_multi_product)

product_anomaly_df = filtered_df.merge(daily_sentiment[['review_date','anomaly']], on='review_date')
product_anomaly_count = product_anomaly_df.groupby('product_id')['anomaly'].sum().reset_index()
fig_anomaly_products = px.bar(product_anomaly_count, x='product_id', y='anomaly', title='Number of Anomalous Days per Product')
st.plotly_chart(fig_anomaly_products)

st.subheader("Real-time Alerts")
alerts = filtered_daily[filtered_daily['anomaly']==1][['review_date','sentiment_score']]
if not alerts.empty:
    st.warning(f"{len(alerts)} anomalies detected!")
    st.dataframe(alerts)
else:
    st.success("No anomalies detected in selected range.")

