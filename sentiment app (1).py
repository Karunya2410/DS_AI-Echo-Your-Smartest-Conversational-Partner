import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from io import BytesIO

st.set_page_config(page_title="ChatGPT Sentiment Analysis", layout="wide")

# Load model and tools
model = load_model("C:/Users/karunya/Documents/Guvi projects/NLP sentiment Analysis/sentiment_lstm_3class_model.h5")
tokenizer = joblib.load("C:/Users/karunya/Documents/Guvi projects/NLP sentiment Analysis/tokenizer.pkl")
label_encoder = joblib.load("C:/Users/karunya/Documents/Guvi projects/NLP sentiment Analysis/label_encoder.pkl")

# Load dataset
df = pd.read_csv("C:/Users/karunya/Documents/Guvi projects/NLP sentiment Analysis/New folder/chatgpt_reviews_with_sentiment (1).csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Decode encoded labels if needed
if df['sentiment'].dtype == 'object' and df['sentiment'].str.startswith("class_").any():
    df['sentiment'] = df['sentiment'].apply(lambda x: label_encoder.inverse_transform([int(x.split("_")[-1])])[0])

# Predict sentiment
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)
    class_idx = pred.argmax(axis=1)[0]
    return label_encoder.inverse_transform([class_idx])[0]

# Sidebar for input
st.sidebar.header("Sentiment Predictor")
user_input = st.sidebar.text_area("Enter a review:")
if user_input:
    result = predict_sentiment(user_input)
    st.sidebar.success(f"Predicted Sentiment: **{result}**")

st.title("ChatGPT Sentiment Analysis Dashboard")

#Overall sentiment
st.header("Overall Sentiment Distribution")
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
st.bar_chart(sentiment_counts)

# Sentiment by Rating
st.header("Sentiment by Rating")
if 'rating' in df.columns:
    rating_sentiment = pd.crosstab(df['rating'], df['sentiment'], normalize='index') * 100
    st.bar_chart(rating_sentiment)

#Word clouds by sentiment
st.header("Keywords by Sentiment")
for sentiment in df['sentiment'].unique():
    text = " ".join(df[df['sentiment'] == sentiment]['corrected_review'].dropna().astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.subheader(f"{sentiment} Reviews")
    st.image(wc.to_array(), use_column_width=True)

    img_bytes = BytesIO()
    wc.to_image().save(img_bytes, format='PNG')
    st.download_button(label=f"Download {sentiment} Word Cloud",
                       data=img_bytes.getvalue(),
                       file_name=f"{sentiment.lower()}_wordcloud.png",
                       mime="image/png")

# Sentiment over time
st.header("Sentiment Over Time")
df_time = df.dropna(subset=['date'])
trend = df_time.groupby([df_time['date'].dt.to_period('M'), 'sentiment']).size().unstack().fillna(0)
trend.index = trend.index.astype(str)
st.line_chart(trend)

# Verified purchases
st.header("Verified vs Unverified Sentiment")
if 'verified_purchase' in df.columns:
    verified = pd.crosstab(df['verified_purchase'], df['sentiment'], normalize='index') * 100
    st.bar_chart(verified)

#Review length
st.header("Review Length vs Sentiment")
df['review_length'] = df['corrected_review'].astype(str).apply(lambda x: len(x.split()))
fig, ax = plt.subplots()
sns.boxplot(data=df, x='sentiment', y='review_length', ax=ax)
ax.set_title("Review Length by Sentiment")
st.pyplot(fig)

#Sentiment by Location
st.header("Sentiment by Location")
if 'location' in df.columns:
    top_locs = df['location'].value_counts().nlargest(10).index
    loc_df = df[df['location'].isin(top_locs)]
    loc_sentiment = pd.crosstab(loc_df['location'], loc_df['sentiment'], normalize='index') * 100
    st.bar_chart(loc_sentiment)

#  Sentiment by Platform
st.header("Sentiment by Platform")
if 'platform' in df.columns:
    platform_sentiment = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
    st.bar_chart(platform_sentiment)

#  Sentiment by Version
st.header("Sentiment by Version")
if 'version' in df.columns:
    version_sentiment = pd.crosstab(df['version'], df['sentiment'], normalize='index') * 100
    st.bar_chart(version_sentiment)


