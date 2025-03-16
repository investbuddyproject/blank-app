import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from transformers import AutoTokenizer
import streamlit as st
import joblib
import pandas as pd
import requests
import datetime
from transformers import AutoTokenizer
from statsmodels.tsa.vector_ar.var_model import VAR
import nltk
from nltk.corpus import stopwords



# ------------------------- Read URL Query Parameters -------------------------

# Set page config as the first Streamlit command in your script.
st.set_page_config(page_title="Stock Sentiment & Price Forecast App", layout="centered")

# Read URL Query Parameters using the new API.
query_params = st.query_params
stock_symbol = query_params.get("stock", "AAPL")

# ------------------------- Tabs -------------------------
tab_sentiment, tab_forecast = st.tabs(["Sentiment Analysis", "Price Forecasting"])

# ------------------------- Textual Data Ingestion -------------------------
def get_reddit_texts(stock_symbol):
    results = []
    url = f"https://reddit-scraper2.p.rapidapi.com/search_posts_v3?query={stock_symbol}&sort=RELEVANCE&time=day&nsfw=0"
    headers = {
        "x-rapidapi-host": "reddit-scraper2.p.rapidapi.com",
        "x-rapidapi-key": "39408f7417msh190420cfe381944p16bd39jsndb389cd3e14e"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Reddit API error: {response.status_code} - {response.text}")
    data = response.json()
    if "data" in data:
        for post in data["data"]:
            content_text = ""
            if "content" in post and post["content"]:
                content_text = post["content"].get("text", "")
            if not content_text.strip():
                content_text = post.get("title", "")
            if content_text.strip():
                results.append(content_text)
    return results

def get_twitter_texts(query):
    results = []
    url = f"https://twitter-api45.p.rapidapi.com/search.php?query={query}&search_type=Top"
    headers = {
        "x-rapidapi-host": "twitter-api45.p.rapidapi.com",
        "x-rapidapi-key": "39408f7417msh190420cfe381944p16bd39jsndb389cd3e14e"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Twitter API error: {response.status_code} - {response.text}")
    data = response.json()
    if "timeline" in data:
        for tweet in data["timeline"]:
            text = tweet.get("text", "")
            if text.strip():
                results.append(text)
    return results

def get_news_texts(query):
    results = []
    url = "https://newsnow.p.rapidapi.com/newsv2"
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    payload = {
        "query": query,
        "time_bounded": True,
        "from_date": yesterday.strftime("%d/%m/%Y"),
        "to_date": today.strftime("%d/%m/%Y"),
        "location": "us",
        "language": "en",
        "page": 1
    }
    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-host": "newsnow.p.rapidapi.com",
        "x-rapidapi-key": "39408f7417msh190420cfe381944p16bd39jsndb389cd3e14e"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"News API error: {response.status_code} - {response.text}")
    data = response.json()
    if "news" in data:
        for article in data["news"]:
            text = article.get("text", "")
            if text.strip():
                results.append(text)
    return results

def get_combined_texts(reddit_symbol, twitter_query, news_query):
    combined = []
    try:
        reddit_texts = ""#get_reddit_texts(reddit_symbol)
        combined.extend(reddit_texts)
    except Exception as e:
        st.error(f"Error fetching Reddit data: {e}")
    try:
        twitter_texts = ""#get_twitter_texts(twitter_query)
        combined.extend(twitter_texts)
    except Exception as e:
        st.error(f"Error fetching Twitter data: {e}")
    try:
        news_texts = get_news_texts(news_query)
        combined.extend(news_texts)
    except Exception as e:
        st.error(f"Error fetching News data: {e}")
    return {"data": combined}

# ------------------------- Loading and caching models -------------------------
@st.cache_resource
def load_sentiment_model():
    return joblib.load("sentiment_classifier.pkl")

@st.cache_resource
def load_sentiment_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-cased")

sentiment_model = load_sentiment_model()
sentiment_tokenizer = load_sentiment_tokenizer()

# ------------------------- Sentiment Analysis Tab -------------------------
with tab_sentiment:
    st.header("Sentiment Analysis")
    st.write(f"Automatically fetching current news, Reddit, and Twitter data for **{stock_symbol.upper()}** to analyze its sentiment.")

    try:
        combined_data = get_combined_texts(stock_symbol, stock_symbol, stock_symbol)
        # If combined_data["data"] is a list, join the texts into one string.
        if isinstance(combined_data.get("data"), list):
            stock_text = " ".join(combined_data.get("data"))
            st.write(f"Data for **{stock_symbol.upper()}** has been fetched successfully.")
        else:
            stock_text = combined_data.get("data", "")
    except Exception as e:
        stock_text = f"Error fetching text data: {e}"

    # Ensure you download the stopwords if not already present.
    nltk.download('stopwords')

    # Define the set of English stop words.
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the text.
    filtered_text = " ".join([word for word in stock_text.split() if word.lower() not in stop_words])

    st.success(filtered_text)

    # --- Preprocess the filtered text using the tokenizer ---
    tokens = sentiment_tokenizer(
        filtered_text, 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors="np"
    )
    # Build a DataFrame with one row for the model input.
    input_df = pd.DataFrame({
        "input_ids": [tokens["input_ids"][0].tolist()],
        "attention_mask": [tokens["attention_mask"][0].tolist()]
    })

    # --- Compute Sentiment Automatically ---
    prediction = sentiment_model.predict(input_df)[0]
    st.success(f"Sentiment Prediction: {prediction}")

# ------------------------- Price Forecasting Tab -------------------------
with tab_forecast:
    st.header("Price Forecasting")
    st.write(f"Fetching price data and forecasting the next 240 hours for **{stock_symbol.upper()}** using a multivariate VAR approach.")
    
    st.info("Fetching stock data and computing forecast...")
    api_key = 'MGi_WdX9ktIi6maLsK_gcGaa7RrObmQf'
    aggregate = 'day'
    to_date = datetime.datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.datetime.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')

    url = (
        f'https://api.polygon.io/v2/aggs/ticker/{stock_symbol}/range/1/{aggregate}/'
        f'{from_date}/{to_date}?adjusted=true&sort=asc&limit=-1&apiKey={api_key}'
    )
    response = requests.get(url)
    data = response.json()
    results = data.get('results', [])
    df_stock = pd.DataFrame(results)

    if df_stock.empty:
        st.error("No data fetched. Please check the stock symbol and API key.")
    else:
        df_stock['Date'] = pd.to_datetime(df_stock['t'], unit='ms')
        df_stock.sort_values('Date', inplace=True)
        df_stock.set_index('Date', inplace=True)
        
        df_stock.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        
        df_stock = df_stock.asfreq('h', method='ffill')
        st.write("DataFrame shape:", df_stock.shape)

        last_data_date = df_stock.index[-1]
        time_diff = datetime.datetime.now() - last_data_date
        st.subheader("Last Fetched Data Point")
        st.write(f"- **Date:** {last_data_date.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"- **Close:** {df_stock.iloc[-1]['Close']:.2f}")
        st.write(f"- This data is from **{time_diff.days} days** and **{time_diff.seconds // 3600} hours** ago.")

        var_data = df_stock[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        from statsmodels.tsa.vector_ar.var_model import VAR
        model = VAR(var_data)
        lag_order_results = model.select_order(maxlags=15)
        selected_lag = lag_order_results.aic  # using AIC criterion (or choose another)
        st.write("Selected VAR lag order (AIC):", selected_lag)
        lag_order = selected_lag if selected_lag is not None and selected_lag > 0 else 1
        var_model = model.fit(lag_order)
        
        # Define forecast_steps here:
        forecast_steps = 240
        
        last_values = var_data.values[-lag_order:]
        forecast_array = var_model.forecast(y=last_values, steps=forecast_steps)
        forecast_columns = var_data.columns
        forecast_df = pd.DataFrame(forecast_array, columns=forecast_columns, 
                                   index=pd.date_range(start=last_data_date + pd.DateOffset(hours=1), periods=forecast_steps, freq='H'))
        
        one_month_ago = df_stock.index.max() - pd.DateOffset(months=6)
        df_recent = df_stock[df_stock.index >= one_month_ago].copy()

        # -------------------------------
        # Create Plotly Chart
        # -------------------------------
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_recent.index,
            y=df_recent['Close'],
            mode='lines',
            name='Historical'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Close'],
            mode='lines',
            name='Forecast (Next 240 hrs)'
        ))
        x_min = df_recent.index.min()
        x_max = forecast_df.index[-1]
        fig.update_layout(
            title=f"{stock_symbol.upper()} Price Forecast (Next 240 Hours)",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis=dict(range=[x_min, x_max])
        )
        st.plotly_chart(fig, use_container_width=True)
