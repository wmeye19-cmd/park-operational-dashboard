import streamlit as st
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Set up beautiful wide page layout
st.set_page_config(layout="wide", page_title="NPS Operational Analytics")

st.title("🌲 National Park Service — Live AI Transformer Dashboard")
st.markdown("*Running real-time multilingual transformer inference layer*")

# ── 1. GLOBAL TRANSLATION MAP ───────────────────────────────────────────────
TOPIC_TRANSLATION_MAP = {
    -1: "General Feedback / Noise",
    0: "Campground Noise & Disturbance",
    1: "Trail Overgrowth & Hazards",
    6: "Restroom Maintenance & Sanitation",
    12: "Road Conditions & Potholes"
}

# ── 2. LOAD THE LIGHTWEIGHT MODEL LIVE ──────────────────────────────────────
@st.cache_resource # Keeps the AI model loaded in memory permanently
def load_production_ai_model():
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    loaded_model = BERTopic.load("unseen_web_data/park_complaints_web_lightweight", embedding_model=embedding_model)
    return loaded_model

with st.spinner("🧠 Loading Multilingual Transformer Model into Web RAM..."):
    topic_model = load_production_ai_model()

# ── 3. LOAD & FORMAT RAW DATA ───────────────────────────────────────────────
@st.cache_data 
def load_data():
    df = pd.read_parquet("unseen_web_data/holdout_reviews.parquet")
    df.columns = df.columns.str.lower()
    
    # Safely convert the unix_time into a normal Datetime column
    if 'unix_time' in df.columns:
        # Most web scrapers use milliseconds. If the year turns out to be 1970, we swap to seconds.
        df['date'] = pd.to_datetime(df['unix_time'], unit='ms')
        if df['date'].dt.year.mean() == 1970:
            df['date'] = pd.to_datetime(df['unix_time'], unit='s')
    else:
        st.error("🚨 Critical Error: Could not find 'unix_time' in the dataset.")
        st.stop()
        
    return df

df = load_data()

# ── 4. SIDEBAR CONTROLS ─────────────────────────────────────────────────────
st.sidebar.header("🕹️ Control Dashboard")

all_parks = sorted(df['park_name'].dropna().unique().tolist()) if 'park_name' in df.columns else ["Yosemite"]
selected_parks = st.sidebar.multiselect("📍 Filter by Park(s)", options=all_parks, default=all_parks[:3])

min_date = df['date'].min().to_pydatetime()
max_date = df['date'].max().to_pydatetime()
selected_date_range = st.sidebar.slider("📅 Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

# ── 5. FILTER RAW DATA BASED ON SIDEBAR ─────────────────────────────────────
raw_filtered_df = df[
    (df['date'] >= selected_date_range[0]) & 
    (df['date'] <= selected_date_range[1])
].copy()

if 'park_name' in raw_filtered_df.columns:
    raw_filtered_df = raw_filtered_df[raw_filtered_df['park_name'].isin(selected_parks)]

# Capped at 100 rows so the free web server doesn't crash during live inference!
inference_batch = raw_filtered_df.head(100).copy()

# ── 6. MAIN PANEL - LARGE-SCALE LIVE INFERENCE ──────────────────────────────
st.divider()
st.subheader(f"⚡ Live Transformer Batch Inference ({len(inference_batch)} Records)")

if not inference_batch.empty:
    with st.spinner("🔮 Model processing batch vectors in real-time..."):
        # Run the RAW text strings through the transformer ALL AT ONCE
        live_topics, _ = topic_model.transform(inference_batch['text'].tolist())
        
        # Inject the live classifications back into our temporary view dataframe
        inference_batch['live_topic_id'] = live_topics
        inference_batch['live_category'] = inference_batch['live_topic_id'].map(TOPIC_TRANSLATION_MAP).fillna("Other Operational Issues")

    # Render Dynamic Live Graph
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Real-Time Model Classification Breakdown**")
        live_counts = inference_batch['live_category'].value_counts().reset_index()
        live_counts.columns = ['Category', 'Live Count']
        
        fig = px.bar(live_counts, x='Live Count', y='Category', orientation='h',
                     color='Live Count', color_continuous_scale='Cividis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("**Performance Metrics**")
        st.metric("Batch Inference Size", f"{len(inference_batch)} rows")
        st.metric("Unique Clusters Found", f"{inference_batch['live_category'].nunique()}")
        st.info("💡 Move the slider or change park filters! The model will dynamically recalculate embeddings for up to 100 raw reviews instantly.")

else:
    st.warning("No raw data fits the selected slider range to run inference on.")

# ── 7. MAIN PANEL - SINGLE LIVE REVIEW PREDICTOR ────────────────────────────
st.divider()
st.subheader("🔮 Run Live AI Inference on a Single Custom Review")

user_review = st.text_input(
    label="Type a custom complaint here to test the live model:",
    value="The bathrooms near the main trail entrance were completely out of toilet paper and filthy."
)

if user_review:
    with st.spinner("🧠 Analyzing custom text..."):
        predicted_topics, probs = topic_model.transform([user_review])
        resolved_id = predicted_topics[0]
        
        try:
            topic_words = [word for word, _ in topic_model.get_topic(resolved_id)[:5]]
            keywords_str = f", ".join(topic_words)
        except:
            keywords_str = "N/A"
        
        clean_label = TOPIC_TRANSLATION_MAP.get(resolved_id, f"Topic {resolved_id}")
        
        st.success(f"**AI Classification Result:** Component mapped to **{clean_label}**")
        st.caption(f"🤖 *Top keywords matching this cluster:* {keywords_str}")