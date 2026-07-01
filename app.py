import streamlit as st
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Set up beautiful wide page layout
st.set_page_config(layout="wide", page_title="NPS Operational Analytics")

st.title("🌲 National Park Service — AI Operational Dashboard")
st.markdown("*Optimized Multi-Scale Transformer Portfolio Deployment*")

# ── 1. GLOBAL TRANSLATION MAP ───────────────────────────────────────────────
TOPIC_TRANSLATION_MAP = {
    -1: "General Feedback / Noise",
    0: "Campground Noise & Disturbance",
    1: "Trail Overgrowth & Hazards",
    6: "Restroom Maintenance & Sanitation",
    12: "Road Conditions & Potholes"
}

# ── 2. LOAD PRE-CALCULATED DATA (ULTRA-LOW RAM) ─────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("unseen_web_data/web_dashboard_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    # If the column doesn't exist yet, map it dynamically from topic_id
    if 'complaint_category' not in df.columns:
        df['complaint_category'] = df['topic_id'].map(TOPIC_TRANSLATION_MAP).fillna("Other Operational Issues")
    return df

df = load_data()

# ── 3. SIDEBAR CONTROLS ──────────────────────────────────────────────────────
st.sidebar.header("🕹️ Control Dashboard")

all_parks = sorted(df['park_name'].unique().tolist()) if 'park_name' in df.columns else ["Yosemite", "Yellowstone", "Grand Canyon", "Acadia"]
selected_parks = st.sidebar.multiselect("📍 Filter by Park(s)", options=all_parks, default=all_parks[:3])

min_date = df['date'].min().to_pydatetime()
max_date = df['date'].max().to_pydatetime()
selected_date_range = st.sidebar.slider("📅 Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

# ── 4. FILTER DATA INSTANTLY WITH ZERO MEMORY SPIKES ────────────────────────
filtered_df = df[
    (df['date'] >= selected_date_range[0]) & 
    (df['date'] <= selected_date_range[1])
].copy()

if 'park_name' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['park_name'].isin(selected_parks)]

# ── 5. MAIN PANEL - FEATURE 1: HIGH-SCALE DYNAMIC ANALYSIS ──────────────────
st.separator()
st.subheader(f"📊 Dynamic Operational Breakdown ({len(filtered_df):,} Matching Records)")

if not filtered_df.empty:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Model Classification Distribution**")
        # Aggregating pre-calculated categories takes 0ms and uses almost no RAM
        live_counts = filtered_df['complaint_category'].value_counts().reset_index()
        live_counts.columns = ['Category', 'Count']
        
        fig = px.bar(live_counts.head(8), x='Count', y='Category', orientation='h',
                     color='Count', color_continuous_scale='Cividis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("**Performance Metrics**")
        st.metric("Filtered Records", f"{len(filtered_df):,}")
        st.metric("Unique Clusters Displayed", f"{filtered_df['complaint_category'].nunique()}")
        st.success("⚡ Database pipeline optimized! Slider shifts are now instantaneous.")
else:
    st.warning("No data fits the selected slider range.")

# ── 6. MAIN PANEL - FEATURE 2: LAZY-LOADED LIVE SINGLE INFERENCE ─────────────
st.separator()
st.subheader("🔮 Run Live AI Inference on a Single Custom Review")
st.markdown("*This feature loads the transformer model on-demand to process custom text inputs.*")

user_review = st.text_input(
    label="Type a custom complaint here to test the live model:",
    value="The bathrooms near the main trail entrance were completely out of toilet paper and filthy."
)

# We wrap the model loading inside the trigger function so it only consumes RAM when needed
@st.cache_resource
def load_production_ai_model():
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    loaded_model = BERTopic.load("park_complaints_web_lightweight", embedding_model=embedding_model)
    return loaded_model

if user_review:
    with st.spinner("🧠 Inference engine analyzing sentence structures..."):
        topic_model = load_production_ai_model()
        predicted_topics, _ = topic_model.transform([user_review])
        resolved_id = predicted_topics[0]
        
        try:
            topic_words = [word for word, _ in topic_model.get_topic(resolved_id)[:5]]
            keywords_str = f", ".join(topic_words)
        except:
            keywords_str = "N/A"
        
        clean_label = TOPIC_TRANSLATION_MAP.get(resolved_id, f"Topic {resolved_id}")
        
        st.success(f"**AI Classification Result:** Component mapped to **{clean_label}**")
        st.caption(f"🤖 *Top keywords matching this cluster:* {keywords_str}")