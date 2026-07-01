import streamlit as st
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

st.title("🌲 NPS Live AI Transformer Dashboard")

# ── 1. LOAD THE LIGHTWEIGHT MODEL LIVE ──────────────────────────────────────
@st.cache_resource # cache_resource keeps the AI model loaded in memory permanently
def load_production_ai_model():
    # 1. Load the lightweight topic architecture you saved
    loaded_model = BERTopic.load("park_complaints_web_lightweight")
    
    # 2. Re-attach the language backbone so it can read new text
    loaded_model.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return loaded_model

with st.spinner("🧠 Loading Multilingual Transformer Model into Web RAM..."):
    topic_model = load_production_ai_model()

# ── 2. THE LIVE INFERENCE INTERACTION ───────────────────────────────────────
st.subheader("🔮 Run Live AI Inference on a New Review")

user_review = st.text_input(
    label="Type a custom complaint here to test the live model:",
    value="The bathrooms near the main trail entrance were completely out of toilet paper and filthy."
)

if user_review:
    # This line runs the text through the transformer live on your machine!
    predicted_topics, probs = topic_model.transform([user_review])
    resolved_id = predicted_topics[0]
    
    # Grab the top words associated with that predicted topic
    topic_words = [word for word, _ in topic_model.get_topic(resolved_id)[:5]]
    
    # Translate to human-readable (Using the dictionary we set up before)
    TOPIC_TRANSLATION_MAP = {
        -1: "General Feedback / Noise",
        0: "Campground Noise & Disturbance",
        1: "Trail Overgrowth & Hazards",
        6: "Restroom Maintenance & Sanitation",
        12: "Road Conditions & Potholes"
    }
    
    clean_label = TOPIC_TRANSLATION_MAP.get(resolved_id, f"Topic {resolved_id}")
    
    # Display the result
    st.success(f"**AI Classification Result:** Component mapped to **{clean_label}**")
    st.caption(f"🤖 *Top keywords matching this cluster:* {', '.join(topic_words)}")