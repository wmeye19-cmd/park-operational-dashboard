# ── 1. SIDEBAR & FILTERS (Same as before) ───────────────────────────────────
# Ensure your date sliders and park multiselects are filtering down your raw dataframe
raw_filtered_df = df[
    (df['date'] >= selected_date_range[0]) & 
    (df['date'] <= selected_date_range[1])
].copy()

if 'park_name' in raw_filtered_df.columns:
    raw_filtered_df = raw_filtered_df[raw_filtered_df['park_name'].isin(selected_parks)]

# Limit to a safe batch size (e.g., max 200 records at once) so the free web server stays fast
inference_batch = raw_filtered_df.head(200).copy()

# ── 2. DYNAMIC LARGE-SCALE INFERENCE ────────────────────────────────────────
st.subheader(f"⚡ Live Transformer Batch Inference ({len(inference_batch)} Records)")

if not inference_batch.empty:
    with st.spinner("🔮 Model processing batch vectors in real-time..."):
        # Run the raw text strings through the transformer ALL AT ONCE
        live_topics, _ = topic_model.transform(inference_batch['text'].tolist())
        
        # Inject the live classifications back into our temporary view dataframe
        inference_batch['live_topic_id'] = live_topics
        
        # Map IDs to your human-readable translation dictionary
        TOPIC_TRANSLATION_MAP = {
            -1: "General Feedback / Noise",
            0: "Campground Noise & Disturbance",
            1: "Trail Overgrowth & Hazards",
            6: "Restroom Maintenance & Sanitation",
            12: "Road Conditions & Potholes"
        }
        inference_batch['live_category'] = inference_batch['live_topic_id'].map(TOPIC_TRANSLATION_MAP).fillna("Other Operational Issues")

    # ── 3. RENDER DYNAMIC LIVE GRAPH ────────────────────────────────────────
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
        st.info("💡 Move the slider or change park filters! The model will recalculate embeddings for the new slice instantly.")

else:
    st.warning("No raw data fits the selected slider range to run inference on.")