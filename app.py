import streamlit as st
from summarize import *

st.set_page_config(
    page_title="TL;DR",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .summary-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

if 'summarizer' not in st.session_state :
    st.session_state.summarizer = None

# Headers
st.markdown('<h1 class="main-header">📝 TL;DR - Text Summarizer</h1>', unsafe_allow_html=True)
st.sidebar.header("⚙️ Configuration")

# Model Selection
model_options = {
    'DistilBART (Recommended)': 'sshleifer/distilbart-cnn-12-6',
    'T5-Small (Fast)': 't5-small',
    'BART-Large (High Quality)': 'facebook/bart-large-cnn'
}

selected_model = st.sidebar.selectbox(
    "Choose Model:",
    options=list(model_options.keys()),
    help="DistilBART offers the best balance of speed and quality"
)

# Advanced parameters
st.sidebar.subheader("Summary Parameters")
max_length = st.sidebar.slider("Maximum Length", 50, 300, 150, 10)
min_length = st.sidebar.slider("Minimum Length", 10, 100, 30, 5)
use_sampling = st.sidebar.checkbox("Enable Sampling", help="More creative but less consistent")
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1) if use_sampling else 1.0

# Load model button
if st.sidebar.button("🔄 Load/Reload Model") :
    model_name = model_options[selected_model]
    
    with st.spinner(f"Loading {selected_model}...") :
        try:
            st.session_state.summarizer = TextSummarizer(model_name)
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")

# Text input options
input_method = st.radio("Input Method:", ["Type/Paste Text", "Upload File"])

text_input = ""
if input_method == "Type/Paste Text" :
    text_input = st.text_area(
        "Enter text to summarize:",
        height=200,
        placeholder="Paste your article, document, or any text here..."
    )
else :
    uploaded_file = st.file_uploader(
        "Upload a text file:",
        type=['txt', 'md'],
        help="Upload .txt or .md files"
    )
    if uploaded_file :
        text_input = str(uploaded_file.read(), "utf-8")
        st.text_area("Uploaded content preview:", text_input[:500] + "...", height=150)

# Summarization
if st.button("🚀 Generate Summary", type="primary") :
    if not text_input.strip() :
        st.warning("⚠️ Please enter some text first.")
    elif st.session_state.summarizer is None :
        st.error("❌ Please load a model first using the sidebar.")
    else :
        with st.spinner("Analyzing and summarizing text...") :
            result = st.session_state.summarizer.summarize(
                text_input,
                max_length=max_length,
                min_length=min_length,
                do_sample=use_sampling,
                temperature=temperature
            )
        
        if result['success'] :
            # Display summary
            st.markdown("### 📋 Summary")
            st.write(result['summary'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display metrics
            st.markdown("### 📊 Analysis Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1 :
                st.metric(
                    "Original Word Count", 
                    result['metadata']['original_word_count']
                )
            
            with col2 :
                st.metric(
                    "Summary Word Count", 
                    result['metadata']['summary_word_count']
                )
            
            with col3 :
                st.metric(
                    "Reduction (%)", 
                    result['metadata']['reduction_percent']
                )
            
            with col4 :
                st.metric(
                    "Processing Time (s)", 
                    result['metadata']['processing_time']
                )
        else :
            st.error(f"❌ {result['error']}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit & Transformers by Shruti Jayaraman</p>
    </div>
    """, 
    unsafe_allow_html=True
)