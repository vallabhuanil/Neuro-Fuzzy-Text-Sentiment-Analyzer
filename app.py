# app.py — Neuro‑Fuzzy Sentiment Analyzer (Premium Dark UI)
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
from matplotlib.colors import LinearSegmentedColormap

# Set dark theme for matplotlib
plt.style.use('dark_background')

# --- Page setup ---
st.set_page_config(
    page_title="Neuro‑Fuzzy Sentiment Analyzer", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Dark Theme CSS with Animated Background
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(-45deg, #0c0c0c, #1a1a2e, #16213e, #0f3460);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .title-container {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2.5rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .title-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(120, 119, 198, 0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .title-text {
        background: linear-gradient(135deg, #ff6b6b 0%, #74b9ff 50%, #00b894 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle-text {
        color: rgba(255, 255, 255, 0.7);
        font-weight: 300;
        font-size: 1.3rem;
        letter-spacing: 0.5px;
    }
    
    .stButton button {
        width: 100%;
        border-radius: 15px;
        height: 3.5rem;
        font-weight: 600;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border: none;
        font-size: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton button:hover::before {
        left: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
    }
    
    .analyze-btn button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .demo-btn button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .clear-btn button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.05);
        color: white;
        font-size: 1rem;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.4);
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ff6b6b, #74b9ff, #00b894);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #74b9ff, #00b894);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .sentiment-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .sentiment-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.3);
    }
    
    .strong-neg { 
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white; 
    }
    .some-neg { 
        background: linear-gradient(135deg, #ffa94d, #ff8c42);
        color: white; 
    }
    .neutral { 
        background: linear-gradient(135deg, #ffeaa7, #fdcb6e);
        color: #2d3436; 
    }
    .some-pos { 
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white; 
    }
    .strong-pos { 
        background: linear-gradient(135deg, #00b894, #00a085);
        color: white; 
    }
    
    .membership-bar {
        height: 10px;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.1);
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .membership-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .membership-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .plot-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    
    .example-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        color: rgba(255, 255, 255, 0.8);
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        font-size: 0.9rem;
    }
    
    .example-btn:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(10px) scale(1.02);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    
    .info-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .info-panel::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #f093fb, #4facfe);
    }
    
    .glow-text {
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'demo_clicked' not in st.session_state:
    st.session_state.demo_clicked = False
if 'clear_clicked' not in st.session_state:
    st.session_state.clear_clicked = False

# Premium Title Section
st.markdown("""
<div class="title-container">
    <div class="title-text glow-text">🧠 Neuro‑Fuzzy Sentiment Analyzer</div>
    <div class="subtitle-text">Hybrid AI • Neural Probability → Fuzzy Human‑Style Sentiment Levels</div>
</div>
""", unsafe_allow_html=True)

# --- EXACT SAME FUZZY LOGIC from your working notebook ---
def tri(x, a, b, c):
    """Robust triangular membership function that handles all edge cases"""
    if a == b == c:
        return 1.0 if x == a else 0.0
    if a == b:
        if x == a: return 1.0
        elif x < a: return 0.0
        elif x <= c: return (c - x) / (c - a)
        else: return 0.0
    if b == c:
        if x == c: return 1.0
        elif x > c: return 0.0
        elif x >= a: return (x - a) / (c - a)
        else: return 0.0
    if x <= a or x >= c: return 0.0
    elif a <= x <= b: return (x - a) / (b - a)
    elif b <= x <= c: return (c - x) / (c - b)
    return 0.0

BINS = {
    "strong_neg": (0.00, 0.00, 0.20),
    "some_neg":   (0.10, 0.30, 0.45),
    "neutral":    (0.35, 0.50, 0.65),
    "some_pos":   (0.55, 0.70, 0.85),
    "strong_pos": (0.80, 1.00, 1.00),
}

def fuzzy_memberships(p: float):
    p = float(np.clip(p, 0.0, 1.0))
    mu = {name: float(tri(p, *abc)) for name, abc in BINS.items()}
    return mu

def fuzzy_decide(p: float):
    mu = fuzzy_memberships(p)
    non_zero = {k: v for k, v in mu.items() if v > 0}
    if not non_zero: return "neutral", mu
    max_score = max(non_zero.values())
    candidates = [k for k, v in non_zero.items() if v == max_score]
    if len(candidates) > 1:
        priority_order = ["neutral", "some_neg", "some_pos", "strong_neg", "strong_pos"]
        for label in priority_order:
            if label in candidates: return label, mu
    return candidates[0], mu

def plot_memberships_fig(mu: dict, title="Fuzzy memberships"):
    keys = ["strong_neg","some_neg","neutral","some_pos","strong_pos"]
    vals = [mu.get(k,0.0) for k in keys]
    
    # Create dark theme plot
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='none')
    ax.set_facecolor('none')
    
    # Custom colors for dark theme
    colors = ["#ff6b6b", "#ffa94d", "#ffeaa7", "#74b9ff", "#00b894"]
    
    bars = ax.bar(keys, vals, color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.set_ylim(0,1)
    ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=20)
    
    # Customize the plot
    ax.tick_params(colors='white', labelsize=10)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.grid(True, alpha=0.2, color='white')
    
    # Add value labels on bars
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f'{v:.2f}', 
                ha='center', va='bottom', color='white', fontweight='bold', fontsize=9)
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig

# --- LOAD YOUR ACTUAL TRAINED MODEL ---
@st.cache_resource
def load_model():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            baseline = pickle.load(f)
        return baseline
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run the Jupyter notebook first.")
        return None

baseline = load_model()

if baseline is not None:
    # Text cleaning
    def clean_text(s: str) -> str:
        HTML_TAG_RE = re.compile(r"<.*?>")
        s = str(s)
        s = re.sub(HTML_TAG_RE, " ", s)
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def analyze_text(txt: str):
        cleaned_text = clean_text(txt)
        p = float(baseline.predict_proba([cleaned_text])[0,1])
        label, mu = fuzzy_decide(p)
        return p, label, mu

    # --- Main UI ---
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown('<div class="main">', unsafe_allow_html=True)
        
        # Handle button clicks first
        demo_text = "The performances were wonderful but the plot dragged; overall, slightly disappointing."
        
        # Check for button clicks and update session state
        if st.session_state.demo_clicked:
            st.session_state.text_input = demo_text
            st.session_state.demo_clicked = False
            st.rerun()
            
        if st.session_state.clear_clicked:
            st.session_state.text_input = ""
            st.session_state.clear_clicked = False
            st.rerun()
        
        # Premium Example Section
        st.markdown("### 🎯 Try These Examples")
        example_texts = [
            "This product is absolutely amazing! I love everything about it.",
            "The service was terrible and the staff was rude.", 
            "It's okay, nothing special but not bad either.",
            "A masterpiece of storytelling with brilliant performances.",
            "Waste of time. Poorly executed and boring."
        ]
        
        # Create beautiful example buttons
        for i, example in enumerate(example_texts):
            if st.button(f"💬 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.text_input = example
                st.rerun()

        # Premium Text Input
        txt = st.text_area(
            "**✨ Enter Text to Analyze**", 
            height=140,
            placeholder="Paste your review, comment, or any text here for advanced sentiment analysis...",
            value=st.session_state.text_input,
            key="text_input"
        )

        # Premium Action Buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1,1,1], gap="medium")
        with btn_col1:
            analyze_btn = st.button("🔍 **Analyze Sentiment**", use_container_width=True, key="analyze")
        with btn_col2:
            if st.button("🎭 **Load Demo**", use_container_width=True, key="demo"):
                st.session_state.demo_clicked = True
                st.rerun()
        with btn_col3:
            if st.button("🗑️ **Clear**", use_container_width=True, key="clear"):
                st.session_state.clear_clicked = True
                st.rerun()

        if analyze_btn and txt.strip():
            with st.spinner("🤖 **Advanced AI Analysis in Progress...**"):
                p, label, mu = analyze_text(txt)
            
            # Premium Results Display
            st.markdown("---")
            st.markdown("### 📊 **Advanced Analysis Results**")
            
            # Premium Metrics Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Neural Probability</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{p:.3f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Fuzzy Sentiment</div>', unsafe_allow_html=True)
                display_labels = {
                    "strong_neg": "Strong Negative",
                    "some_neg": "Somewhat Negative", 
                    "neutral": "Neutral",
                    "some_pos": "Somewhat Positive", 
                    "strong_pos": "Strong Positive"
                }
                display_label = display_labels.get(label, label)
                badge_class = label.replace("_", "-")
                st.markdown(f'<div class="sentiment-badge {badge_class} pulse">{display_label}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">AI Confidence</div>', unsafe_allow_html=True)
                confidence = max(mu.values())
                st.markdown(f'<div class="metric-value">{confidence:.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Premium Membership Visualization
            st.markdown("#### 🎚️ **Fuzzy Membership Distribution**")
            for sentiment, value in sorted(mu.items(), key=lambda x: x[1], reverse=True):
                display_name = display_labels.get(sentiment, sentiment)
                color_map = {
                    "strong_neg": "#ff6b6b",
                    "some_neg": "#ffa94d", 
                    "neutral": "#ffeaa7",
                    "some_pos": "#74b9ff",
                    "strong_pos": "#00b894"
                }
                color = color_map.get(sentiment, "#666666")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{display_name}**")
                with col2:
                    st.markdown(f"`{value:.3f}`")
                    st.markdown(
                        f'<div class="membership-bar">'
                        f'<div class="membership-fill" style="width: {value*100}%; background: {color};"></div>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )

            # Premium Visualizations
            st.markdown("#### 📈 **Advanced Visualizations**")
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("##### **Membership Distribution**")
                f1 = plot_memberships_fig(mu, title=f"Fuzzy Membership Analysis")
                st.pyplot(f1)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with viz_col2:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("##### **Sentiment Landscape**")
                # Create a simple radar-like visualization
                fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
                ax.set_facecolor('none')
                
                categories = list(mu.keys())
                values = list(mu.values())
                
                colors = ["#ff6b6b", "#ffa94d", "#ffeaa7", "#74b9ff", "#00b894"]
                bars = ax.barh(categories, values, color=colors, alpha=0.8)
                
                ax.set_xlim(0, 1)
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.grid(True, alpha=0.2, color='white')
                
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif analyze_btn and not txt.strip():
            st.warning("⚠️ **Please enter some text to analyze**")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Premium Info Panel
        st.markdown('<div class="info-panel">', unsafe_allow_html=True)
        st.markdown("### ℹ️ **How It Works**")
        st.markdown("""
        <div style="color: rgba(255,255,255,0.8); line-height: 1.6;">
        🧠 **Neuro-Fuzzy Architecture:**
        
        • **Neural Layer**: TF-IDF + Logistic Regression
        • **Fuzzy Logic**: Triangular membership functions  cat
        • **Hybrid AI**: Combines ML accuracy with human-like reasoning
        • **Real-time**: Instant analysis with visual feedback
        
        ⚡ **Powered by your trained 90%+ accurate model**
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 **Sentiment Spectrum**")
        st.markdown("""
        <div style="color: rgba(255,255,255,0.8);">
        """, unsafe_allow_html=True)
        st.markdown('<div class="sentiment-badge strong-pos">🌟 Strong Positive</div>', unsafe_allow_html=True)
        st.markdown('<div class="sentiment-badge some-pos">💙 Somewhat Positive</div>', unsafe_allow_html=True) 
        st.markdown('<div class="sentiment-badge neutral">⚖️ Neutral</div>', unsafe_allow_html=True)
        st.markdown('<div class="sentiment-badge some-neg">💔 Somewhat Negative</div>', unsafe_allow_html=True)
        st.markdown('<div class="sentiment-badge strong-neg">🔥 Strong Negative</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Fuzzy Logic Test
        st.markdown("### 🔧 **Fuzzy Logic Test**")
        test_p = st.slider("Test Probability", 0.0, 1.0, 0.5, 0.01, key="test_slider")
        test_mu = fuzzy_memberships(test_p)
        test_label = max(test_mu.items(), key=lambda kv: kv[1])[0]
        display_label = {
            "strong_neg": "Strong Negative",
            "some_neg": "Somewhat Negative", 
            "neutral": "Neutral",
            "some_pos": "Somewhat Positive", 
            "strong_pos": "Strong Positive"
        }.get(test_label, test_label)
        
        st.markdown(f"**Probability**: `{test_p:.2f}`")
        st.markdown(f"**Fuzzy Label**: `{display_label}`")
        
        for k, v in test_mu.items():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"`{k}`")
            with col2:
                st.write(f"`{v:.3f}`")
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("""
    ## 🚀 **Setup Required**
    
    1. **Run your Jupyter notebook** and execute the save cell
    2. **Place this app.py** in the same folder as your notebook  
    3. **Run**: `streamlit run app.py`
    
    The save cell should create **'sentiment_model.pkl'**
    """)