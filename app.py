"""
IMDb Movie Review Sentiment Analyzer  (Bi-LSTM)

Run with:
    streamlit run app.py

Folder must contain:
    app.py
    bilstm_sentiment.weights.h5     (or bilstm_sentiment_model.keras)
    word_index.pkl
    requirements.txt
    .streamlit/config.toml          (optional - theme colours)
"""

import os
import re
import html
import json
import pickle
import zipfile
import tempfile

import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# SETTINGS - MUST MATCH THE NOTEBOOK
# ============================================================
VOCAB_SIZE = 10_000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64
INDEX_OFFSET = 3          # 0=PAD, 1=START, 2=UNK, 3=UNUSED

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "bilstm_sentiment.weights.h5")
KERAS_PATH = os.path.join(BASE_DIR, "bilstm_sentiment_model.keras")
WORD_INDEX_PATH = os.path.join(BASE_DIR, "word_index.pkl")

# Test-set results from the notebook (Task 7)
METRICS = {
    "Accuracy": "78.9%",
    "F1-score": "78.8%",
    "ROC-AUC": "0.866",
    "MCC": "0.578",
}

EXAMPLES = {
    "😊 Positive": (
        "This movie was absolutely fantastic! Brilliant acting, a gripping story "
        "and a wonderful soundtrack. I loved every single minute of it."
    ),
    "😞 Negative": (
        "A terrible, boring film. The plot made no sense, the acting was awful "
        "and I wanted to leave halfway through. A complete waste of time."
    ),
    "😐 Mixed": (
        "The visuals were stunning and the music was great, but the story was "
        "slow and the ending felt rushed and disappointing."
    ),
}


# ============================================================
# PAGE CONFIG + STYLING
# ============================================================
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextArea textarea, button {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 820px; }

/* ---------- Force light appearance (works even if browser/OS is in dark mode) ---------- */
:root { color-scheme: light; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #F5F7FB !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] strong,
.stApp label, .stApp label *, [data-testid="stWidgetLabel"] *,
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #0F172A !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #64748B !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color: #4F46E5 !important; }
[data-testid="stSlider"] [role="slider"] { background-color: #6366F1 !important; }
[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #E5E9F2 !important; border-radius: 14px; }
[data-testid="stSidebar"] hr, hr { border-color: #E2E8F0 !important; }

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);
    border-radius: 22px;
    padding: 2.2rem 2rem 2rem 2rem;
    color: #fff;
    box-shadow: 0 18px 40px -14px rgba(79, 70, 229, .55);
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute; right: -60px; top: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,.10);
}
.hero h1 { color:#fff !important; font-size: 2rem; font-weight: 800; margin: 0 0 .35rem 0; letter-spacing:-.5px; }
.hero p  { color: rgba(255,255,255,.9) !important; margin: 0; font-size: 1rem; }
.pill-row { margin-top: 1.1rem; display:flex; gap:.5rem; flex-wrap:wrap; }
.pill {
    background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.28);
    padding: .25rem .75rem; border-radius: 999px;
    font-size: .78rem; font-weight: 600; color:#fff !important;
    backdrop-filter: blur(6px);
}

/* ---------- Section titles ---------- */
.section-title {
    font-size: .8rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: #64748B; margin: .2rem 0 .5rem 0;
}

/* ---------- Cards (bordered containers) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-radius: 18px !important;
    border: 1px solid #E5E9F2 !important;
    box-shadow: 0 6px 22px -12px rgba(15, 23, 42, .18);
}

/* ---------- Text area ---------- */
.stTextArea textarea {
    border-radius: 14px !important;
    border: 1.5px solid #E2E8F0 !important;
    background: #F8FAFC !important;
    font-size: 1rem !important;
    line-height: 1.55 !important;
    padding: .9rem 1rem !important;
}
.stTextArea textarea, .stTextArea textarea:focus {
    color: #0F172A !important; -webkit-text-fill-color: #0F172A !important; caret-color: #4F46E5;
}
.stTextArea textarea::placeholder { color: #94A3B8 !important; -webkit-text-fill-color: #94A3B8 !important; opacity: 1; }
.stTextArea [data-baseweb="textarea"], .stTextArea [data-baseweb="base-input"] {
    background: #F8FAFC !important; border-radius: 14px !important;
}
.stTextArea textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,.15) !important;
    background: #fff !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 12px; font-weight: 600; padding: .55rem 1rem;
    border: 1.5px solid #E2E8F0; transition: all .15s ease; width: 100%;
}
.stButton > button { background: #FFFFFF; }
.stButton > button, .stButton > button p { color: #0F172A !important; }
.stButton > button[kind="primary"], .stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] * { color: #FFFFFF !important; }
.stButton > button:hover { border-color:#6366F1; color:#4F46E5; transform: translateY(-1px); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: #fff; border: none; font-size: 1rem; padding: .7rem 1rem;
    box-shadow: 0 10px 22px -10px rgba(79,70,229,.8);
}
.stButton > button:hover, .stButton > button:hover p { color:#4F46E5 !important; }
.stButton > button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover p { color:#fff !important; filter: brightness(1.07); transform: translateY(-1px); }

/* ---------- Result ---------- */
.result {
    border-radius: 18px; padding: 1.4rem 1.5rem; margin-bottom: 1rem;
    display:flex; align-items:center; gap: 1.2rem;
}
.result.pos { background: linear-gradient(135deg,#ECFDF5,#D1FAE5); border:1px solid #A7F3D0; }
.result.neg { background: linear-gradient(135deg,#FEF2F2,#FEE2E2); border:1px solid #FECACA; }
.result .emoji { font-size: 3rem; line-height:1; }
.result .label { font-size: .75rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
.result.pos .label { color:#047857; }  .result.neg .label { color:#B91C1C; }
.result .verdict { font-size: 1.9rem; font-weight: 800; letter-spacing:-.5px; color:#0F172A; line-height:1.15; }
.result .sub { color:#475569; font-size:.9rem; margin-top:.15rem; }

/* ---------- Sentiment meter ---------- */
.meter-wrap { margin: .4rem 0 1.1rem 0; }
.meter {
    position: relative; height: 14px; border-radius: 999px;
    background: linear-gradient(90deg, #EF4444 0%, #F59E0B 50%, #10B981 100%);
}
.meter .marker {
    position:absolute; top:50%; width: 24px; height: 24px; border-radius:50%;
    background:#fff; border: 4px solid #0F172A; transform: translate(-50%,-50%);
    box-shadow: 0 4px 10px rgba(0,0,0,.25);
}
.meter-labels span { color:#64748B !important; }
.meter-labels { display:flex; justify-content:space-between; font-size:.78rem; font-weight:600; color:#64748B; margin-top:.55rem; }

/* ---------- Stat tiles ---------- */
.stats { display:grid; grid-template-columns: repeat(3, 1fr); gap:.8rem; }
.stat { background:#F8FAFC; border:1px solid #E8ECF4; border-radius:14px; padding:.9rem 1rem; }
.stat .k { font-size:.72rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:#64748B; }
.stat .v { font-size:1.55rem; font-weight:800; color:#0F172A; margin-top:.15rem; letter-spacing:-.5px; }
.stat .d { font-size:.76rem; color:#94A3B8; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E5E9F2; }
.side-title { font-weight:800; font-size:1.05rem; color:#0F172A; }
.kv { display:flex; justify-content:space-between; padding:.42rem 0; border-bottom:1px dashed #E2E8F0; font-size:.88rem; }
.kv span:first-child { color:#64748B !important; } .kv span:last-child { font-weight:700; color:#0F172A !important; }
.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:.55rem; margin:.5rem 0 1rem 0; }
.metric-box { background:linear-gradient(135deg,#EEF2FF,#F5F3FF); border:1px solid #E0E7FF; border-radius:12px; padding:.6rem .7rem; }
.metric-box .k { font-size:.68rem; font-weight:700; color:#6366F1; text-transform:uppercase; letter-spacing:.06em; }
.metric-box .v { font-size:1.15rem; font-weight:800; color:#1E1B4B; }

.hist { width:100%; border-collapse:collapse; font-size:.88rem; }
.hist th { text-align:left; font-size:.72rem; letter-spacing:.07em; text-transform:uppercase; color:#64748B !important;
           padding:.5rem .6rem; border-bottom:2px solid #E2E8F0; }
.hist td { padding:.55rem .6rem; border-bottom:1px solid #EEF2F7; color:#0F172A !important; }
.hist tr:last-child td { border-bottom:none; }
.footer { text-align:center; color:#94A3B8; font-size:.8rem; margin-top:1.6rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================
def _strip_key(obj, key="quantization_config"):
    """Remove a config key that older Keras versions don't recognise."""
    if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
            _strip_key(v, key)
    elif isinstance(obj, list):
        for v in obj:
            _strip_key(v, key)


def _load_keras_file_compat(path):
    """Load a .keras file, patching configs written by newer Keras versions."""
    try:
        return keras.models.load_model(path)
    except (TypeError, ValueError):
        patched = os.path.join(tempfile.mkdtemp(), "patched_model.keras")
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(patched, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "config.json":
                    cfg = json.loads(data)
                    _strip_key(cfg)
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(item, data)
        return keras.models.load_model(patched, compile=False)


@st.cache_resource(show_spinner=False)
def load_model():
    # Preferred: rebuild the architecture and load the weights file
    if os.path.exists(WEIGHTS_PATH):
        model = keras.Sequential([
            keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,)),
            keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True),
            keras.layers.Bidirectional(keras.layers.LSTM(LSTM_UNITS, return_sequences=False)),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
        model.load_weights(WEIGHTS_PATH)
        return model
    # Fallback: full saved model
    return _load_keras_file_compat(KERAS_PATH)


@st.cache_resource(show_spinner=False)
def load_word_index():
    with open(WORD_INDEX_PATH, "rb") as f:
        return pickle.load(f)


# ============================================================
# PREPROCESSING + PREDICTION
# ============================================================
def encode_review(text, word_index):
    """Raw text -> padded integer sequence (identical to the notebook)."""
    text = re.sub(r"<br\s*/?>", " ", text.lower())
    words = re.findall(r"[a-z0-9']+", text)

    sequence, unknown = [1], 0                       # <START>
    for word in words:
        idx = word_index.get(word)
        idx = None if idx is None else idx + INDEX_OFFSET
        if idx is None or idx >= VOCAB_SIZE:
            sequence.append(2)                       # <UNK>
            unknown += 1
        else:
            sequence.append(idx)

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",          # notebook: padding="post"
        truncating="post",       # notebook: truncating="post"
    )
    return padded, len(words), unknown


def predict(text, model, word_index):
    padded, n_words, unknown = encode_review(text, word_index)
    prob = float(model.predict(padded, verbose=0)[0][0])
    return prob, n_words, unknown


# ============================================================
# SESSION STATE
# ============================================================
st.session_state.setdefault("review", "")
st.session_state.setdefault("history", [])


def set_example(name):
    st.session_state["review"] = EXAMPLES[name]


def clear_text():
    st.session_state["review"] = ""


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="side-title">🎬 Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.caption("Deep-learning movie review classifier")

    st.markdown('<div class="section-title" style="margin-top:1rem">Test performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-grid">'
        + "".join(f'<div class="metric-box"><div class="k">{k}</div><div class="v">{v}</div></div>'
                  for k, v in METRICS.items())
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Model details</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="kv"><span>Architecture</span><span>Bi-LSTM</span></div>
<div class="kv"><span>Vocabulary</span><span>{VOCAB_SIZE:,} words</span></div>
<div class="kv"><span>Sequence length</span><span>{MAX_SEQUENCE_LENGTH} tokens</span></div>
<div class="kv"><span>Embedding dim</span><span>{EMBEDDING_DIM}</span></div>
<div class="kv"><span>LSTM units</span><span>{LSTM_UNITS} × 2</span></div>
<div class="kv"><span>Parameters</span><span>1.38 M</span></div>
<div class="kv"><span>Dataset</span><span>IMDb (50K)</span></div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title" style="margin-top:1.1rem">Settings</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Decision threshold", 0.10, 0.90, 0.50, 0.05,
        help="A review is labelled Positive when its positive probability is at or above this value.",
    )
    show_history = st.toggle("Show analysis history", value=True)

    st.divider()
    st.caption("Built with TensorFlow/Keras & Streamlit")


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
<div class="hero">
  <h1>🎬 IMDb Review Sentiment Analyzer</h1>
  <p>Paste any movie review and a Bidirectional LSTM will tell you whether it's positive or negative.</p>
  <div class="pill-row">
    <span class="pill">Bi-LSTM</span>
    <span class="pill">10K vocabulary</span>
    <span class="pill">100-token input</span>
    <span class="pill">78.9% accuracy</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FILE CHECK + LOAD
# ============================================================
missing = []
if not (os.path.exists(WEIGHTS_PATH) or os.path.exists(KERAS_PATH)):
    missing.append("bilstm_sentiment.weights.h5  (or bilstm_sentiment_model.keras)")
if not os.path.exists(WORD_INDEX_PATH):
    missing.append("word_index.pkl")
if missing:
    st.error("Missing required file(s):\n\n" + "\n".join(f"- {m}" for m in missing))
    st.stop()

try:
    with st.spinner("Loading model..."):
        model = load_model()
        word_index = load_word_index()
except Exception as e:  # noqa: BLE001
    st.error("❌ Failed to load the Bi-LSTM model.")
    st.exception(e)
    st.stop()


# ============================================================
# INPUT CARD
# ============================================================
with st.container(border=True):
    st.markdown('<div class="section-title">Your review</div>', unsafe_allow_html=True)

    st.text_area(
        "Review",
        key="review",
        height=170,
        placeholder="e.g. The acting was superb and the story kept me hooked until the very end...",
        label_visibility="collapsed",
    )

    n_chars = len(st.session_state["review"])
    n_words_live = len(st.session_state["review"].split())
    st.caption(f"{n_words_live} words · {n_chars} characters")

    st.markdown('<div class="section-title" style="margin-top:.4rem">Try an example</div>', unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLES))
    for col, name in zip(ex_cols, EXAMPLES):
        col.button(name, key=f"ex_{name}", on_click=set_example, args=(name,))

    b1, b2 = st.columns([3, 1])
    analyze = b1.button("🔍  Analyze Sentiment", type="primary")
    b2.button("Clear", on_click=clear_text)


# ============================================================
# RESULT
# ============================================================
if analyze:
    review = st.session_state["review"]
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            prob, n_words, unknown = predict(review, model, word_index)

        is_pos = prob >= threshold
        label = "Positive" if is_pos else "Negative"
        confidence = prob if prob >= 0.5 else 1 - prob
        used = min(n_words + 1, MAX_SEQUENCE_LENGTH)
        known_pct = 100 * (n_words - unknown) / n_words if n_words else 0

        st.session_state["history"].insert(
            0,
            {
                "Review": (review[:70] + "…") if len(review) > 70 else review,
                "Sentiment": ("😊 " if is_pos else "😞 ") + label,
                "Positive prob.": f"{prob * 100:.1f}%",
            },
        )
        st.session_state["history"] = st.session_state["history"][:8]

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
<div class="result {'pos' if is_pos else 'neg'}">
  <div class="emoji">{'😊' if is_pos else '😞'}</div>
  <div>
    <div class="label">Predicted sentiment</div>
    <div class="verdict">{label} review</div>
    <div class="sub">The model is <b>{confidence * 100:.1f}%</b> confident in this prediction.</div>
  </div>
</div>

<div class="meter-wrap">
  <div class="meter"><div class="marker" style="left:{max(2, min(98, prob * 100)):.1f}%"></div></div>
  <div class="meter-labels"><span>Negative</span><span>Positive probability: {prob * 100:.1f}%</span><span>Positive</span></div>
</div>

<div class="stats">
  <div class="stat"><div class="k">Confidence</div><div class="v">{confidence * 100:.1f}%</div><div class="d">in predicted class</div></div>
  <div class="stat"><div class="k">Tokens used</div><div class="v">{used}<span style="font-size:.9rem;color:#94A3B8"> / {MAX_SEQUENCE_LENGTH}</span></div><div class="d">{n_words} words in review</div></div>
  <div class="stat"><div class="k">Known words</div><div class="v">{known_pct:.0f}%</div><div class="d">{unknown} unknown / rare</div></div>
</div>
""",
                unsafe_allow_html=True,
            )

            if n_words + 1 > MAX_SEQUENCE_LENGTH:
                st.info(
                    f"Your review has {n_words} words. The model only reads the first "
                    f"{MAX_SEQUENCE_LENGTH} tokens, so the rest was ignored."
                )
            if n_words and known_pct < 60:
                st.warning("Many words were not in the model's vocabulary, so this prediction may be less reliable.")


# ============================================================
# HISTORY
# ============================================================
if show_history and st.session_state["history"]:
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.markdown('<div class="section-title">Recent analyses</div>', unsafe_allow_html=True)
        if h2.button("Reset", key="reset_hist"):
            st.session_state["history"] = []
            st.rerun()
        rows = "".join(
            f"<tr><td>{html.escape(r['Review'])}</td><td>{html.escape(r['Sentiment'])}</td>"
            f"<td>{html.escape(r['Positive prob.'])}</td></tr>"
            for r in st.session_state["history"]
        )
        st.markdown(
            "<table class='hist'><thead><tr><th>Review</th><th>Sentiment</th><th>Positive prob.</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )


# ============================================================
# HOW IT WORKS + FOOTER
# ============================================================
with st.expander("ℹ️  How does it work?"):
    st.markdown(
        f"""
1. **Tokenise** – the review is lower-cased, HTML tags are removed and it is split into words.
2. **Encode** – each word becomes its IMDb index (+3 offset). Words outside the top {VOCAB_SIZE:,} become `<UNK>`.
3. **Pad / truncate** – sequences are fixed to {MAX_SEQUENCE_LENGTH} tokens (same as training).
4. **Predict** – an Embedding layer feeds a Bidirectional LSTM that reads the text forwards and backwards; a sigmoid output gives the probability of a *positive* review.
"""
    )

st.markdown(
    '<div class="footer">Bi-LSTM · IMDb sentiment classification · Test accuracy ≈ 78.9% · ROC-AUC ≈ 0.866</div>',
    unsafe_allow_html=True,
)