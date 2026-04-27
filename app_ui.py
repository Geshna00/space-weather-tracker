import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv(dotenv_path=".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NASA_API_KEY       = os.getenv("API_KEY")

# ─────────────────────────────────────────────
# Model config (shown in sidebar for evaluation)
# ─────────────────────────────────────────────
MODEL_NAME = "openrouter/free"
MAX_TOKENS = 1024

TEMP_MODES = {
    "🎯 Factual (0.2)":  {"temperature": 0.2, "top_p": 0.80,
                           "desc": "Very focused, deterministic answers. Best for precise science questions."},
    "⚖️ Balanced (0.5)": {"temperature": 0.5, "top_p": 0.85,
                           "desc": "Balanced between accuracy and natural language flow."},
    "💡 Creative (0.9)": {"temperature": 0.9, "top_p": 0.95,
                           "desc": "More varied, expressive responses. Good for explaining to beginners."},
}

# ─────────────────────────────────────────────
# NASA data fetch + ML model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(api_key):
    today = datetime.today()
    past  = today - timedelta(days=30)
    url   = (f"https://api.nasa.gov/DONKI/FLR"
             f"?startDate={past.date()}&endDate={today.date()}&api_key={api_key}")

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.text.strip() not in ("", "[]"):
            df          = pd.DataFrame(resp.json())
            data_source = "🛰️ Live NASA DONKI API"
        else:
            raise ValueError("Empty response")
    except Exception:
        df = pd.DataFrame({
            'classType': ['C1.0','C2.5','M1.2','M5.3','X1.0',
                          'X2.1','C3.1','M2.0','C4.5','M3.3'],
            'peakTime':  ['2025-12-01T10:00Z','2025-12-05T08:30Z',
                          '2025-12-08T14:00Z','2025-12-12T11:00Z',
                          '2025-12-15T09:00Z','2025-12-18T16:30Z',
                          '2025-12-20T07:00Z','2025-12-22T13:00Z',
                          '2025-12-26T10:30Z','2025-12-29T12:00Z'],
        })
        data_source = "📦 Fallback sample data (NASA API returned no recent flares)"

    def encode_class(x):
        if isinstance(x, str) and x.startswith('X'): return 3
        if isinstance(x, str) and x.startswith('M'): return 2
        if isinstance(x, str) and x.startswith('C'): return 1
        return 0

    df = df[['classType', 'peakTime']].copy()
    df['flare_level'] = df['classType'].apply(encode_class)
    df['peakTime']    = pd.to_datetime(df['peakTime'], errors='coerce')
    df['hour']        = df['peakTime'].dt.hour
    df['day']         = df['peakTime'].dt.day
    df['month']       = df['peakTime'].dt.month
    df = df.dropna()

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(df[['hour','day','month']], df['flare_level'])
    return clf, data_source


# ─────────────────────────────────────────────
# LLM call — robust response extraction
# ─────────────────────────────────────────────
def ask_llm(messages: list, temperature: float, top_p: float) -> str:
    if not OPENROUTER_API_KEY:
        return "⚠️ OPENROUTER_API_KEY not found in .env file."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://space-weather-tracker.local",
        "X-Title":       "Space Weather Tracker",
    }
    body = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "temperature": temperature,
        "top_p":       top_p,
        "max_tokens":  MAX_TOKENS,
    }
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=45,
        )
        resp.raise_for_status()
        result  = resp.json()
        choices = result.get("choices", [])

        if not choices:
            return f"⚠️ Empty choices in API response. Full response: {result}"

        content = choices[0].get("message", {}).get("content", "")
        if content and content.strip():
            return content.strip()

        # Fallback for alternate response shapes
        if "text" in choices[0]:
            return choices[0]["text"].strip()

        return f"⚠️ Unexpected response format: {result}"

    except requests.exceptions.HTTPError:
        return f"⚠️ HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"⚠️ Error: {e}"


# ─────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────
st.set_page_config(page_title="Space Weather Tracker 🌌", page_icon="🌌", layout="wide")
st.title("🌌 Space Weather Tracker")
st.caption("Domain-specific AI chatbot · NASA DONKI API · Random Forest ML · OpenRouter LLM")

model, data_source = load_model(NASA_API_KEY)
st.info(f"**Data source:** {data_source}", icon="ℹ️")

# Current activity prediction
now    = datetime.now()
sample = pd.DataFrame([[now.hour, now.day, now.month]], columns=['hour','day','month'])
pred   = model.predict(sample)[0]

level_map = {
    3: ("🔴 High (X-class)",    "extreme",
        "X-class flares are the most powerful. They can cause radio blackouts, damage satellites, and disrupt power grids."),
    2: ("🟠 Moderate (M-class)","moderate",
        "M-class flares cause brief radio blackouts and minor radiation storms. Affects HF radio in polar regions."),
    1: ("🟡 Low (C-class)",     "low",
        "C-class flares are weak with minimal Earth impact. Considered routine solar activity."),
    0: ("🟢 Minimal",           "minimal",
        "Very quiet Sun. No significant solar activity expected right now."),
}
level_label, level_word, level_info = level_map.get(pred, level_map[0])

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("☀️ Predicted Solar Activity", level_label)
with col2:
    st.info(level_info)

st.divider()

# ─────────────────────────────────────────────
# Sidebar — model configuration panel
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Model Configuration")

    selected_mode = st.radio(
        "**Temperature Mode** *(switch to compare outputs)*",
        options=list(TEMP_MODES.keys()),
        index=0,
    )
    cfg         = TEMP_MODES[selected_mode]
    temperature = cfg["temperature"]
    top_p       = cfg["top_p"]

    st.caption(f"_{cfg['desc']}_")

    st.code(f"""Model:       {MODEL_NAME}
Temperature: {temperature}
Top-p:       {top_p}
Max tokens:  {MAX_TOKENS}""")

    st.markdown("""
**Temperature** controls randomness:
- `0.2` → deterministic, factual
- `0.5` → balanced accuracy + fluency
- `0.9` → creative, varied wording

**Top-p** (nucleus sampling) limits which tokens the model picks from:
- `0.80` → very focused pool
- `0.95` → broader expression allowed
""")

    st.divider()
    st.markdown("**Stack:**")
    st.markdown("- Frontend/Backend: Streamlit")
    st.markdown("- ML Model: Random Forest Classifier")
    st.markdown("- Data API: NASA DONKI")
    st.markdown("- LLM API: OpenRouter (free tier)")
    st.markdown("- Memory: Session-based (multi-turn)")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
# System prompt — domain-constrained role
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are SolarBot, a solar physics and space weather expert assistant.

Current ML prediction (from NASA DONKI solar flare data): activity level is {level_word} — {level_label}.

Your role:
1. Answer questions about solar flares, geomagnetic storms, coronal mass ejections (CMEs), and space weather
2. Explain real-world impacts: GPS accuracy, satellite operations, power grids, aviation, HF radio, and astronaut safety
3. Reference the current predicted activity level in your answers when relevant
4. Stay strictly within the space weather and solar physics domain
5. If asked something unrelated, politely redirect the user back to space weather topics
6. Use clear, concise language (4–6 sentences). Go deeper only if the user asks for technical detail."""

# ─────────────────────────────────────────────
# Chat interface
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render existing conversation
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Quick-start suggestion buttons (only when chat is empty)
if not st.session_state.chat_history:
    st.markdown("**💬 Try asking:**")
    suggestions = [
        "Is solar activity dangerous today?",
        "How do solar flares affect GPS?",
        "What is an X-class solar flare?",
        "Can a solar storm damage satellites?",
        "What is a coronal mass ejection?",
        "How does space weather affect aviation?",
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, key=f"sug_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": s})
            reply = ask_llm(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",   "content": s}],
                temperature, top_p
            )
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

# Chat input
user_input = st.chat_input("Ask about solar flares, space weather, GPS disruption…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    api_messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + st.session_state.chat_history
        + [{"role": "user", "content": user_input}]
    )

    with st.chat_message("assistant"):
        with st.spinner("SolarBot is thinking…"):
            reply = ask_llm(api_messages, temperature, top_p)
        st.markdown(reply)

    st.session_state.chat_history.append({"role": "user",      "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": reply})