"""
🌿 Alerta Verde — Sistema de Monitoreo de Paneles Solares

Punto de entrada Streamlit: navegación multi-página, branding e inicialización de recursos.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import streamlit as st

from src.chatbot_logic import get_gemini_api_key_from_secrets, get_gemini_generative_model
from src.firebase_client import get_firestore_client
from src.ml_model import get_trained_model_and_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_shared_resources() -> Tuple[Any, Dict[str, Any]]:
    """Precarga modelo ML, cliente Firebase y modelo Gemini (si hay API key).

    Los fallos parciales (p. ej. Gemini) no impiden cargar el modelo ni Firebase.
    """
    model, metrics = get_trained_model_and_metrics()
    get_firestore_client()
    key = get_gemini_api_key_from_secrets()
    if key:
        try:
            get_gemini_generative_model(key)
        except Exception as exc:
            logger.warning("Gemini no pudo precargarse (la app sigue en modo demo): %s", exc)
    return model, metrics


st.set_page_config(
    page_title="🌿 Alerta Verde — Monitoreo Solar",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "## 🌿 Alerta Verde\n"
            "Monitoreo inteligente de paneles solares con ML (Random Forest), "
            "Firestore y análisis contextual con Google Gemini.\n\n"
            "Hackathon 2026."
        ),
    },
)

# Precarga modelo, Firebase y Gemini (cuando aplique)
try:
    _SHARED_MODEL, _SHARED_METRICS = init_shared_resources()
except Exception as exc:
    logger.exception("Error en inicialización de recursos compartidos")
    st.session_state["_init_error"] = str(exc)
    _SHARED_MODEL, _SHARED_METRICS = None, {}

st.markdown("""
<style>
    :root {
        --verde-primario: #00C853;
        --verde-oscuro: #1B5E20;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .hero-card {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 40%, #43A047 70%, #66BB6A 100%);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(27,94,32,0.35);
        animation: fadeInUp 0.8s ease-out;
    }
    .hero-card h1 {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    .hero-card p {
        font-size: 1.15rem;
        opacity: 0.92;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .stat-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #e0e0e0;
        animation: fadeInUp 0.8s ease-out;
    }
    .stat-value { font-size: 2.2rem; font-weight: 800; color: #1B5E20; }
    .stat-label { font-size: 0.9rem; color: #757575; text-transform: uppercase; }
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border-left: 5px solid var(--verde-primario);
        height: 100%;
    }
    .feature-card h3 { color: #1B5E20; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown span {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌿 Alerta Verde")
    st.markdown("---")
    st.markdown("""
    ### 📍 Navegación
    Usa el menú de páginas para:
    - 📊 **Dashboard**
    - 🔍 **Detección de Anomalías**
    - 🤖 **Asistente Solar**
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Estado del sistema")

    try:
        gemini_ok = bool(st.secrets.get("GEMINI_API_KEY", ""))
    except Exception:
        gemini_ok = False

    try:
        firebase_ok = bool(st.secrets.get("FIREBASE_PROJECT_ID", ""))
    except Exception:
        firebase_ok = False

    st.markdown(f"- Gemini API: {'🟢 Configurado' if gemini_ok else '🟡 Modo demo'}")
    st.markdown(f"- Firebase: {'🟢 Configurado' if firebase_ok else '🟡 Modo demo'}")

    if st.session_state.get("_init_error"):
        st.warning(f"Inicialización parcial: {st.session_state['_init_error']}")

    st.markdown("---")
    st.caption("Hackathon 2026 — Universidad de los Andes")

st.markdown("""
<div class="hero-card">
    <h1>🌿 Alerta Verde</h1>
    <p>
        Monitoreo de plantas fotovoltaicas con detección de anomalías (Random Forest),
        persistencia en Firebase y explicaciones con Gemini — más el Asistente Solar para comunidades rurales.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div style="font-size:2.5rem;">📡</div>
        <div class="stat-value">10+</div>
        <div class="stat-label">Sensores (demo)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div style="font-size:2.5rem;">📊</div>
        <div class="stat-value">ML</div>
        <div class="stat-label">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div style="font-size:2.5rem;">🤖</div>
        <div class="stat-value">Gemini</div>
        <div class="stat-label">IA contextual</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div style="font-size:2.5rem;">☁️</div>
        <div class="stat-value">Cloud</div>
        <div class="stat-label">Streamlit + GCP</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## 🚀 Módulos")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Dashboard</h3>
        <p>Series temporales, KPIs y mapas de calor con Plotly; datos Firebase o CSV.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <h3>🔍 Detección</h3>
        <p>Clasificación de anomalías con métricas completas y explicación Gemini.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 Asistente Solar</h3>
        <p>Chatbot educativo para instalación, costos y sistemas off-grid.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("🌿 Alerta Verde — listo para Streamlit Cloud")
