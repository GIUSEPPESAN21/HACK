"""
🤖 Asistente Solar — Chat educativo con Gemini (modo demo sin API key).
"""

from __future__ import annotations

import streamlit as st

from src.chatbot_logic import (
    clear_chat_history,
    generate_response,
    get_gemini_api_key_from_secrets,
    get_suggested_questions,
    initialize_chat_history,
)

st.set_page_config(
    page_title="🤖 Asistente Solar — Alerta Verde",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
    .chat-header {
        background: linear-gradient(135deg, #E65100 0%, #FF9800 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .welcome-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #FFE0B2;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

initialize_chat_history()
gemini_key = get_gemini_api_key_from_secrets()

with st.sidebar:
    st.markdown("## 🤖 Asistente Solar")
    st.markdown("**Estado:** " + ("🟢 Gemini activo" if gemini_key else "🟡 Modo demo"))
    if not gemini_key:
        st.caption("Configure GEMINI_API_KEY en los secrets de Streamlit Cloud.")
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        clear_chat_history()
        st.rerun()
    st.caption(f"Mensajes en sesión: {len(st.session_state.chat_messages)}")

st.markdown("""
<div class="chat-header">
    <h1 style="margin:0;">🤖 Asistente Solar</h1>
    <p style="margin:0.35rem 0 0; opacity:0.95;">Energía solar para comunidades rurales — instalación, costos, mantenimiento y sistemas aislados</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.chat_messages:
    st.markdown("""
    <div class="welcome-card">
        <div style="font-size:3rem;">☀️</div>
        <h2 style="color:#E65100; margin:0.4rem 0;">Bienvenido</h2>
        <p style="color:#616161;">Pregunta lo que necesites o elige una sugerencia abajo.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 Preguntas sugeridas")
    suggestions = get_suggested_questions()
    col_x, col_y = st.columns(2)
    for idx, question in enumerate(suggestions):
        target = col_x if idx % 2 == 0 else col_y
        with target:
            if st.button(question, key=f"sq_{idx}", use_container_width=True):
                with st.spinner("Escribiendo respuesta…"):
                    answer = generate_response(question, gemini_key)
                st.session_state.chat_messages.append({"role": "user", "content": question})
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                st.rerun()

for message in st.session_state.chat_messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🌞"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

prompt = st.chat_input("Escribe tu pregunta sobre energía solar…")

if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.spinner("Escribiendo…"):
        reply = generate_response(prompt, gemini_key)
    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
    st.rerun()
