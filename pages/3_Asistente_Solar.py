import streamlit as st
from src.chatbot_logic import get_solar_assistant_response

st.set_page_config(page_title="Asistente Solar SAVA", page_icon="💬", layout="centered")

st.title("💬 Asistente Solar SAVA")
st.markdown("¡Hola! Soy tu experto virtual en paneles solares. Pregúntame sobre mantenimiento, instalación o dudas generales.")

# Inicializar historial de chat en session_state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! ¿En qué te puedo ayudar hoy con tu sistema de energía solar?"}
    ]

# Mostrar el historial de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar entrada del usuario
if prompt := st.chat_input("Escribe tu pregunta aquí... (ej. ¿Cómo limpio mi panel?)"):
    # Agregar mensaje del usuario al historial y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Mostrar "escribiendo..." mientras se obtiene la respuesta
    with st.chat_message("assistant"):
        with st.spinner("SAVA está pensando..."):
            # Pasamos todo el historial menos el último mensaje (que ya lo pasamos en el prompt)
            history_for_api = st.session_state.messages[:-1] 
            response = get_solar_assistant_response(prompt, history_for_api)
            
            st.markdown(response)
            
    # Agregar respuesta del modelo al historial
    st.session_state.messages.append({"role": "assistant", "content": response})
