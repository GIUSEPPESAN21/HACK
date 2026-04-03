import google.generativeai as genai
import streamlit as st

# Usar el modelo actual estable
GEMINI_MODEL_NAME = "gemini-2.5-flash"

@st.cache_resource
def configure_gemini():
    """Configura la API de Gemini usando los secretos de Streamlit."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return True
    except KeyError:
        return False
    except Exception as e:
        st.error(f"Error configurando Gemini: {e}")
        return False

def get_solar_assistant_response(user_message, chat_history):
    """
    Envía el mensaje a Gemini con un system prompt educativo.
    chat_history: lista de diccionarios [{'role': 'user'/'assistant', 'content': '...'}, ...]
    """
    if not configure_gemini():
        return "⚠️ Error: Clave de API de Gemini no encontrada. Por favor, configúrala en st.secrets."

    system_prompt = """
    Eres SAVA, un asistente experto y amigable en energía solar, diseñado para ayudar a comunidades rurales.
    Tus respuestas deben ser:
    1. Sencillas y sin jerga técnica compleja. Usa analogías de la vida diaria (ej. el agua, las plantas).
    2. Prácticas y orientadas al mantenimiento preventivo y solución de problemas básicos.
    3. Empáticas y motivadoras.
    4. Breves, claras y estructuradas con viñetas si es necesario.
    """

    try:
        # Inicializar el modelo
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            system_instruction=system_prompt
        )
        
        # Formatear el historial para la API de Gemini
        formatted_history = []
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append({"role": role, "parts": [msg["content"]]})
            
        # Iniciar chat
        chat = model.start_chat(history=formatted_history)
        
        # Enviar mensaje y obtener respuesta
        response = chat.send_message(user_message)
        return response.text
        
    except Exception as e:
        return f"⚠️ Disculpa, tuve un problema al procesar tu solicitud. Detalle técnico: {e}"
