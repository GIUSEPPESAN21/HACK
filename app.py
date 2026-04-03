import streamlit as st
import logging

# Configuración inicial (debe ser la primera llamada de Streamlit)
st.set_page_config(
    page_title="SAVA | Sistema Alerta Verde Automático",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una interfaz más profesional
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)

def main():
    st.title("☀️ Bienvenidos a SAVA")
    st.subheader("Sistema de Alerta Verde Automático para Paneles Solares")
    
    st.markdown("""
    ### 🚀 Tu plataforma inteligente para la gestión solar
    
    Utiliza el menú lateral izquierdo para navegar por las diferentes herramientas:
    
    * **📊 Dashboard:** Monitoreo en tiempo real y análisis histórico de tus sensores.
    * **🔍 Detección de Anomalías:** Motor de IA para predecir fallas basado en temperatura y humedad.
    * **💬 Asistente Solar:** Chatbot educativo para resolver dudas sobre instalación y mantenimiento.
    
    ---
    *Desarrollado para la eficiencia y sostenibilidad energética.*
    """)
    
    # Imagen ilustrativa (opcional, si tienes una en tu repo)
    # st.image("ruta/a/tu/imagen.png", use_container_width=True)

if __name__ == "__main__":
    main()
