import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import json

# @st.cache_resource evita que Firebase tire error de "App ya inicializada" 
# y mantiene la conexión viva entre páginas.
@st.cache_resource
def init_firebase():
    """Inicializa la conexión a Firebase usando st.secrets."""
    try:
        # Verifica si ya existe una app inicializada para evitar errores
        if not firebase_admin._apps:
            # Lee el secreto configurado en Streamlit Cloud
            if "firebase" in st.secrets:
                cert = dict(st.secrets["firebase"])
                cred = credentials.Certificate(cert)
                firebase_admin.initialize_app(cred)
            else:
                st.error("⚠️ Faltan credenciales de Firebase en st.secrets.")
                return None
        return firestore.client()
    except Exception as e:
        st.error(f"Error conectando a Firebase: {e}")
        return None

def save_log_to_firestore(sensor_id, temp, hum, anomalia_proba):
    """Guarda un registro de monitoreo en la nube."""
    db = init_firebase()
    if db:
        try:
            doc_ref = db.collection('logs_sensores').document()
            doc_ref.set({
                'sensor_id': sensor_id,
                'temperatura': temp,
                'humedad': hum,
                'probabilidad_falla': anomalia_proba,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            st.error(f"Error guardando en BD: {e}")
