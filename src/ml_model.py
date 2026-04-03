import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import streamlit as st

MODEL_PATH = "solar_model.pkl"

# @st.cache_resource es vital aquí. Evita inicializar/cargar el modelo en cada recarga.
@st.cache_resource(show_spinner="Cargando motor de IA...")
def load_model():
    """Carga el modelo desde el disco si existe."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def train_and_save_model(df):
    """Entrena el modelo SOLO si el usuario lo solicita explícitamente."""
    st.info("Iniciando entrenamiento del modelo con +100k registros. Esto puede tomar unos segundos...")
    
    # Preparar features (X) y target (y)
    X = df[['Temperatura_C', 'Humedad_%']]
    y = df['Anomalia']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Usar n_jobs=-1 para usar todos los núcleos del servidor en Streamlit Cloud
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluar
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Guardar en disco
    joblib.dump(model, MODEL_PATH)
    
    # Limpiar la caché para que la app cargue el nuevo modelo
    st.cache_resource.clear()
    
    st.success(f"Modelo entrenado y guardado con Precisión: {acc:.2%}")
    return model

def predict_anomaly(temperatura, humedad):
    """Realiza una predicción rápida."""
    model = load_model()
    if model:
        # Se espera un DataFrame o array 2D
        input_data = pd.DataFrame({'Temperatura_C': [temperatura], 'Humedad_%': [humedad]})
        proba = model.predict_proba(input_data)[0][1] # Probabilidad de anomalía (clase 1)
        return proba
    else:
        st.warning("El modelo aún no ha sido entrenado.")
        return None
