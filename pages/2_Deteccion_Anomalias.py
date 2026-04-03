import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.ml_model import train_and_save_model, predict_anomaly, load_model
from src.data_processing import load_and_process_data

st.set_page_config(page_title="Detección de Anomalías", page_icon="🔍", layout="wide")

st.title("🔍 Detección de Anomalías con IA")
st.markdown("Utiliza nuestro modelo de Machine Learning (Random Forest) para predecir fallas en tus paneles solares.")

tab1, tab2 = st.tabs(["⚡ Predicción Rápida", "🧠 Entrenamiento del Modelo"])

with tab1:
    st.subheader("Simular Lectura de Sensor")
    st.markdown("Ingresa los valores actuales para evaluar el riesgo de falla.")
    
    col1, col2 = st.columns(2)
    with col1:
        temp_input = st.slider("Temperatura (°C)", min_value=10.0, max_value=60.0, value=30.0, step=0.1)
    with col2:
        hum_input = st.slider("Humedad (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        
    if st.button("Realizar Predicción", type="primary"):
        model = load_model()
        if model is None:
            st.error("⚠️ El modelo no está entrenado. Ve a la pestaña 'Entrenamiento' primero.")
        else:
            with st.spinner("Analizando patrones..."):
                proba = predict_anomaly(temp_input, hum_input)
                
                st.markdown("### Resultado del Análisis")
                if proba > 0.5:
                    st.error(f"🚨 **ALTO RIESGO DE ANOMALÍA** (Probabilidad: {proba:.2%})")
                    st.markdown("Se recomienda una inspección física inmediata del panel solar.")
                else:
                    st.success(f"✅ **SISTEMA ESTABLE** (Probabilidad de falla: {proba:.2%})")
                    st.markdown("Los parámetros están dentro de los rangos operativos normales.")

with tab2:
    st.subheader("Gestión del Motor de IA")
    st.markdown("Entrena el modelo utilizando los datos históricos generados (simulación de estrés).")
    
    df = load_and_process_data()
    
    if df.empty:
         st.warning("No se encontraron datos para entrenar.")
    else:
        st.info(f"Datos disponibles para entrenamiento: **{len(df):,} registros**")
        
        if st.button("Entrenar Modelo Ahora"):
            model = train_and_save_model(df)
            if model:
                # Mostrar importancia de las variables
                importances = model.feature_importances_
                df_imp = pd.DataFrame({'Variable': ['Temperatura', 'Humedad'], 'Importancia': importances})
                fig = px.bar(df_imp, x='Variable', y='Importancia', title="Importancia de Variables Predictoras", color='Variable')
                st.plotly_chart(fig, use_container_width=True)
