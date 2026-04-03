import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.express as px

# 1. Configuración de página
st.set_page_config(
    page_title="Detección ML | SAVA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inyección de CSS (Manteniendo la identidad visual corporativa)
def inject_ml_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0f172a;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .alert-danger {
            background-color: #fef2f2;
            color: #991b1b;
            border-left: 4px solid #ef4444;
            padding: 1rem;
            border-radius: 4px;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)

inject_ml_css()

# 3. Encabezado de la página
st.markdown("## 🧠 Motor de Inferencia y Machine Learning")
st.caption("Ejecuta modelos predictivos sobre datos de sensores para prevenir fallos en la infraestructura solar.")

# 4. Configuración del Modelo (Barra Lateral)
with st.sidebar:
    st.markdown("### ⚙️ Parámetros del Modelo")
    modelo_seleccionado = st.selectbox(
        "Seleccionar Algoritmo",
        ["Random Forest (Recomendado)", "XGBoost", "Regresión Logística"]
    )
    umbral_confianza = st.slider("Umbral de Sensibilidad (%)", min_value=50, max_value=99, value=85, step=1)
    
    st.markdown("---")
    st.info("💡 **Nota:** Ajustar la sensibilidad cambia qué tan estricto es el modelo para clasificar un comportamiento como anomalía.")

# 5. Pestañas de Navegación
tab1, tab2 = st.tabs(["📁 Análisis por Lotes (Subir CSV)", "📡 Conexión en Tiempo Real (API)"])

with tab1:
    st.markdown("### Carga de Datos de Sensores")
    uploaded_file = st.file_uploader("Sube tu archivo de lecturas (.csv)", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Archivo cargado correctamente.")
            
            # Mostrar vista previa de los datos crudos
            with st.expander("Vista previa de los datos cargados"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Botón de ejecución
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Ejecutar Análisis Predictivo", type="primary", use_container_width=True):
                
                # Simulación de carga e inferencia del modelo
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    time.sleep(0.01) # Simula el tiempo de procesamiento
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("Preprocesando datos y escalando características...")
                    elif i < 70:
                        status_text.text(f"Evaluando con {modelo_seleccionado}...")
                    else:
                        status_text.text("Generando reporte de probabilidades...")
                
                status_text.empty()
                progress_bar.empty()
                
                # --- Lógica de Inferencia Simulada (A ser reemplazada por src.ml_model) ---
                # Generamos predicciones sintéticas basadas en reglas lógicas para el hackathon
                df_results = df.copy()
                
                # Simulamos que el modelo es inteligente: Temperaturas altas o humedad extrema = anomalía
                if 'Temperatura_C' in df_results.columns and 'Humedad_%' in df_results.columns:
                    probabilidades = np.where(
                        (df_results['Temperatura_C'] > 28) | (df_results['Humedad_%'] > 65),
                        np.random.uniform(0.75, 0.99, len(df_results)), # Alta prob si pasa los umbrales
                        np.random.uniform(0.01, 0.40, len(df_results))  # Baja prob si está normal
                    )
                else:
                    probabilidades = np.random.uniform(0, 1, len(df_results))
                
                df_results['Probabilidad_Fallo'] = probabilidades
                # Clasificamos según el umbral seleccionado en la barra lateral
                df_results['Prediccion_ML'] = (df_results['Probabilidad_Fallo'] >= (umbral_confianza/100)).astype(int)
                # -------------------------------------------------------------------------

                # Resultados: KPIs
                st.markdown("### 📊 Resultados del Diagnóstico")
                total_analizados = len(df_results)
                anomalias_detectadas = df_results['Prediccion_ML'].sum()
                
                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><div class="metric-value">{total_analizados}</div><div class="metric-label">Registros Analizados</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #ef4444;">{anomalias_detectadas}</div><div class="metric-label">Anomalías Detectadas</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="metric-value">{modelo_seleccionado}</div><div class="metric-label">Modelo Utilizado</div></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

                if anomalias_detectadas > 0:
                    st.markdown('<div class="alert-danger">⚠️ <strong>Acción Requerida:</strong> Se han detectado patrones de falla. Se recomienda revisar los paneles resaltados en la tabla inferior y consultar con el Asistente Solar.</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ La infraestructura se encuentra operando en condiciones óptimas.")

                # Resultados: Gráfico y Tabla
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    fig_pie = px.pie(
                        names=['Normal', 'Anomalía'], 
                        values=[total_analizados - anomalias_detectadas, anomalias_detectadas],
                        hole=0.4,
                        color_discrete_sequence=['#10b981', '#ef4444']
                    )
                    fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_res2:
                    # Mostrar tabla formateada
                    # Resaltamos las filas donde la predicción es 1 (Anomalía) y formateamos la probabilidad
                    st.dataframe(
                        df_results.style.format({'Probabilidad_Fallo': "{:.2%}"})
                        .applymap(lambda x: 'background-color: #fee2e2; color: #991b1b' if x == 1 else '', subset=['Prediccion_ML']),
                        use_container_width=True,
                        height=300
                    )
                
                # Opción para descargar resultados
                csv_export = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Reporte de Predicciones (CSV)",
                    data=csv_export,
                    file_name='reporte_anomalias_sava.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")

with tab2:
    st.markdown("### 📡 Monitoreo en Tiempo Real (IoT)")
    st.info("Esta sección está reservada para la conexión directa mediante WebSockets o MQTT con el hardware IoT en campo.")
    st.image("https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", caption="Arquitectura IoT - SAVA Engineering", use_column_width=True)
    st.button("Configurar Credenciales MQTT", disabled=True)
