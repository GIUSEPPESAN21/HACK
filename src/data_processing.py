import pandas as pd
import numpy as np
import streamlit as st

# Usamos @st.cache_data para que esta función pesada solo se ejecute UNA VEZ.
# El resultado se guarda en memoria. Si el archivo cambia, se vuelve a ejecutar.
@st.cache_data(show_spinner="Cargando y procesando datos...")
def load_and_process_data(file_path="01_Alerta Verde.csv", target_rows=100000):
    try:
        # 1. Cargar datos base
        df_base = pd.read_csv(file_path)
        
        # 2. Generación de datos sintéticos (solo si es necesario para el Hackathon)
        current_rows = len(df_base)
        if current_rows < target_rows:
            rows_to_add = target_rows - current_rows
            
            # Generar variaciones aleatorias para simular sensores
            temp_std = df_base['Temperatura_C'].std() if not pd.isna(df_base['Temperatura_C'].std()) else 2.0
            hum_std = df_base['Humedad_%'].std() if not pd.isna(df_base['Humedad_%'].std()) else 5.0
            
            synthetic_temp = np.random.normal(df_base['Temperatura_C'].mean(), temp_std, rows_to_add)
            synthetic_hum = np.random.normal(df_base['Humedad_%'].mean(), hum_std, rows_to_add)
            
            # Asignar anomalías aleatorias (ej. 10% de probabilidad para equilibrar un poco)
            synthetic_anomalia = np.random.choice([0, 1], size=rows_to_add, p=[0.9, 0.1])
            
            df_synthetic = pd.DataFrame({
                'Fecha': pd.date_range(start='2024-03-01', periods=rows_to_add, freq='min'), # Fechas simuladas
                'Sensor_ID': np.random.choice(['sensor01', 'sensor02', 'sensor03'], rows_to_add),
                'Temperatura_C': synthetic_temp.round(1),
                'Humedad_%': synthetic_hum.round(1),
                'Anomalia': synthetic_anomalia,
                'Usuario': 'sistema_auto'
            })
            
            df_final = pd.concat([df_base, df_synthetic], ignore_index=True)
        else:
            df_final = df_base.copy()
            
        # 3. Limpieza y Normalización
        df_final['Fecha'] = pd.to_datetime(df_final['Fecha'])
        df_final.fillna(method='ffill', inplace=True) # Imputación básica
        
        return df_final
        
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
        return pd.DataFrame()
