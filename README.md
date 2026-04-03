# 🌿 Alerta Verde — Sistema Inteligente de Monitoreo Solar

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Sistema de monitoreo de paneles solares con detección de anomalías mediante Machine Learning y análisis contextual con Google Gemini AI. Incluye chatbot educativo para comunidades rurales.

---

## 🚀 Módulos

### 1. Alerta Verde — Detección de Anomalías
- **Dashboard en tiempo real** con gráficos interactivos (Plotly).
- **Modelo de ML** (RandomForestClassifier) para detectar fallos en sensores.
- **Análisis contextual** con Gemini AI: explicaciones y recomendaciones automáticas.
- **Firebase Firestore** para persistencia de datos en la nube.

### 2. Asistente Solar — Chatbot Educativo
- Chatbot conversacional powered by Google Gemini.
- Especializado en energía solar fotovoltaica para **comunidades rurales**.
- Temas: instalación, mantenimiento, costos, beneficios, dimensionamiento.

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Frontend | Streamlit |
| ML | scikit-learn (RandomForest) |
| IA | Google Gemini API |
| Base de datos | Firebase Firestore |
| Visualización | Plotly |
| Deploy | Streamlit Cloud |

---

## 📁 Estructura del Proyecto

```
├── app.py                         # Entry point – página de inicio
├── 01_Alerta Verde.csv            # Dataset de ejemplo (entrenamiento / demo)
├── requirements.txt               # Dependencias con versiones fijas
├── .env.example                   # Template de variables (referencia para secrets)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Constantes (rutas, colecciones, límites)
│   ├── validation.py             # Validación de sensores y lecturas
│   ├── data_processing.py        # Carga, limpieza, feature engineering
│   ├── ml_model.py               # Entrenamiento, evaluación, predicción
│   ├── firebase_client.py        # Conexión CRUD con Firestore
│   └── chatbot_logic.py          # Lógica del Asistente Solar
└── pages/
    ├── 1_Dashboard.py            # Visualización de sensores
    ├── 2_Deteccion_Anomalias.py  # Predicciones ML + Gemini
    └── 3_Asistente_Solar.py      # Chatbot pedagógico
```

---

## ⚡ Despliegue en Streamlit Cloud

### 1. Fork / Push a GitHub

```bash
git init
git add .
git commit -m "Initial commit: Alerta Verde v1.0"
git remote add origin https://github.com/TU_USUARIO/tu-repo.git
git push -u origin main
```

### 2. Configurar Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io).
2. Conecta tu repositorio de GitHub.
3. Configura el **Main file path** como `app.py`.
4. En **Settings > Secrets**, agrega las variables (formato TOML):

```toml
GEMINI_API_KEY = "tu_api_key_de_gemini"
FIREBASE_TYPE = "service_account"
FIREBASE_PROJECT_ID = "tu_project_id"
FIREBASE_PRIVATE_KEY_ID = "tu_private_key_id"
FIREBASE_PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL = "firebase-adminsdk@tu_project.iam.gserviceaccount.com"
FIREBASE_CLIENT_ID = "123456789"
```

### 3. ¡Listo! 🎉

La aplicación funciona **sin** configurar Firebase ni Gemini (modo demo).

---

## 🧪 Ejecución Local

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # macOS/Linux

# Instalar dependencias
pip install -r requirements.txt

# Crear secrets locales (opcional)
mkdir .streamlit
# Copia .env.example a .streamlit/secrets.toml en formato TOML

# Ejecutar
streamlit run app.py
```

---

## 📊 Dataset

El sistema espera un CSV con el siguiente esquema:

| Columna | Tipo | Descripción |
|---|---|---|
| `Fecha` | YYYY-MM-DD | Fecha del registro |
| `Sensor_ID` | string | ID del sensor |
| `Temperatura_C` | float | Temperatura en °C |
| `Humedad_%` | float | Humedad relativa |
| `Anomalia` | int (0/1) | 0: normal, 1: fallo |
| `Usuario` | string | Usuario asociado |

Si no se proporciona CSV, la app genera **datos demo automáticamente**.

---

## 📝 Licencia

MIT License — ver [LICENSE](LICENSE) para detalles.

---

<p align="center">
  🌿 <strong>Alerta Verde</strong> — Hackathon 2026<br>
  <em>Detección inteligente de anomalías en paneles solares</em>
</p>
