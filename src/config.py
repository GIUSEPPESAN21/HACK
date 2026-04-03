"""
Constantes de aplicación para Alerta Verde (rutas, nombres de colección, límites físicos).

Centralizar aquí evita cadenas mágicas dispersas y facilita ajustes sin tocar la lógica de negocio.
"""

from __future__ import annotations

from pathlib import Path

# --- Dataset empaquetado (mismo nombre que en el repositorio público) ---
DEFAULT_CSV_FILENAME: str = "01_Alerta Verde.csv"

# --- Modelo persistido ---
MODEL_DIR: Path = Path("models")
MODEL_FILENAME: str = "alerta_verde_rf.joblib"
MODEL_PATH: Path = MODEL_DIR / MODEL_FILENAME
MODEL_VERSION_LABEL: str = "rf_v1"

# --- Entrenamiento ---
MIN_ROWS_FOR_STRATIFIED_SPLIT: int = 80
MIN_SAMPLES_PER_CLASS_FOR_STRATIFY: int = 2
TRAIN_TEST_SPLIT: float = 0.2
TRAIN_RANDOM_STATE: int = 42

# --- Gemini ---
GEMINI_MODEL_NAME: str = "gemini-1.5-flash"

# --- Firestore ---
COLLECTION_LECTURAS: str = "lecturas_sensores"
COLLECTION_ANOMALIAS: str = "anomalias_detectadas"

# --- Rangos físicos razonables para sensores ambientales en planta (validación suave) ---
TEMP_MIN_C: float = -40.0
TEMP_MAX_C: float = 125.0
HUMIDITY_MIN_PCT: float = 0.0
HUMIDITY_MAX_PCT: float = 100.0

# --- Chat ---
MAX_CHAT_MESSAGE_CHARS: int = 6_000
