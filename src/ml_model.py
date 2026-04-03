"""
Alerta Verde — Módulo de Machine Learning.

Responsabilidades:
    - Entrenar un RandomForestClassifier con hiperparámetros afinados para el dominio.
    - Evaluar con métricas completas incluyendo ROC-AUC y matriz de confusión.
    - Serializar y cargar el modelo con joblib (cacheado en Streamlit).
    - Integrar Gemini para explicación contextual y recomendación de acción ante anomalías.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    DEFAULT_CSV_FILENAME,
    GEMINI_MODEL_NAME,
    MIN_ROWS_FOR_STRATIFIED_SPLIT,
    MODEL_PATH,
    TRAIN_RANDOM_STATE,
    TRAIN_TEST_SPLIT,
)
from src.data_processing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_features,
    clean_data,
    generate_demo_data,
    load_csv,
    prepare_ml_data,
)
from src.validation import validate_reading_pair

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Entrena un RandomForestClassifier con hiperparámetros optimizados para detección de fallos.

    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        random_state: Semilla para reproducibilidad.

    Returns:
        Modelo entrenado.
    """
    model = RandomForestClassifier(
        n_estimators=280,
        max_depth=24,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features="sqrt",
        max_samples=0.9,
        class_weight="balanced_subsample",
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evalúa el modelo y devuelve métricas detalladas incluyendo ROC-AUC y curva ROC.

    Args:
        model: Modelo entrenado.
        X_test: Features de prueba.
        y_test: Etiquetas reales de prueba.

    Returns:
        Diccionario con accuracy, precision, recall, f1, roc_auc, confusion_matrix,
        roc_curve (fpr, tpr), thresholds y classification_report.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        roc_auc = float("nan")

    fpr_arr, tpr_arr, thr_arr = roc_curve(y_test, y_proba)

    metrics: Dict[str, Any] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {
            "fpr": fpr_arr.tolist(),
            "tpr": tpr_arr.tolist(),
            "thresholds": thr_arr.tolist(),
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Anomalía"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def get_feature_importance(model: RandomForestClassifier) -> pd.DataFrame:
    """Obtiene la importancia relativa de cada feature.

    Args:
        model: Modelo entrenado.

    Returns:
        DataFrame con columnas Feature e Importance, ordenado descendente.
    """
    importance_df = (
        pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": model.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def save_model(model: RandomForestClassifier, path: Optional[Path] = None) -> Path:
    """Serializa el modelo a disco con joblib.

    Args:
        model: Modelo entrenado.
        path: Ruta de destino (por defecto ``MODEL_PATH``).

    Returns:
        Ruta donde se guardó el modelo.
    """
    save_path = path or MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return save_path


def delete_saved_model_file() -> None:
    """Elimina el archivo serializado del modelo si existe (forzar reentrenamiento)."""
    p = MODEL_PATH
    if p.exists():
        try:
            p.unlink()
        except OSError as exc:
            logger.warning("No se pudo eliminar el modelo guardado: %s", exc)


def load_model(path: Optional[Path] = None) -> Optional[RandomForestClassifier]:
    """Carga un modelo serializado desde disco.

    Args:
        path: Ruta al archivo joblib (por defecto ``MODEL_PATH``).

    Returns:
        Modelo cargado o ``None`` si el archivo no existe o está corrupto.
    """
    load_path = path or MODEL_PATH
    if not load_path.exists():
        return None
    try:
        return joblib.load(load_path)
    except Exception as exc:
        logger.error("Modelo corrupto o incompatible en %s: %s", load_path, exc)
        return None


def predict_single(
    model: RandomForestClassifier,
    temperatura: float,
    humedad: float,
    hora: int = 12,
    dia_semana: int = 2,
    mes: int = 6,
    temp_rolling: Optional[float] = None,
    humedad_rolling: Optional[float] = None,
    temp_humedad_ratio: Optional[float] = None,
    temp_zscore: Optional[float] = None,
    humedad_zscore: Optional[float] = None,
) -> Dict[str, Any]:
    """Realiza una predicción individual.

    Los valores de temperatura y humedad se acotan a rangos físicos razonables antes de inferir.

    Args:
        model: Modelo entrenado.
        temperatura: Temperatura en °C del sensor.
        humedad: Humedad % del sensor.
        hora: Hora del día (0-23).
        dia_semana: Día de la semana (0=lunes, 6=domingo).
        mes: Mes (1-12).
        temp_rolling: Media móvil de temperatura (se estima si es None).
        humedad_rolling: Media móvil de humedad (se estima si es None).
        temp_humedad_ratio: Ratio temp/humedad (se calcula si es None).
        temp_zscore: Z-score de temperatura (se estima si es None).
        humedad_zscore: Z-score de humedad (se estima si es None).

    Returns:
        Diccionario con prediction (0/1), probabilidades y datos de entrada.
    """
    temperatura, humedad = validate_reading_pair(temperatura, humedad)

    hora = int(max(0, min(23, hora)))
    dia_semana = int(max(0, min(6, dia_semana)))
    mes = int(max(1, min(12, mes)))

    if temp_rolling is None:
        temp_rolling = temperatura
    if humedad_rolling is None:
        humedad_rolling = humedad
    if temp_humedad_ratio is None:
        temp_humedad_ratio = round(temperatura / max(humedad, 0.01), 4)
    if temp_zscore is None:
        temp_zscore = round((temperatura - 35.0) / 5.0, 4)
    if humedad_zscore is None:
        humedad_zscore = round((humedad - 55.0) / 10.0, 4)

    features = pd.DataFrame([{
        "Temperatura_C": temperatura,
        "Humedad_%": humedad,
        "Hora": hora,
        "Dia_semana": dia_semana,
        "Mes": mes,
        "Temp_rolling_mean_5": temp_rolling,
        "Humedad_rolling_mean_5": humedad_rolling,
        "Temp_Humedad_ratio": temp_humedad_ratio,
        "Temp_zscore": temp_zscore,
        "Humedad_zscore": humedad_zscore,
    }])

    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]

    return {
        "prediction": prediction,
        "label": "Anomalía" if prediction == 1 else "Normal",
        "probability_normal": round(float(proba[0]), 4),
        "probability_anomaly": round(float(proba[1]), 4),
        "input_features": features.iloc[0].to_dict(),
    }


def predict_batch(
    model: RandomForestClassifier,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Realiza predicciones sobre un DataFrame completo.

    Args:
        model: Modelo entrenado.
        df: DataFrame con las columnas de FEATURE_COLUMNS.

    Returns:
        DataFrame original con columnas Prediccion y Probabilidad_Anomalia añadidas.

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas para predicción: {missing}")

    df = df.copy()
    X = df[FEATURE_COLUMNS]
    df["Prediccion"] = model.predict(X)
    probas = model.predict_proba(X)
    df["Probabilidad_Anomalia"] = np.round(probas[:, 1], 4)
    return df


def _fallback_anomaly_explanation() -> Dict[str, str]:
    """Mensaje predeterminado cuando Gemini no está disponible o falla.

    Returns:
        Diccionario con explicación y recomendación genéricas.
    """
    return {
        "tipo_anomalia": "Sobrecalentamiento / estrés térmico o condiciones ambientales extremas",
        "causa_raiz_probable": (
            "Combinación atípica de temperatura y humedad respecto al perfil histórico del sensor; "
            "posible suciedad acumulada, sombreado parcial o fallo de conexión."
        ),
        "recomendacion": (
            "Inspección visual del panel y cableado; verificar conexiones MC4; limpiar superficie; "
            "comparar con sensores vecinos; si persiste, sustituir módulo o revisar string en el inversor."
        ),
        "texto_completo": (
            "**Análisis (modo sin IA):** Se detectó una anomalía estadística. "
            "Configure GEMINI_API_KEY en los secrets de Streamlit para un diagnóstico contextual detallado."
        ),
    }


def explain_anomaly_with_gemini(
    prediction_result: Dict[str, Any],
    gemini_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """Genera explicación contextual y recomendación usando Gemini para lecturas anómalas.

    Args:
        prediction_result: Resultado de ``predict_single``.
        gemini_api_key: Clave API de Gemini; si falta, se usa fallback.

    Returns:
        Diccionario con claves: tipo_anomalia, causa_raiz_probable, recomendacion, texto_completo.
    """
    if prediction_result["prediction"] == 0:
        return {
            "tipo_anomalia": "N/A",
            "causa_raiz_probable": "Operación nominal.",
            "recomendacion": "Continuar monitoreo rutinario.",
            "texto_completo": (
                "Los valores del sensor son coherentes con operación normal. "
                "No se requiere intervención inmediata."
            ),
        }

    if not gemini_api_key:
        return _fallback_anomaly_explanation()

    try:
        import google.generativeai as genai

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        features = prediction_result["input_features"]
        prob = prediction_result["probability_anomaly"]

        prompt = f"""Eres un ingeniero experto en plantas fotovoltaicas y monitoreo IoT de sensores.

Se detectó una ANOMALÍA en un sensor con estos datos:
- Temperatura: {features.get('Temperatura_C', 'N/A')} °C
- Humedad: {features.get('Humedad_%', 'N/A')} %
- Hora: {features.get('Hora', 'N/A')}
- Día semana (0=lun): {features.get('Dia_semana', 'N/A')}
- Mes: {features.get('Mes', 'N/A')}
- Probabilidad modelo de anomalía: {prob * 100:.1f}%

Responde en español con EXACTAMENTE estas secciones y encabezados:

### Tipo de anomalía probable
(una línea)

### Causa raíz posible
(2-4 frases)

### Recomendación de acción para el técnico
(lista numerada de 3 pasos concretos y verificables)

### Urgencia
(Baja / Media / Alta / Crítica — una palabra)"""

        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        if not text:
            fb = _fallback_anomaly_explanation()
            fb["texto_completo"] = fb["texto_completo"] + "\n\n(Respuesta vacía del modelo.)"
            return fb

        return {
            "tipo_anomalia": "Ver sección en texto completo",
            "causa_raiz_probable": "Ver sección en texto completo",
            "recomendacion": "Ver sección en texto completo",
            "texto_completo": text,
        }

    except Exception as exc:
        logger.warning("Fallo al llamar a Gemini para explicación de anomalía: %s", exc)
        return _fallback_anomaly_explanation()


def _load_training_dataframe() -> pd.DataFrame:
    """Obtiene datos de entrenamiento desde CSV empaquetado o datos sintéticos.

    Returns:
        DataFrame en esquema crudo del proyecto.
    """
    try:
        return load_csv(DEFAULT_CSV_FILENAME)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.info("Usando datos sintéticos (CSV no disponible o inválido): %s", exc)
        return generate_demo_data(n_rows=6_000, seed=42)


@st.cache_resource(show_spinner="Cargando modelo de detección…")
def get_trained_model_and_metrics() -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Entrena o carga el modelo Random Forest y devuelve métricas sobre holdout interno.

    Returns:
        Tupla (modelo, métricas de evaluación).
    """
    df_raw = _load_training_dataframe()
    if len(df_raw) < MIN_ROWS_FOR_STRATIFIED_SPLIT:
        logger.info(
            "Dataset pequeño (%s filas); usando muestra sintética para entrenamiento estable.",
            len(df_raw),
        )
        df_raw = generate_demo_data(n_rows=6_000, seed=42)

    df_clean = clean_data(df_raw)
    df_feat = add_features(df_clean)

    X_train, X_test, y_train, y_test = prepare_ml_data(
        df_feat,
        test_size=TRAIN_TEST_SPLIT,
        random_state=TRAIN_RANDOM_STATE,
    )

    loaded = load_model()
    if loaded is not None:
        model = loaded
    else:
        model = train_model(X_train, y_train, random_state=TRAIN_RANDOM_STATE)
        save_model(model)

    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics
