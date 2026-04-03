"""
Alerta Verde — Módulo de procesamiento de datos.

Responsabilidades:
    - Cargar el dataset CSV de sensores solares con validación de esquema.
    - Limpiar valores nulos, duplicados, outliers y datos inconsistentes.
    - Generar features derivados para el modelo de ML y para inferencia en tiempo real.
    - Alinear lecturas en vivo (Firebase) con el mismo pipeline que el entrenamiento.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from src.config import DEFAULT_CSV_FILENAME

logger = logging.getLogger(__name__)


def generate_demo_data(n_rows: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame sintético que replica el esquema del dataset real.

    Args:
        n_rows: Cantidad de registros a generar.
        seed: Semilla para reproducibilidad.

    Returns:
        DataFrame con columnas: Fecha, Sensor_ID, Temperatura_C,
        Humedad_%, Anomalia, Usuario.

    Raises:
        ValueError: Si ``n_rows`` no es positivo.
    """
    if n_rows <= 0:
        raise ValueError("n_rows debe ser un entero positivo.")

    rng = np.random.RandomState(seed)

    fechas = pd.date_range(start="2024-01-01", periods=n_rows, freq="15min")
    sensores = [f"sensor{str(i).zfill(2)}" for i in range(1, 11)]
    usuarios = ["tecnico1", "tecnico2", "admin", "supervisor1", "operador1"]

    temperatura_normal = rng.normal(loc=35.0, scale=5.0, size=n_rows)
    humedad_normal = rng.normal(loc=55.0, scale=10.0, size=n_rows)

    anomalia = rng.choice([0, 1], size=n_rows, p=[0.92, 0.08])

    temperatura = np.where(
        anomalia == 1,
        rng.uniform(55.0, 80.0, size=n_rows),
        temperatura_normal,
    )
    humedad = np.where(
        anomalia == 1,
        rng.uniform(85.0, 100.0, size=n_rows),
        humedad_normal,
    )

    df = pd.DataFrame({
        "Fecha": fechas[:n_rows],
        "Sensor_ID": rng.choice(sensores, size=n_rows),
        "Temperatura_C": np.round(temperatura, 2),
        "Humedad_%": np.round(humedad, 2),
        "Anomalia": anomalia,
        "Usuario": rng.choice(usuarios, size=n_rows),
    })
    return df


def load_csv(path_or_buffer: str | io.IOBase) -> pd.DataFrame:
    """Carga un CSV con el esquema esperado de sensores.

    Args:
        path_or_buffer: Ruta al archivo CSV o buffer (UploadedFile de Streamlit).

    Returns:
        DataFrame crudo tal cual viene del CSV.

    Raises:
        ValueError: Si faltan columnas obligatorias, el archivo está vacío o no es legible.
    """
    try:
        df = pd.read_csv(path_or_buffer)
    except EmptyDataError as exc:
        raise ValueError("El CSV está vacío o no contiene filas de datos.") from exc
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"No se pudo leer el CSV: {exc}") from exc

    required_columns = {"Fecha", "Sensor_ID", "Temperatura_C", "Humedad_%", "Anomalia", "Usuario"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {missing}")

    if len(df) == 0:
        raise ValueError("El CSV no contiene registros después del encabezado.")

    return df


def _clip_outliers_iqr(
    series: pd.Series,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    """Recorta valores extremos usando percentiles robustos por columna.

    Args:
        series: Serie numérica.
        lower_q: Percentil inferior.
        upper_q: Percentil superior.

    Returns:
        Serie con valores acotados al rango [q_low, q_high].
    """
    low = series.quantile(lower_q)
    high = series.quantile(upper_q)
    if pd.isna(low) or pd.isna(high) or low >= high:
        return series
    return series.clip(lower=low, upper=high)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el DataFrame: duplicados, tipos, nulos, outliers suaves e inconsistencias.

    Args:
        df: DataFrame crudo.

    Returns:
        DataFrame limpio con tipos corregidos y valores acotados.
    """
    df = df.copy()

    df.drop_duplicates(inplace=True)

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    df["Temperatura_C"] = pd.to_numeric(df["Temperatura_C"], errors="coerce")
    df["Humedad_%"] = pd.to_numeric(df["Humedad_%"], errors="coerce")
    df["Anomalia"] = pd.to_numeric(df["Anomalia"], errors="coerce").fillna(0).astype(int)
    df["Anomalia"] = df["Anomalia"].clip(0, 1)

    df["Sensor_ID"] = df["Sensor_ID"].astype(str).str.strip()
    df["Usuario"] = df["Usuario"].astype(str).str.strip()

    df.dropna(subset=["Fecha"], inplace=True)

    for col in ["Temperatura_C", "Humedad_%"]:
        df[col] = df.groupby("Sensor_ID")[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col].fillna(df[col].median(), inplace=True)

    for col in ["Temperatura_C", "Humedad_%"]:
        df[col] = df.groupby("Sensor_ID", group_keys=False)[col].transform(
            _clip_outliers_iqr
        )

    med_temp = float(df["Temperatura_C"].median())
    df.loc[df["Temperatura_C"] < -50, "Temperatura_C"] = med_temp
    df.loc[df["Temperatura_C"] > 120, "Temperatura_C"] = med_temp
    df.loc[df["Humedad_%"] < 0, "Humedad_%"] = 0.0
    df.loc[df["Humedad_%"] > 100, "Humedad_%"] = 100.0

    df.reset_index(drop=True, inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features derivados útiles para el modelo de ML.

    Features generados:
        - Hora, Dia_semana, Mes: componentes temporales.
        - Temp_rolling_mean_5, Humedad_rolling_mean_5: medias móviles por sensor.
        - Temp_Humedad_ratio: ratio temperatura / humedad.
        - Temp_zscore, Humedad_zscore: z-score por sensor.

    Args:
        df: DataFrame limpio.

    Returns:
        DataFrame con features adicionales.
    """
    df = df.copy()
    df = df.sort_values(["Sensor_ID", "Fecha"]).reset_index(drop=True)

    df["Hora"] = df["Fecha"].dt.hour.astype(int)
    df["Dia_semana"] = df["Fecha"].dt.dayofweek.astype(int)
    df["Mes"] = df["Fecha"].dt.month.astype(int)

    for col, new_col in [
        ("Temperatura_C", "Temp_rolling_mean_5"),
        ("Humedad_%", "Humedad_rolling_mean_5"),
    ]:
        df[new_col] = (
            df.groupby("Sensor_ID")[col]
            .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        )

    df["Temp_Humedad_ratio"] = np.where(
        df["Humedad_%"] != 0,
        np.round(df["Temperatura_C"] / df["Humedad_%"], 4),
        0.0,
    )

    for col, new_col in [
        ("Temperatura_C", "Temp_zscore"),
        ("Humedad_%", "Humedad_zscore"),
    ]:
        group_mean = df.groupby("Sensor_ID")[col].transform("mean")
        group_std = df.groupby("Sensor_ID")[col].transform("std").replace(0, 1)
        df[new_col] = np.round((df[col] - group_mean) / group_std, 4)

    return df


FEATURE_COLUMNS: list[str] = [
    "Temperatura_C",
    "Humedad_%",
    "Hora",
    "Dia_semana",
    "Mes",
    "Temp_rolling_mean_5",
    "Humedad_rolling_mean_5",
    "Temp_Humedad_ratio",
    "Temp_zscore",
    "Humedad_zscore",
]

TARGET_COLUMN: str = "Anomalia"


def prepare_ml_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide el DataFrame en conjuntos de entrenamiento y prueba.

    Intenta estratificar por la etiqueta; si no es posible (clases insuficientes), hace un split simple.

    Args:
        df: DataFrame con features (salida de ``add_features``).
        test_size: Proporción del conjunto de prueba.
        random_state: Semilla para reproducibilidad.

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split

    from src.config import MIN_SAMPLES_PER_CLASS_FOR_STRATIFY

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    counts = y.value_counts()
    can_stratify = len(counts) > 1 and all(counts >= MIN_SAMPLES_PER_CLASS_FOR_STRATIFY)

    try:
        if can_stratify:
            return train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y,
            )
    except ValueError as exc:
        logger.warning("Stratify no aplicable (%s); usando split aleatorio.", exc)

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )


def get_full_pipeline(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Pipeline completo: limpiar → features → split.

    Args:
        df_raw: DataFrame crudo (del CSV o datos demo).

    Returns:
        Tupla (df_featured, X_train, X_test, y_train, y_test).
    """
    df_clean = clean_data(df_raw)
    df_feat = add_features(df_clean)
    X_train, X_test, y_train, y_test = prepare_ml_data(df_feat)
    return df_feat, X_train, X_test, y_train, y_test


def prepare_live_reading_row(
    fecha: pd.Timestamp,
    sensor_id: str,
    temperatura_c: float,
    humedad_pct: float,
    usuario: str,
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Construye una fila con el mismo pipeline de features para inferencia en vivo.

    Args:
        fecha: Marca temporal de la lectura.
        sensor_id: Identificador del sensor.
        temperatura_c: Temperatura en °C.
        humedad_pct: Humedad relativa en %.
        usuario: Usuario asociado.
        history_df: Historial reciente del mismo sensor (opcional) para rolling y z-score.

    Returns:
        DataFrame de una fila con ``FEATURE_COLUMNS`` listas para ``predict``.
    """
    base = pd.DataFrame([{
        "Fecha": pd.Timestamp(fecha),
        "Sensor_ID": str(sensor_id).strip(),
        "Temperatura_C": float(temperatura_c),
        "Humedad_%": float(humedad_pct),
        "Anomalia": 0,
        "Usuario": str(usuario).strip(),
    }])

    if history_df is not None and not history_df.empty:
        need_cols = {"Fecha", "Sensor_ID", "Temperatura_C", "Humedad_%", "Anomalia", "Usuario"}
        if need_cols.issubset(set(history_df.columns)):
            hist = clean_data(history_df.copy())
            combined = pd.concat([hist, base], ignore_index=True)
            combined = combined.sort_values("Fecha").drop_duplicates(
                subset=["Fecha", "Sensor_ID"], keep="last"
            )
            feat = add_features(combined)
            return feat.iloc[[-1]][FEATURE_COLUMNS].copy()

    return add_features(clean_data(base))[FEATURE_COLUMNS].copy()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convierte un valor a float de forma segura."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int01(value: Any) -> int:
    """Convierte a entero 0/1 para etiqueta de anomalía."""
    try:
        v = int(float(value))
        return 1 if v != 0 else 0
    except (TypeError, ValueError):
        return 0


def dataframe_from_firebase_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convierte lecturas de Firestore al esquema esperado por ``clean_data``.

    Args:
        records: Lista de dicts con claves compatibles (sensor_id, fecha, etc.).

    Returns:
        DataFrame con columnas del dataset estándar.
    """
    rows: list[dict[str, Any]] = []
    for r in records:
        raw_fecha = r.get("fecha") or r.get("created_at")
        fecha_ts = pd.to_datetime(raw_fecha, errors="coerce")
        rows.append({
            "Fecha": fecha_ts,
            "Sensor_ID": str(r.get("sensor_id", "sensor01")).strip() or "sensor01",
            "Temperatura_C": _safe_float(r.get("temperatura_c"), 0.0),
            "Humedad_%": _safe_float(r.get("humedad_pct"), 0.0),
            "Anomalia": _safe_int01(r.get("anomalia", 0)),
            "Usuario": str(r.get("usuario", "tecnico1")).strip() or "tecnico1",
        })
    return pd.DataFrame(rows)


def default_csv_path() -> str:
    """Ruta por defecto del CSV empaquetado (nombre fijado en el repo)."""
    return DEFAULT_CSV_FILENAME
