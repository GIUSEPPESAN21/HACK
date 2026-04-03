"""
Alerta Verde — Cliente de Firebase Firestore.

Responsabilidades:
    - Inicializar Firestore con credenciales de servicio leídas desde ``st.secrets``.
    - CRUD: lecturas de sensores, historial paginado, registro de anomalías con metadatos.
    - Degradación elegante si no hay configuración (sin romper la UI).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.config import COLLECTION_ANOMALIAS, COLLECTION_LECTURAS, MODEL_VERSION_LABEL
from src.validation import normalize_sensor_id, validate_reading_pair

logger = logging.getLogger(__name__)


def _normalize_private_key(raw: str) -> str:
    """Convierte claves privadas con ``\\n`` literales en saltos de línea reales.

    Args:
        raw: Valor de FIREBASE_PRIVATE_KEY desde secrets.

    Returns:
        Clave privada en formato PEM válido.
    """
    if not raw:
        return raw
    if "\\n" in raw:
        return raw.replace("\\n", "\n")
    return raw


def _build_credentials_dict() -> Optional[Dict[str, Any]]:
    """Construye el diccionario de credenciales desde ``st.secrets``.

    Returns:
        Diccionario de cuenta de servicio o None si faltan claves obligatorias.
    """
    try:
        private_key = _normalize_private_key(str(st.secrets["FIREBASE_PRIVATE_KEY"]))
        creds: Dict[str, Any] = {
            "type": st.secrets["FIREBASE_TYPE"],
            "project_id": st.secrets["FIREBASE_PROJECT_ID"],
            "private_key_id": st.secrets["FIREBASE_PRIVATE_KEY_ID"],
            "private_key": private_key,
            "client_email": st.secrets["FIREBASE_CLIENT_EMAIL"],
            "client_id": st.secrets["FIREBASE_CLIENT_ID"],
            "auth_uri": st.secrets.get("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": st.secrets.get("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
            "auth_provider_x509_cert_url": st.secrets.get(
                "FIREBASE_AUTH_PROVIDER_X509_CERT_URL",
                "https://www.googleapis.com/oauth2/v1/certs",
            ),
            "client_x509_cert_url": st.secrets.get("FIREBASE_CLIENT_X509_CERT_URL", ""),
        }
        return creds
    except (KeyError, AttributeError, TypeError):
        return None


@st.cache_resource(show_spinner="Conectando con Firebase…")
def get_firestore_client() -> Optional[Any]:
    """Inicializa y cachea el cliente de Firestore.

    Returns:
        Instancia de ``firestore.Client`` o None si no hay configuración válida.
    """
    creds_dict = _build_credentials_dict()
    if creds_dict is None:
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)

        return firestore.client()

    except Exception as exc:
        st.session_state["_firebase_last_error"] = str(exc)
        logger.warning("No se pudo inicializar Firebase: %s", exc)
        return None


def get_last_firebase_error() -> Optional[str]:
    """Devuelve el último error de conexión a Firebase almacenado en sesión.

    Returns:
        Mensaje de error o None.
    """
    return st.session_state.get("_firebase_last_error")


def is_firebase_available() -> bool:
    """Indica si Firestore está operativo.

    Returns:
        True si existe cliente Firestore.
    """
    return get_firestore_client() is not None


def save_sensor_reading(
    sensor_id: str,
    temperatura: float,
    humedad: float,
    usuario: str,
    fecha: Optional[str] = None,
    anomalia: int = 0,
) -> bool:
    """Persiste una lectura de sensor en la colección de lecturas.

    Args:
        sensor_id: Identificador del sensor.
        temperatura: Temperatura en °C.
        humedad: Humedad relativa en %.
        usuario: Usuario asociado.
        fecha: ISO8601 opcional; si falta, se usa UTC actual.
        anomalia: 0 normal, 1 anomalía etiquetada.

    Returns:
        True si el documento se escribió correctamente.
    """
    db = get_firestore_client()
    if db is None:
        return False

    try:
        sid = normalize_sensor_id(sensor_id)
        temp_c, hum_pct = validate_reading_pair(temperatura, humedad)
        usuario_clean = str(usuario).strip() or "unknown"

        doc_data = {
            "sensor_id": sid,
            "temperatura_c": temp_c,
            "humedad_pct": hum_pct,
            "usuario": usuario_clean,
            "fecha": fecha or datetime.utcnow().isoformat(),
            "anomalia": 1 if int(anomalia) != 0 else 0,
            "created_at": datetime.utcnow().isoformat(),
        }
        db.collection(COLLECTION_LECTURAS).add(doc_data)
        return True

    except ValueError as exc:
        st.session_state["_firebase_last_error"] = str(exc)
        logger.warning("Validación al guardar lectura: %s", exc)
        return False
    except Exception as exc:
        st.session_state["_firebase_last_error"] = str(exc)
        logger.warning("Error al guardar lectura: %s", exc)
        return False


def save_batch_readings(readings: List[Dict[str, Any]]) -> int:
    """Guarda múltiples lecturas usando batches de Firestore (máx. 500 por commit).

    Args:
        readings: Lista de dicts con sensor_id, temperatura, humedad, usuario, fecha, anomalia.

    Returns:
        Número de lecturas confirmadas en batch.
    """
    db = get_firestore_client()
    if db is None:
        return 0

    saved = 0
    try:
        batch = db.batch()
        collection_ref = db.collection(COLLECTION_LECTURAS)
        ops_in_batch = 0

        for idx, reading in enumerate(readings):
            try:
                sid = normalize_sensor_id(str(reading.get("sensor_id", "unknown")))
                temp_c, hum_pct = validate_reading_pair(
                    float(reading.get("temperatura", 0.0)),
                    float(reading.get("humedad", 0.0)),
                )
            except (ValueError, TypeError) as exc:
                logger.warning("Lectura %s omitida por validación: %s", idx, exc)
                continue

            doc_ref = collection_ref.document()
            doc_data = {
                "sensor_id": sid,
                "temperatura_c": temp_c,
                "humedad_pct": hum_pct,
                "usuario": str(reading.get("usuario", "unknown")).strip() or "unknown",
                "fecha": reading.get("fecha", datetime.utcnow().isoformat()),
                "anomalia": 1 if int(reading.get("anomalia", 0)) != 0 else 0,
                "created_at": datetime.utcnow().isoformat(),
            }
            batch.set(doc_ref, doc_data)
            saved += 1
            ops_in_batch += 1

            if ops_in_batch >= 500:
                batch.commit()
                batch = db.batch()
                ops_in_batch = 0

        if ops_in_batch > 0:
            batch.commit()

        return saved

    except Exception as exc:
        st.session_state["_firebase_last_error"] = str(exc)
        logger.warning("Error en batch write: %s", exc)
        return saved


def get_sensor_history(
    sensor_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Recupera lecturas recientes (primera página).

    Args:
        sensor_id: Filtro por sensor o None para todos.
        limit: Tamaño de página (máximo razonable recomendado: 500).

    Returns:
        Lista de documentos como dicts con ``doc_id``.
    """
    return get_sensor_history_paginated(sensor_id=sensor_id, page_size=limit, cursor_id=None)[0]


def get_sensor_history_paginated(
    sensor_id: Optional[str],
    page_size: int,
    cursor_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Historial paginado ordenado por ``created_at`` descendente.

    Si se filtra por ``sensor_id``, se aplica sobre una ventana reciente obtenida de Firestore
    para evitar índices compuestos adicionales en proyectos nuevos.

    Args:
        sensor_id: Filtro opcional por sensor.
        page_size: Registros por página.
        cursor_id: ID del último documento de la página anterior (para ``start_after``).

    Returns:
        Tupla (resultados, siguiente_cursor_id o None si no hay más páginas).
    """
    db = get_firestore_client()
    if db is None:
        return [], None

    if page_size <= 0:
        return [], None

    try:
        col = db.collection(COLLECTION_LECTURAS)
        fetch_limit = page_size if not sensor_id else min(2000, max(page_size * 15, 200))
        if cursor_id:
            snap = col.document(cursor_id).get()
            if snap.exists:
                query = col.order_by("created_at", direction="DESCENDING").start_after(snap).limit(fetch_limit)
            else:
                query = col.order_by("created_at", direction="DESCENDING").limit(fetch_limit)
        else:
            query = col.order_by("created_at", direction="DESCENDING").limit(fetch_limit)

        docs = list(query.stream())
        results: List[Dict[str, Any]] = []
        for doc in docs:
            data = doc.to_dict() or {}
            data["doc_id"] = doc.id
            if sensor_id and data.get("sensor_id") != sensor_id:
                continue
            results.append(data)
            if len(results) >= page_size:
                break

        next_cursor = results[-1]["doc_id"] if len(results) == page_size else None
        return results, next_cursor

    except Exception as exc:
        st.session_state["_firebase_last_error"] = str(exc)
        logger.warning("Error en historial paginado: %s", exc)
        return [], None


def register_anomaly(
    sensor_id: str,
    temperatura: float,
    humedad: float,
    probabilidad: float,
    explicacion: str,
    usuario: str,
    modelo_version: str = MODEL_VERSION_LABEL,
) -> bool:
    """Registra una anomalía detectada con metadatos del modelo y texto explicativo.

    Args:
        sensor_id: Identificador del sensor.
        temperatura: Temperatura al momento de la detección.
        humedad: Humedad al momento de la detección.
        probabilidad: Probabilidad de anomalía del modelo.
        explicacion: Texto de explicación (p. ej. salida de Gemini o fallback).
        usuario: Usuario que ejecutó la predicción.
        modelo_version: Etiqueta de versión del clasificador.

    Returns:
        True si se persistió correctamente.
    """
    db = get_firestore_client()
    if db is None:
        return False

    try:
        sid = normalize_sensor_id(sensor_id)
        temp_c, hum_pct = validate_reading_pair(temperatura, humedad)
        prob = float(max(0.0, min(1.0, probabilidad)))
        explicacion_clean = (explicacion or "").strip() or "(sin texto)"
        usuario_clean = str(usuario).strip() or "unknown"

        doc_data = {
            "sensor_id": sid,
            "temperatura_c": temp_c,
            "humedad_pct": hum_pct,
            "probabilidad_anomalia": prob,
            "explicacion_gemini": explicacion_clean[:50_000],
            "usuario": usuario_clean,
            "modelo_version": modelo_version,
            "detected_at": datetime.utcnow().isoformat(),
        }
        db.collection(COLLECTION_ANOMALIAS).add(doc_data)
        return True

    except ValueError as exc:
        st.session_state["_firebase_last_error"] = str(exc)
        return False
    except Exception as exc:
        st.session_state["_firebase_last_error"] = str(exc)
        logger.warning("Error al registrar anomalía: %s", exc)
        return False


def get_anomaly_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Obtiene anomalías recientes.

    Args:
        limit: Máximo de documentos.

    Returns:
        Lista ordenada por fecha de detección descendente.
    """
    db = get_firestore_client()
    if db is None:
        return []

    try:
        query = (
            db.collection(COLLECTION_ANOMALIAS)
            .order_by("detected_at", direction="DESCENDING")
            .limit(max(1, limit))
        )
        results: List[Dict[str, Any]] = []
        for doc in query.stream():
            data = doc.to_dict() or {}
            data["doc_id"] = doc.id
            results.append(data)
        return results

    except Exception as exc:
        st.session_state["_firebase_last_error"] = str(exc)
        logger.warning("Error al leer anomalías: %s", exc)
        return []


def get_sensor_stats() -> Dict[str, Any]:
    """Estadísticas rápidas basadas en muestras acotadas de Firestore.

    Returns:
        Conteos aproximados y número de sensores distintos en la muestra.
    """
    db = get_firestore_client()
    if db is None:
        return {"total_lecturas": 0, "total_anomalias": 0, "sensores_activos": 0}

    try:
        lecturas = list(db.collection(COLLECTION_LECTURAS).limit(1000).stream())
        anomalias = list(db.collection(COLLECTION_ANOMALIAS).limit(1000).stream())

        sensores = {doc.to_dict().get("sensor_id", "unknown") for doc in lecturas}

        return {
            "total_lecturas": len(lecturas),
            "total_anomalias": len(anomalias),
            "sensores_activos": len(sensores),
        }

    except Exception as exc:
        logger.warning("Error en estadísticas Firebase: %s", exc)
        return {"total_lecturas": 0, "total_anomalias": 0, "sensores_activos": 0}
