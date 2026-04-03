"""
Validación y normalización de lecturas de sensores (temperatura, humedad, identificadores).

Usado por predicción ML y persistencia en Firebase para mantener coherencia de datos.
"""

from __future__ import annotations

import re
from typing import Tuple

from src.config import HUMIDITY_MAX_PCT, HUMIDITY_MIN_PCT, TEMP_MAX_C, TEMP_MIN_C

# Identificadores tipo sensor01, sensor_02, etc.
_SENSOR_ID_PATTERN = re.compile(r"^[\w.\-]{1,64}$")


def clamp_temperature_c(value: float) -> float:
    """Acota la temperatura al rango operativo configurado.

    Args:
        value: Temperatura en °C.

    Returns:
        Valor acotado entre ``TEMP_MIN_C`` y ``TEMP_MAX_C``.
    """
    return float(max(TEMP_MIN_C, min(TEMP_MAX_C, value)))


def clamp_humidity_pct(value: float) -> float:
    """Acota la humedad relativa al intervalo [0, 100].

    Args:
        value: Humedad en %.

    Returns:
        Valor acotado.
    """
    return float(max(HUMIDITY_MIN_PCT, min(HUMIDITY_MAX_PCT, value)))


def normalize_sensor_id(raw: str) -> str:
    """Normaliza el identificador de sensor (trim, no vacío).

    Args:
        raw: Texto de entrada.

    Returns:
        Identificador saneado.

    Raises:
        ValueError: Si queda vacío tras el saneado.
    """
    s = str(raw).strip()
    if not s:
        raise ValueError("El identificador de sensor no puede estar vacío.")
    if not _SENSOR_ID_PATTERN.match(s):
        raise ValueError(
            "Sensor_ID inválido: use letras, números, guiones o puntos (máx. 64 caracteres)."
        )
    return s


def validate_reading_pair(temperatura_c: float, humedad_pct: float) -> Tuple[float, float]:
    """Valida y devuelve temperatura y humedad acotadas para inferencia o almacenamiento.

    Args:
        temperatura_c: Temperatura en °C.
        humedad_pct: Humedad en %.

    Returns:
        Tupla (temperatura acotada, humedad acotada).
    """
    return clamp_temperature_c(float(temperatura_c)), clamp_humidity_pct(float(humedad_pct))
