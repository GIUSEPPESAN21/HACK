"""
Alerta Verde — Lógica del Asistente Solar (chatbot pedagógico).

Responsabilidades:
    - Configurar Gemini con instrucciones de sistema para contexto rural y fotovoltaico.
    - Gestionar historial en ``st.session_state`` y respuestas con manejo de errores.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import streamlit as st

from src.config import GEMINI_MODEL_NAME, MAX_CHAT_MESSAGE_CHARS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT: str = """Eres el **Asistente Solar**, experto en energía solar fotovoltaica para comunidades rurales \
y zonas no interconectadas a la red eléctrica (off-grid y sistemas aislados).

Tu misión es educar con claridad sobre:
- Instalación segura de sistemas fotovoltaicos residenciales y comunitarios.
- Mantenimiento preventivo y diagnóstico de fallos comunes (inversor, baterías, cableado, sombras).
- Costos orientativos, ahorro energético, beneficios ambientales y sociales.
- Consideraciones específicas para zonas rurales: transporte de equipos, climas diversos, \
disponibilidad de técnicos y continuidad del servicio.

Estilo:
- Español neutro, cercano y respetuoso.
- Explica términos técnicos con ejemplos simples.
- Si falta información, pide datos concretos (consumo kWh/mes, ubicación aproximada, presupuesto).
- No inventes normativas exactas ni precios precisos sin aclarar que son orientativos.
- Prioriza seguridad eléctrica y recomendaciones verificables."""


@st.cache_resource(show_spinner=False)
def get_gemini_generative_model(api_key: str) -> Any:
    """Construye y cachea el modelo generativo Gemini con instrucciones de sistema.

    Args:
        api_key: Clave de API de Google AI Studio / Gemini.

    Returns:
        Instancia de ``GenerativeModel`` configurada.
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
    )
    return model


def get_gemini_api_key_from_secrets() -> Optional[str]:
    """Lee la clave Gemini desde ``st.secrets`` (Streamlit Cloud).

    Returns:
        Clave API o None si no está definida o hay error de contexto.
    """
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        return key if key else None
    except (AttributeError, KeyError, TypeError):
        return None


def initialize_chat_history() -> None:
    """Inicializa estructuras de chat en ``st.session_state`` si no existen."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "gemini_chat" not in st.session_state:
        st.session_state.gemini_chat = None


def get_chat_session(model: Any) -> Any:
    """Obtiene o crea la sesión de chat con historial nativo de Gemini.

    Args:
        model: Modelo generativo configurado.

    Returns:
        Chat session con ``send_message``.
    """
    if st.session_state.gemini_chat is None:
        st.session_state.gemini_chat = model.start_chat(history=[])
    return st.session_state.gemini_chat


def _sanitize_user_message(text: str) -> str:
    """Recorta espacios y limita longitud para evitar abusos o errores de API."""
    cleaned = (text or "").strip()
    if len(cleaned) > MAX_CHAT_MESSAGE_CHARS:
        return cleaned[:MAX_CHAT_MESSAGE_CHARS]
    return cleaned


def generate_response(user_message: str, api_key: Optional[str] = None) -> str:
    """Genera la respuesta del Asistente Solar para un mensaje de usuario.

    Args:
        user_message: Texto enviado por el usuario.
        api_key: Clave API; si es None se intenta leer desde ``st.secrets``.

    Returns:
        Respuesta del asistente o mensaje de fallback en caso de error o modo demo.
    """
    cleaned = _sanitize_user_message(user_message)
    if not cleaned:
        return "Escribe una pregunta concreta sobre energía solar para poder ayudarte."

    key = api_key if api_key is not None else get_gemini_api_key_from_secrets()

    if not key:
        return _demo_response(cleaned)

    try:
        model = get_gemini_generative_model(key)
        chat = get_chat_session(model)
        response = chat.send_message(cleaned)
        text = getattr(response, "text", None)
        if text:
            return text
        return (
            "No recibí una respuesta textual del modelo. Intente reformular la pregunta "
            "o verifique la cuota de la API."
        )

    except Exception as exc:
        logger.warning("Error en generate_response (Gemini): %s", exc)
        return (
            "No pudimos completar la respuesta en este momento. "
            "Puede reintentar en unos segundos. Si el problema continúa, revise la cuota de la API."
        )


def clear_chat_history() -> None:
    """Reinicia mensajes y sesión de chat."""
    st.session_state.chat_messages = []
    st.session_state.gemini_chat = None


def get_suggested_questions() -> List[str]:
    """Devuelve preguntas frecuentes para botones de inicio rápido.

    Returns:
        Lista de strings en español.
    """
    return [
        "¿Qué es un panel solar y cómo funciona?",
        "¿Cuántos paneles necesito para una casa pequeña en zona rural?",
        "¿Cuánto cuesta instalar un sistema solar básico?",
        "¿Cómo mantener y limpiar los paneles sin dañarlos?",
        "¿Los paneles funcionan si hay nubes o lluvia?",
        "¿Qué diferencia hay entre sistema conectado a red y aislado (baterías)?",
        "¿Cuál es la vida útil típica de un panel solar?",
        "¿Cómo sé si un panel está fallando?",
    ]


def _demo_response(user_message: str) -> str:
    """Respuesta heurística local cuando no hay API key de Gemini.

    Args:
        user_message: Mensaje del usuario.

    Returns:
        Texto educativo breve en español.
    """
    msg_lower = user_message.lower()

    responses = {
        "panel": (
            "🌞 **Paneles solares (módulos fotovoltaicos)**\n\n"
            "Convierten la luz del sol en electricidad mediante celdas de silicio. "
            "En una vivienda rural típica suelen usarse varios paneles en serie (string) "
            "conectados a un inversor o a un regulador de carga si hay baterías.\n\n"
            "💡 *Configure GEMINI_API_KEY en los secrets de Streamlit para respuestas más personalizadas.*"
        ),
        "costo": (
            "💰 **Costos (orden de magnitud)**\n\n"
            "El precio final depende del país, logística rural y tipo de sistema. "
            "Suele evaluarse en **USD/Wp instalado** y trabajo de montaje eléctrico. "
            "Pida siempre cotizaciones comparables y verifique garantías de paneles e inversor.\n\n"
            "💡 *Con Gemini activo podré ayudarte a listar partidas a considerar.*"
        ),
        "manteni": (
            "🔧 **Mantenimiento**\n\n"
            "Limpieza periódica (agua suave, sin abrasivos), revisión visual de fisuras y "
            "conexiones, y monitoreo de producción esperada por temporada.\n\n"
            "💡 *Activa la API para listas de chequeo según tu clima y tipo de techo.*"
        ),
        "instala": (
            "🏠 **Instalación**\n\n"
            "Implica diseño eléctrico, estructura de soporte, cableado DC/AC y puesta a tierra. "
            "En zonas rurales sume traslados, seguros y disponibilidad de técnico certificado.\n\n"
            "⚠️ La parte en corriente alterna y conexión a red debe hacerla personal calificado.\n\n"
            "💡 *Con Gemini: guías paso a paso más adaptadas a tu caso.*"
        ),
    }

    for keyword, response in responses.items():
        if keyword in msg_lower:
            return response

    return (
        "🌞 **Asistente Solar — modo demo**\n\n"
        "Puedo orientarte sobre instalación, mantenimiento, costos y beneficios de la energía solar "
        "en contextos rurales. Para respuestas más completas, configura **GEMINI_API_KEY** "
        "en los secrets de Streamlit Cloud.\n\n"
        "Prueba preguntar por paneles, costos, mantenimiento o sistemas aislados."
    )
