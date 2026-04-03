import streamlit as st
import time

# 1. Configuración de página
st.set_page_config(
    page_title="Asistente Solar AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inyección de CSS Avanzado para UI/UX
def inject_custom_css():
    st.markdown("""
        <style>
        /* Tipografía global y fondo */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ----- BARRA LATERAL (SIDEBAR) ----- */
        [data-testid="stSidebar"] {
            min-width: 260px !important;
            max-width: 260px !important;
            background-color: #f4f7f9;
            border-right: 1px solid #e1e8ed;
        }
        [data-testid="stSidebar"] * {
            font-size: 0.95rem;
        }
        .sidebar-title {
            font-weight: 700;
            color: #1e293b;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .sidebar-subtitle {
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 2rem;
        }
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            margin-right: 6px;
            box-shadow: 0 0 8px #10b981;
        }

        /* ----- INTERFAZ DE CHAT ----- */
        /* Animación de aparición suave */
        @keyframes fadeInSlideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        [data-testid="stChatMessage"] {
            animation: fadeInSlideUp 0.4s ease-out forwards;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.02);
            transition: transform 0.2s ease;
        }

        /* Burbuja del Asistente */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #3b82f6;
        }

        /* Burbuja del Usuario */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-right: 4px solid #0f172a;
        }

        /* Avatar styling */
        [data-testid="stChatMessageAvatar"] {
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Input de Chat flotante */
        [data-testid="stChatInput"] {
            border-radius: 24px !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01) !important;
            padding: 4px 8px !important;
            background-color: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(10px);
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }

        /* ----- ALERTAS DE ANOMALÍAS (UI) ----- */
        .anomaly-alert {
            display: flex;
            align-items: flex-start;
            padding: 1rem 1.25rem;
            border-radius: 8px;
            margin: 0.5rem 0 1rem 0;
            font-size: 0.95rem;
            line-height: 1.5;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-left-width: 5px;
            border-left-style: solid;
            animation: fadeInSlideUp 0.5s ease-out;
        }
        .anomaly-icon {
            font-size: 1.5rem;
            margin-right: 12px;
            line-height: 1;
        }
        .anomaly-content strong {
            display: block;
            margin-bottom: 4px;
            font-size: 1.05rem;
        }
        /* Severidad: Crítica */
        .anomaly-critical {
            background-color: #fef2f2;
            border-left-color: #ef4444;
            color: #7f1d1d;
        }
        /* Severidad: Advertencia */
        .anomaly-warning {
            background-color: #fffbeb;
            border-left-color: #f59e0b;
            color: #78350f;
        }
        /* Severidad: Info/Solucionado */
        .anomaly-info {
            background-color: #f0fdf4;
            border-left-color: #10b981;
            color: #14532d;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# 3. Utilidades de Interfaz (Renderizado de Anomalías)
def render_anomaly(severity: str, title: str, description: str):
    config = {
        "critical": {"class": "anomaly-critical", "icon": "🚨"},
        "warning": {"class": "anomaly-warning", "icon": "⚠️"},
        "info": {"class": "anomaly-info", "icon": "✅"}
    }
    cfg = config.get(severity, config["info"])
    
    html = f"""
    <div class="anomaly-alert {cfg['class']}">
        <div class="anomaly-icon">{cfg['icon']}</div>
        <div class="anomaly-content">
            <strong>{title}</strong>
            {description}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# 4. Diseño de la Barra Lateral (Sidebar)
with st.sidebar:
    st.markdown('<div class="sidebar-title">🤖 SAVA Asistente</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Motor de IA Generativa</div>', unsafe_allow_html=True)
    
    st.markdown('<div><span class="status-indicator"></span> Sistema En Línea</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("**Diagnósticos Activos**")
    st.metric(label="Anomalías Detectadas Hoy", value="2", delta="-1 vs Ayer", delta_color="inverse")
    st.metric(label="Eficiencia Promedio", value="94.2%", delta="0.5%")
    
    st.markdown("---")
    st.caption("⚙️ Configuración")
    verbosity = st.select_slider("Nivel de Explicación", options=["Sencillo", "Técnico", "Experto"], value="Sencillo")
    
    if st.button("🗑️ Limpiar Historial", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. Inicialización del Estado del Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu asistente de energía solar de SAVA. Estoy monitoreando los sensores en tiempo real. ¿En qué te puedo ayudar hoy?"}
    ]

# 6. Renderizado del Historial de Mensajes
st.markdown("### 💬 Centro de Diagnóstico Interactivo")
st.caption("Interactúa con la IA para interpretar datos de sensores o resolver dudas de mantenimiento.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Soporte para renderizar UI de anomalías si está guardada en el historial
        if "anomaly" in message:
            render_anomaly(
                message["anomaly"]["severity"], 
                message["anomaly"]["title"], 
                message["anomaly"]["description"]
            )

# 7. Manejo de la Entrada del Usuario
if prompt := st.chat_input("Escribe tu consulta o pide un reporte de sensores..."):
    # Agregar mensaje del usuario al estado
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostrar mensaje del usuario en la UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta del Asistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # --- Lógica Simulada de Respuesta y Detección de Anomalías ---
        # Aquí se integraría la llamada real a Gemini API (gemini_client.py)
        
        if "anomalia" in prompt.lower() or "fallo" in prompt.lower() or "revisar" in prompt.lower():
            full_response = "He analizado los datos recientes del inversor y del panel principal. He detectado un patrón inusual en el Sensor_02 (Temperatura elevada y caída de voltaje)."
            
            # Efecto máquina de escribir para el texto
            for chunk in full_response.split():
                message_placeholder.markdown(chunk + " ")
                time.sleep(0.05)
            message_placeholder.markdown(full_response)
            
            # Desplegar la UI de Anomalía
            time.sleep(0.3)
            render_anomaly(
                severity="critical",
                title="Sobrecalentamiento Detectado (Sensor_02)",
                description="La temperatura ha superado los 65°C de forma sostenida durante los últimos 15 minutos. Riesgo de degradación de celda o fallo en la ventilación."
            )
            
            # Guardar la anomalía en el historial
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "anomaly": {
                    "severity": "critical",
                    "title": "Sobrecalentamiento Detectado (Sensor_02)",
                    "description": "La temperatura ha superado los 65°C de forma sostenida durante los últimos 15 minutos. Riesgo de degradación de celda o fallo en la ventilación."
                }
            })
            
        else:
            full_response = f"Entiendo tu consulta. Como estamos en modo nivel '{verbosity}', te explico: Los paneles están operando dentro de los parámetros normales. La irradiación actual es óptima. Si notas alguna baja en la producción, avísame para correr un diagnóstico profundo."
            
            # Efecto máquina de escribir
            partial_text = ""
            for char in full_response:
                partial_text += char
                message_placeholder.markdown(partial_text + "▌")
                time.sleep(0.01)
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
