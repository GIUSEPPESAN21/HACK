import streamlit as st

# 1. Configuración de la página (DEBE ser el primer comando de Streamlit)
st.set_page_config(
    page_title="SAVA | Detección Solar AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inicialización de estado global (Crucial para Streamlit Cloud)
# Esto asegura que cuando el usuario vaya a otras páginas, las variables ya existan.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "modelo_entrenado" not in st.session_state:
    st.session_state.modelo_entrenado = False

# 3. Sección Hero / Encabezado
st.title("☀️ Plataforma Inteligente de Monitoreo Solar")
st.markdown("### Desarrollado por **SAVA Software for Engineering**")

st.write("""
Bienvenido a la plataforma integral de detección de anomalías para paneles solares. 
Nuestra solución, diseñada para este hackathon, combina el poder del **Internet de las Cosas (IoT)**, 
**Machine Learning** y la **Inteligencia Artificial Generativa (Gemini)** para ofrecer diagnósticos 
precisos y explicaciones accesibles para comunidades rurales y técnicos de campo.
""")

st.info("👈 **Comienza a explorar:** Utiliza el menú lateral izquierdo para navegar entre el Dashboard en tiempo real, el motor de Detección de Anomalías y nuestro Asistente Solar con IA.")

st.markdown("---")

# 4. Sección del Equipo (Código proporcionado por el usuario, optimizado)
st.subheader("👥 Nuestro Equipo Fundador")

# Usamos columnas asimétricas para darle más peso al texto que a la imagen
col1_founder, col2_founder = st.columns([1, 3])

with col1_founder:
    # Mostramos el logo desde GitHub
    st.image(
        "https://github.com/GIUSEPPESAN21/LOGO-SAVA/blob/main/LOGO%20COLIBRI.png?raw=true", 
        width=200, 
        caption="SAVA Engineering"
    )

with col2_founder:
    st.markdown("#### Joseph Javier Sánchez Acuña")
    st.markdown("**CEO - SAVA SOFTWARE FOR ENGINEERING**")
    st.write("""
    Líder visionario con una profunda experiencia en inteligencia artificial y desarrollo de software.
    Joseph es el cerebro detrás de la arquitectura tecnológica, impulsando la innovación
    y asegurando que nuestra tecnología se mantenga a la vanguardia para resolver problemas reales.
    """)
    st.markdown(
        """
        - 🔗 **LinkedIn:** [joseph-javier-sánchez-acuña](https://www.linkedin.com/in/joseph-javier-sánchez-acuña-150410275)
        - 💻 **GitHub:** [GIUSEPPESAN21](https://github.com/GIUSEPPESAN21)
        """
    )

st.markdown("---")

# 5. Sección de Cofundadores
st.markdown("##### Cofundadores y Junta Directiva")

# Usamos contenedores de color para resaltar los roles
c1_cof, c2_cof, c3_cof = st.columns(3)

with c1_cof:
    st.success("**Xammy Alexander Victoria Gonzalez**\n\n👔 *Director Comercial*")
with c2_cof:
    st.info("**Jaime Eduardo Aragon Campo**\n\n⚙️ *Director de Operaciones*")
with c3_cof:
    st.warning("**Joseph Javier Sanchez Acuña**\n\n🚀 *Director de Proyecto*")

# Pie de página opcional
st.markdown("<br><br><center><small>Hackathon 2024 | Construido con ❤️ usando Streamlit y Google Gemini</small></center>", unsafe_allow_html=True)
