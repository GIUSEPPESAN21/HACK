"""
🔍 Detección de anomalías — Random Forest, métricas y explicación Gemini.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_processing import (
    add_features,
    clean_data,
    default_csv_path,
    generate_demo_data,
    load_csv,
    prepare_ml_data,
)
from src.firebase_client import is_firebase_available, register_anomaly, save_sensor_reading
from src.ml_model import (
    delete_saved_model_file,
    evaluate_model,
    explain_anomaly_with_gemini,
    get_feature_importance,
    get_trained_model_and_metrics,
    predict_batch,
    predict_single,
)
from src.chatbot_logic import get_gemini_api_key_from_secrets

st.set_page_config(
    page_title="🔍 Detección de Anomalías — Alerta Verde",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .anomaly-header {
        background: linear-gradient(135deg, #1A237E 0%, #3949AB 70%, #5C6BC0 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .result-normal {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border: 2px solid #43A047;
        border-radius: 14px;
        padding: 1.2rem;
    }
    .result-anomaly {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        border: 2px solid #E53935;
        border-radius: 14px;
        padding: 1.2rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1A237E;
        margin: 1.2rem 0 0.6rem;
        border-left: 4px solid #3949AB;
        padding-left: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def _load_featured_data(uploaded_file: object | None) -> pd.DataFrame:
    """Carga CSV y aplica el mismo pipeline de features que el entrenamiento."""
    if uploaded_file is not None:
        raw = load_csv(uploaded_file)
    else:
        try:
            raw = load_csv(default_csv_path())
        except (FileNotFoundError, ValueError, OSError):
            raw = generate_demo_data(n_rows=5000)
    return add_features(clean_data(raw))


st.markdown("""
<div class="anomaly-header">
    <h1 style="margin:0;">🔍 Detección de Anomalías</h1>
    <p style="margin:0.35rem 0 0; opacity:0.9;">
        Random Forest + métricas completas + explicación contextual (Gemini) y registro opcional en Firebase
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔍 Opciones")
    uploaded_train = st.file_uploader(
        "📁 CSV para análisis por lote",
        type=["csv"],
        key="det_csv",
        help="Mismo esquema que el dataset de referencia.",
    )
    if st.button("🔄 Forzar reentrenamiento (borra modelo guardado)", use_container_width=True):
        delete_saved_model_file()
        get_trained_model_and_metrics.clear()
        st.success("Cache limpiada. Recarga la página para reentrenar.")
    gemini_key = get_gemini_api_key_from_secrets()
    st.caption(f"Gemini: {'configurado' if gemini_key else 'modo demo'}")
    st.caption(f"Firebase: {'configurado' if is_firebase_available() else 'modo demo'}")

df_feat = _load_featured_data(uploaded_train)

model, _cached_metrics = get_trained_model_and_metrics()

tab1, tab2, tab3 = st.tabs(["📈 Métricas del modelo", "🎯 Predicción individual", "📋 Predicción por lote"])

with tab1:
    st.markdown('<div class="section-title">Rendimiento en holdout interno</div>', unsafe_allow_html=True)

    _, X_eval, _, y_eval = prepare_ml_data(
        df_feat,
        test_size=0.25,
        random_state=99,
    )
    live_metrics = evaluate_model(model, X_eval, y_eval)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{live_metrics['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{live_metrics['precision']*100:.2f}%")
    m3.metric("Recall", f"{live_metrics['recall']*100:.2f}%")
    m4.metric("F1", f"{live_metrics['f1']*100:.2f}%")
    roc = live_metrics.get("roc_auc")
    m5.metric("ROC-AUC", "—" if roc is None else f"{roc*100:.2f}%")

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("**Matriz de confusión**")
        cm = np.array(live_metrics["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicción", y="Real", color="Cantidad"),
            x=["Normal", "Anomalía"],
            y=["Normal", "Anomalía"],
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_cm.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_right:
        st.markdown("**Curva ROC**")
        roc_data = live_metrics.get("roc_curve") or {}
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color="#3949AB", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color="#9E9E9E", dash="dash")))
        fig_roc.update_layout(
            height=360,
            template="plotly_white",
            xaxis_title="Tasa falsos positivos",
            yaxis_title="Tasa verdaderos positivos",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("**Importancia de variables**")
    imp = get_feature_importance(model)
    fig_i = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
    fig_i.update_layout(height=420, template="plotly_white", yaxis=dict(autorange="reversed"), showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_i, use_container_width=True)

    with st.expander("Reporte de clasificación detallado"):
        rep = pd.DataFrame(live_metrics["classification_report"]).transpose()
        st.dataframe(rep.style.format("{:.4f}", na_rep="—"), use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Formulario de lectura</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        temperatura = st.number_input("Temperatura (°C)", -20.0, 100.0, 35.0, 0.5)
        humedad = st.number_input("Humedad (%)", 0.0, 100.0, 55.0, 0.5)
        sensor_id = st.selectbox("Sensor", sorted(df_feat["Sensor_ID"].unique().tolist()))
    with col_b:
        hora = st.slider("Hora", 0, 23, 12)
        dia = st.selectbox(
            "Día de la semana",
            list(range(7)),
            format_func=lambda x: ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"][x],
            index=2,
        )
        mes = st.selectbox("Mes", list(range(1, 13)), index=5)
        usuario = st.text_input("Usuario", value="tecnico1")

    guardar_fb = st.checkbox("Guardar lectura en Firebase tras analizar", value=False)

    if st.button("🔍 Analizar", type="primary", use_container_width=True):
        with st.spinner("Ejecutando modelo…"):
            result = predict_single(
                model=model,
                temperatura=temperatura,
                humedad=humedad,
                hora=hora,
                dia_semana=dia,
                mes=mes,
            )

        if result["prediction"] == 0:
            st.markdown(f"""
            <div class="result-normal">
                <h3 style="margin-top:0;">✅ Operación normal</h3>
                <p>Prob. normalidad: <b>{result['probability_normal']*100:.1f}%</b> · Prob. anomalía: <b>{result['probability_anomaly']*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            expl = explain_anomaly_with_gemini(result, gemini_key)
            st.info(expl["texto_completo"])
        else:
            st.markdown(f"""
            <div class="result-anomaly">
                <h3 style="margin-top:0;">🚨 Anomalía detectada</h3>
                <p>Prob. anomalía: <b>{result['probability_anomaly']*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("🤖 Generando explicación con Gemini…"):
                expl = explain_anomaly_with_gemini(result, gemini_key)

            st.markdown("### 📌 Análisis para el técnico")
            st.markdown(expl.get("texto_completo", ""))

            if is_firebase_available():
                if register_anomaly(
                    sensor_id=sensor_id,
                    temperatura=temperatura,
                    humedad=humedad,
                    probabilidad=float(result["probability_anomaly"]),
                    explicacion=str(expl.get("texto_completo", "")),
                    usuario=usuario,
                ):
                    st.success("Anomalía registrada en Firebase.")

        if guardar_fb and is_firebase_available():
            ok = save_sensor_reading(
                sensor_id=sensor_id,
                temperatura=temperatura,
                humedad=humedad,
                usuario=usuario,
                anomalia=int(result["prediction"]),
            )
            if ok:
                st.success("Lectura guardada en Firebase.")
            else:
                st.error("No se pudo guardar la lectura.")

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability_anomaly"] * 100,
            title={"text": "Probabilidad de anomalía"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#C62828" if result["prediction"] == 1 else "#2E7D32"},
            },
        ))
        fig_g.update_layout(height=280, template="plotly_white")
        st.plotly_chart(fig_g, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Predicción por lote</div>', unsafe_allow_html=True)
    fuente = st.radio("Fuente", ["Datos cargados en memoria", "Subir otro CSV"], horizontal=True)

    if fuente == "Subir otro CSV":
        batch_file = st.file_uploader("CSV", type=["csv"], key="batch_only")
        if batch_file is None:
            st.info("Seleccione un archivo.")
            st.stop()
        raw_b = load_csv(batch_file)
        df_b = add_features(clean_data(raw_b))
    else:
        df_b = df_feat.copy()

    n = st.slider("Filas a analizar", 50, min(5000, max(50, len(df_b))), min(1000, len(df_b)), step=50)
    sample = df_b.head(n)

    if st.button("🚀 Ejecutar lote", type="primary", use_container_width=True):
        with st.spinner("Prediciendo…"):
            out = predict_batch(model, sample)

        st.metric("Anomalías detectadas", int(out["Prediccion"].sum()))
        st.metric("Registros", len(out))

        fig_h = px.histogram(out, x="Probabilidad_Anomalia", nbins=40, color_discrete_sequence=["#3949AB"])
        fig_h.add_vline(x=0.5, line_dash="dash", line_color="red")
        fig_h.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_h, use_container_width=True)

        st.dataframe(
            out[
                [c for c in ["Fecha", "Sensor_ID", "Temperatura_C", "Humedad_%", "Prediccion", "Probabilidad_Anomalia", "Usuario"] if c in out.columns]
            ].head(200),
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Descargar resultados",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_lote.csv",
            mime="text/csv",
        )
