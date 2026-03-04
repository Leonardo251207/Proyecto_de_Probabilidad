import streamlit as st
import pandas as pd

# 1. Importamos nuestras propias piezas
from modulos.procesador import cargar_datos, optimizar_tipos
from modulos.estadistica import calcular_probabilidades_bayes, evaluar_modelo
from vistas.componentes import mostrar_sidebar_info, mostrar_metricas_principales, alerta_insight
from vistas.graficos import grafico_comparativo_bayes, grafico_matriz_confusion

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Analizador de Bayes Pro", layout="wide")
st.title("📊 Analizador de Teorema de Bayes")

# --- 1. CARGA DE DATOS ---
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo:
    # Usamos el procesador
    df_raw = cargar_datos(archivo)
    df, tipos = optimizar_tipos(df_raw)
    
    # Usamos componente de UI para la sidebar
    mostrar_sidebar_info(tipos)
    
    st.subheader("📋 Vista previa del Dataset")
    st.dataframe(df.head(10))

    # --- 2. CONFIGURACIÓN DEL EVENTO (A) ---
    st.divider()
    col_interes = st.selectbox("Selecciona la Variable Objetivo (A):", tipos['categoricas'] + tipos['numericas'])
    
    opciones = df[col_interes].unique()
    evento_si = st.selectbox(f"¿Qué valor representa el acierto/fallo?", opciones)
    
    # --- 3. ANÁLISIS DE EVIDENCIA (B) ---
    st.divider()
    col_evidencia = st.selectbox("Selecciona la Evidencia (B):", tipos['categoricas'] + tipos['numericas'])
    
    es_num = col_evidencia in tipos['numericas']
    valor_ev = None

    if es_num:
        valor_ev = st.number_input(f"Define el umbral para '{col_evidencia}':", value=float(df[col_evidencia].mean()))
    else:
        valor_ev = st.selectbox(f"¿Qué valor de '{col_evidencia}' es la evidencia?", df[col_evidencia].unique())

    # --- 4. CÁLCULOS Y LÓGICA ---
    # Llamamos al motor estadístico
    res = calcular_probabilidades_bayes(df, col_interes, evento_si, col_evidencia, valor_ev, es_num)
    
    # --- 5. MOSTRAR RESULTADOS ---
    alerta_insight(res['prob_a'])
    mostrar_metricas_principales(res, evento_si)
    
    # Gráfico de Bayes
    fig_bayes = grafico_comparativo_bayes(res['prob_a'], res['prob_posterior'], evento_si)
    st.plotly_chart(fig_bayes, use_container_width=True)

    # --- 6. EVALUACIÓN ---
    st.divider()
    st.header("🏁 Evaluación del Clasificador")
    
    acc, cm = evaluar_modelo(df, col_interes, evento_si, col_evidencia, valor_ev, es_num)
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Exactitud (Accuracy)", f"{acc:.2%}")
    
    fig_cm = grafico_matriz_confusion(cm, [evento_si, "Otros"])
    col_m2.pyplot(fig_cm)