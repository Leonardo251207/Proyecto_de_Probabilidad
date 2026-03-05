import streamlit as st
import pandas as pd
import base64
from modulos.procesador import cargar_datos, optimizar_tipos
from modulos.estadistica import calcular_probabilidades_bayes, evaluar_modelo
from vistas.componentes import mostrar_sidebar_info, mostrar_metricas_principales, alerta_insight
from vistas.graficos import grafico_comparativo_bayes, grafico_matriz_confusion

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Bayes Engine Pro", page_icon="🧬", layout="wide")

# --- FUNCIÓN PARA VIDEO DE FONDO ---
def añadir_video_fondo(video_file):
    with open(video_file, "rb") as f:
        data = f.read()
        bin_str = base64.b64encode(data).decode()
    
    st.markdown(f"""
        <style>
        #video-fondo {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%; 
            min-height: 100%;
            z-index: -1;
            filter: brightness(0.5); /* Ajusta la oscuridad del video */
            object-fit: cover;
        }}
        .stApp {{
            background-color: transparent;
        }}
        </style>
        <video autoplay loop muted playsinline id="video-fondo">
            <source src="data:video/mp4;base64,{bin_str}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

# --- APLICAR DISEÑO PREMIUM ---
def aplicar_diseno_premium():
    st.markdown("""
        <style>
        /* Estilos generales */
        .stApp { color: #F2F2F2; }
        [data-testid="stSidebar"] { background-color: rgba(13, 13, 13, 0.5); border-right: 1px solid #3A1F73; }
        
        /* Contenedores con efecto de cristal (Glassmorphism) */
        div[data-testid="stVerticalBlockBorderWrapper"] { 
            background-color: rgb(22, 22, 22, 0.6) !important; 
            backdrop-filter: blur(20px);
            border: 1px solid #3A1F73 !important; 
            border-radius: 12px; 
        }
        
        /* Botones y Métricas */
        .stButton>button { background-color: #643DF2 !important; color: white !important; border-radius: 8px; }
        [data-testid="stMetric"] { 
            background-color: rgb(26, 26, 26); padding: 20px; border-radius: 10px; border-top: 3px solid #643DF2; 
        }
        [data-testid="stMetricValue"] { color: #F27F3D !important; }
        
        /* Textos */
        h1, h2, h3 { color: #643DF2 !important; font-family: 'Inter', sans-serif; }
        .stMarkdown p { color: #B0B0B0; }

        /* Estilo personalizado para el Cargador de Archivos */
        [data-testid="stFileUploaderIcon"] { color: #643DF2 !important; }
        [data-testid="stFileUploaderFileName"] { color: #F27F3D !important; }
        </style>
    """, unsafe_allow_html=True)

# Ejecutar funciones de estilo
añadir_video_fondo("data/fondoo.mp4")
aplicar_diseno_premium()

# --- HEADER CON ICONO SVG (FIJO) ---
col_t1, col_t2 = st.columns([3, 1])

with col_t1:
    st.title(":material/dynamic_form: Analizador de Teorema de Bayes")
# --- 1. CARGA DE DATOS ---
with st.container(border=True):
    st.markdown("### :material/cloud_upload: Centro de Datos")
    archivo = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if archivo:
    df_raw = cargar_datos(archivo)
    df_trabajo, tipos = optimizar_tipos(df_raw)
    mostrar_sidebar_info(tipos)
    
    st.markdown("#### :material/database: Dataset Original")
    df_final = st.data_editor(df_trabajo, use_container_width=True, num_rows="dynamic", key="editor_principal")
    
    # --- 2. CONFIGURACIÓN ---
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown(":material/target: **Evento Objetivo (A)**")
            col_interes = st.selectbox("Variable Objetivo:", tipos['categoricas'] + tipos['numericas'])
            evento_si = st.selectbox("Valor de éxito:", df_final[col_interes].unique())
    with c2:
        with st.container(border=True):
            st.markdown(":material/search: **Evidencia (B)**")
            col_evidencia = st.selectbox("Variable de Evidencia:", tipos['categoricas'] + tipos['numericas'])
            es_num = col_evidencia in tipos['numericas']
            valor_ev = st.number_input("Umbral:", value=0.0) if es_num else st.selectbox("Valor:", df_final[col_evidencia].unique())

    # --- 3. CÁLCULOS Y DASHBOARD ---
    res = calcular_probabilidades_bayes(df_final, col_interes, evento_si, col_evidencia, valor_ev, es_num)
    
    st.markdown("---")
    alerta_insight(res['prob_a'])
    mostrar_metricas_principales(res, evento_si)
    
    with st.container(border=True):
        st.markdown("#### :material/stacked_line_chart: Cambio en la Probabilidad")
        fig_bayes = grafico_comparativo_bayes(res['prob_a'], res['prob_posterior'], evento_si)
        fig_bayes.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#F2F2F2")
        st.plotly_chart(fig_bayes, use_container_width=True)

    # --- 4. EVALUACIÓN ---
    st.markdown("### :material/verified: Precisión del Modelo")
    acc, cm = evaluar_modelo(df_final, col_interes, evento_si, col_evidencia, valor_ev, es_num)
    
    m_col1, m_col2 = st.columns([1, 2])
    m_col1.metric("Precisión del Modelo", f"{acc:.2%}")
    with m_col2:
        fig_cm = grafico_matriz_confusion(cm, [f"{evento_si}", "Otros"])
        st.pyplot(fig_cm)