import streamlit as st
import pandas as pd
import base64
from sklearn.metrics import confusion_matrix, accuracy_score  # <-- Se añadió accuracy_score
from modulos.procesador import cargar_datos, optimizar_tipos
from modulos.estadistica import calcular_probabilidades_bayes, evaluar_modelo
from vistas.componentes import mostrar_sidebar_info, mostrar_metricas_principales, alerta_insight

from vistas.graficos import (
    grafico_comparativo_bayes, 
    grafico_matriz_confusion, 
    exportar_plotly_jpg, 
    exportar_matplotlib_jpg,
    grafico_histograma,
    grafico_temporal,
    grafico_evolucion_posterior
)

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Bayes Engine Pro", page_icon=":material/analytics:", layout="wide")

def cargar_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def añadir_video_fondo(video_file):
    try:
        with open(video_file, "rb") as f:
            data = f.read()
            bin_str = base64.b64encode(data).decode()
        
        st.markdown(f"""
            <style>
            #video-fondo {{
                position: fixed; right: 0; bottom: 0;
                min-width: 100%; min-height: 100%;
                z-index: -1; filter: opacity(0.15);
            }}
            </style>
            <video autoplay loop muted playsinline id="video-fondo">
                <source src="data:video/mp4;base64,{bin_str}" type="video/mp4">
            </video>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

cargar_css("asets/estilos.css") 
añadir_video_fondo("data/fondoo.mp4")

# --- INTERFAZ PRINCIPAL ---
st.title(":material/query_stats: Analizador Bayesiano")
    
with st.container(border=True):
    st.subheader(":material/upload_file: Carga de Datos")
    archivo = st.file_uploader("Arrastra tu CSV aquí", type=["csv"], label_visibility="collapsed")

if archivo:
    df_raw = cargar_datos(archivo)
    df_trabajo, tipos = optimizar_tipos(df_raw)
    
    mostrar_sidebar_info(df_trabajo)
    
    st.write(":material/table_rows: **Dataset cargado**")
    df_final = st.data_editor(df_trabajo, use_container_width=True, num_rows="dynamic", key="editor_principal")
    
    st.divider()
    
    # --- PARÁMETROS DEL MODELO ---
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown(":material/ads_click: **Evento Objetivo (A)**")
            col_interes = st.selectbox("Variable objetivo:", tipos['categoricas'] + tipos['numericas'])
            evento_si = st.selectbox("Condición de éxito:", df_final[col_interes].unique())
    with col2:
        with st.container(border=True):
            st.markdown(":material/experiment: **Evidencia (B)**")
            col_evidencia = st.selectbox("Variable de evidencia:", tipos['categoricas'] + tipos['numericas'])
            es_num = col_evidencia in tipos['numericas']
            valor_ev = st.number_input("Umbral:", value=0.0) if es_num else st.selectbox("Valor esperado:", df_final[col_evidencia].unique())

    # --- RESULTADOS ---
    res = calcular_probabilidades_bayes(df_final, col_interes, evento_si, col_evidencia, valor_ev, es_num)
    
    st.divider()
    alerta_insight(res['prob_a'], evento_si, df_final)
    mostrar_metricas_principales(res, evento_si, df_final)
    
    with st.container(border=True):
        st.subheader(":material/leaderboard: Probabilidad Posterior")
        c_bayes1, c_bayes2 = st.columns(2)
        
        with c_bayes1:
            st.markdown("**Contraste de Probabilidades**")
            fig_bayes = grafico_comparativo_bayes(res['prob_a'], res['prob_posterior'], evento_si)
            st.plotly_chart(fig_bayes, use_container_width=True)
            
            try:
                st.download_button(":material/download: Descargar Comparativa", 
                                 data=exportar_plotly_jpg(fig_bayes), 
                                 file_name="comparativa_bayes.jpg")
            except: 
                st.caption(":material/warning: Instale 'kaleido' para habilitar descargas.")
                
        with c_bayes2:
            st.markdown("**Evolución de Probabilidad**")
            fig_evolucion = grafico_evolucion_posterior(res['prob_a'], [f"{col_evidencia}={valor_ev}"], [res['prob_posterior']])
            st.plotly_chart(fig_evolucion, use_container_width=True)
            
            # --- NUEVO BOTÓN DE DESCARGA ---
            try:
                st.download_button(":material/download: Descargar Evolución", 
                                 data=exportar_plotly_jpg(fig_evolucion), 
                                 file_name="evolucion_bayes.jpg")
            except: 
                st.caption(":material/warning: Instale 'kaleido' para habilitar descargas.")

    # --- ANÁLISIS EXPLORATORIO ---
    st.divider()
    st.header(":material/troubleshoot: Exploración de Datos")

    if tipos['numericas']:
        st.subheader(":material/bar_chart: Distribución de Variables")
        col_hist = st.selectbox("Seleccionar métrica:", tipos['numericas'])
        fig_hist = grafico_histograma(df_final[col_hist].dropna(), col_hist)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # --- NUEVO BOTÓN DE DESCARGA ---
        try:
            st.download_button(":material/download: Descargar Histograma", 
                             data=exportar_plotly_jpg(fig_hist), 
                             file_name=f"histograma_{col_hist}.jpg")
        except: 
            pass
    
    st.divider()

    st.subheader(":material/timeline: Series Temporales")
    t_col1, t_col2 = st.columns(2)
    with t_col1: col_fecha = st.selectbox("Eje temporal (X):", df_final.columns)
    with t_col2: col_valor_temp = st.selectbox("Métrica (Y):", tipos['numericas'] if tipos['numericas'] else df_final.columns)
    
    try:
        df_temp = df_final.sort_values(by=col_fecha)
        fig_temp = grafico_temporal(df_temp[col_fecha], df_temp[col_valor_temp], col_valor_temp)
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # --- NUEVO BOTÓN DE DESCARGA ---
        try:
            st.download_button(":material/download: Descargar Temporal", 
                             data=exportar_plotly_jpg(fig_temp), 
                             file_name=f"temporal_{col_valor_temp}.jpg")
        except:
            pass
            
    except:
        st.info(":material/info: Selecciona una columna con formato fecha para graficar.")

    st.divider()

    # --- 4.3 MATRIZ DE CONFUSIÓN Y VALIDACIÓN ---
    st.markdown("#### :material/grid_view: Evaluación del Modelo") 
    st.write("Esta sección valida qué tan efectivas fueron las predicciones basadas en la evidencia.") 
    
    # 1. Copia limpia y eliminación de nulos
    df_eval = df_final[[col_interes, col_evidencia]].dropna().copy()
    
    if not df_eval.empty:
        # 2. Conversión ESTRICTA a enteros y FORZADO a 1D (.ravel())
        y_real = (df_eval[col_interes] == evento_si).astype(int).values.ravel()
        
        if es_num:
            y_pred = (df_eval[col_evidencia] >= valor_ev).astype(int).values.ravel()
        else:
            y_pred = (df_eval[col_evidencia] == valor_ev).astype(int).values.ravel()
        
        try:
            # --- NUEVO: CÁLCULO DE EXACTITUD (ACCURACY) ---
            exactitud = accuracy_score(y_real, y_pred)
            
            # Mostramos el porcentaje en un panel destacado antes de la matriz
            with st.container(border=True):
                st.metric(
                    label="Exactitud de Validación (Accuracy)", 
                    value=f"{exactitud:.1%}",
                    help="Porcentaje de casos donde la presencia/ausencia de la evidencia predijo correctamente el evento objetivo."
                )
            
           

           
            cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
            
            # 4. Aseguramos que las etiquetas sean texto puro
            etiq_neg = f"No {evento_si}"
            etiq_pos = str(evento_si)
            
            fig_matriz = grafico_matriz_confusion(cm, [etiq_neg, etiq_pos])
            st.pyplot(fig_matriz)
            
            st.download_button(
                label=":material/download: Descargar Matriz (JPG)",
                data=exportar_matplotlib_jpg(fig_matriz),
                file_name="matriz_confusion.jpg",
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f":material/warning: No se pudo generar la matriz: {e}")
    else:
        st.warning(":material/info: No hay datos suficientes para generar la validación.")

else:
    st.info(":material/upload_file: Por favor, carga un archivo CSV para comenzar el análisis.")