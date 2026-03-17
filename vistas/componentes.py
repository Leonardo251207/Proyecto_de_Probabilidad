import streamlit as st
import contextlib
from modulos.ia import generar_explicaciones_bayes, generar_insight_ia, generar_sugerencias_analisis

# --- SPINNER FUTURISTA PERSONALIZADO ---
@contextlib.contextmanager
def spinner_futurista(texto, contenedor=st):
    """Crea una animación de carga conectada a estilos.css"""
    placeholder = contenedor.empty()
    
    # Solo el HTML puro, las clases ya están en tu estilos.css
    html_spinner = f"""
    <div class="cyber-loader-container">
        <div class="cyber-spinner"></div>
        <div class="cyber-text">{texto}</div>
    </div>
    """
    
    # Mostramos la animación
    placeholder.markdown(html_spinner, unsafe_allow_html=True)
    
    try:
        # Ejecutamos la llamada a la IA
        yield
    finally:
        # Borramos la animación al terminar
        placeholder.empty()

# --- COMPONENTES DE LA INTERFAZ ---

def mostrar_sidebar_info(df):
    st.sidebar.header(":material/psychology: Ideas de Análisis")
    st.sidebar.caption("SUGERENCIAS CON IA")
    
    muestra_contexto = df.head(3).to_string()
    
    with spinner_futurista("PROCESANDO IDEAS...", contenedor=st.sidebar):
        sugerencias = generar_sugerencias_analisis(muestra_contexto)
    
    for sug in sugerencias:
        st.sidebar.info(f":material/lightbulb: {sug}")

def mostrar_metricas_principales(res_bayes, evento_si, df):
    muestra_contexto = df.head(2).to_string()

    with spinner_futurista("CALCULANDO MODELO PROBABILÍSTICO..."):
        explicaciones = generar_explicaciones_bayes(
            res_bayes['prob_b_dado_a'], 
            res_bayes['prob_posterior'], 
            res_bayes['prob_b'], 
            evento_si,
            muestra_contexto
        )

    c1, c2, c3 = st.columns(3)
    
    c1.metric("Evidencia Condicional", f"{res_bayes['prob_b_dado_a']:.2%}", 
              help=explicaciones.get("evidencia"))
              
    c2.metric("Probabilidad Posterior", f"{res_bayes['prob_posterior']:.2%}", 
              delta="Resultado Bayes", delta_color="normal", 
              help=explicaciones.get("posterior"))
              
    c3.metric("Probabilidad Global", f"{res_bayes['prob_b']:.2%}", 
              help=explicaciones.get("global"))
    
def alerta_insight(prob_a, evento_si, df):
    muestra_contexto = df.head(2).to_string()
    
    with spinner_futurista("SINTETIZANDO INSIGHTS..."):
        mensaje_ia = generar_insight_ia(prob_a, evento_si, muestra_contexto)
    
    if prob_a < 0.15:
        st.warning(mensaje_ia, icon=":material/priority_high:")
    else:
        st.success(mensaje_ia, icon=":material/check_circle:")