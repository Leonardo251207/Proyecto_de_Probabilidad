import streamlit as st

def mostrar_sidebar_info(tipos):
    """Uso de iconos de Material Design (:material/nombre:)"""
    st.sidebar.header("Resumen de Datos") # Título limpio
    
    # Usamos st.sidebar.status o columnas para que se vea más profesional
    st.sidebar.caption("DETECCIÓN DE COLUMNAS")
    st.sidebar.markdown(f":material/calendar_month: **Fechas:** {len(tipos['fechas'])}")
    st.sidebar.markdown(f":material/pin: **Numéricas:** {len(tipos['numericas'])}")
    st.sidebar.markdown(f":material/label: **Categóricas:** {len(tipos['categoricas'])}")

def mostrar_metricas_principales(res_bayes, evento_si):
    c1, c2, c3 = st.columns(3)
    # Agregamos 'help' para dar contexto profesional sin saturar la vista
    c1.metric("Evidencia Condicional", f"{res_bayes['prob_b_dado_a']:.2%}", 
              help=f"Probabilidad de la evidencia dado que ocurre {evento_si}")
    c2.metric("Probabilidad Posterior", f"{res_bayes['prob_posterior']:.2%}", 
              delta="Resultado Bayes", delta_color="normal")
    c3.metric("Probabilidad Global", f"{res_bayes['prob_b']:.2%}")

def alerta_insight(prob_a):
    if prob_a < 0.15:
        # Usamos iconos integrados en los mensajes
        st.warning("El evento es considerado de baja frecuencia (Raro).", icon=":material/priority_high:")
    else:
        st.success("El evento tiene una frecuencia dentro de los parámetros normales.", icon=":material/check_circle:")