import streamlit as st

def mostrar_sidebar_info(tipos):
    """Dibuja el resumen de columnas en la barra lateral."""
    st.sidebar.header("🔎 Columnas Detectadas")
    st.sidebar.write(f"📅 Fechas: {len(tipos['fechas'])}")
    st.sidebar.write(f"🔢 Numéricas: {len(tipos['numericas'])}")
    st.sidebar.write(f"🗂️ Categóricas: {len(tipos['categoricas'])}")

def mostrar_metricas_principales(res_bayes, evento_si):
    """Muestra las 3 métricas clave de Bayes en columnas."""
    c1, c2, c3 = st.columns(3)
    c1.metric(f"P(B|A): Evidencia dado {evento_si}", f"{res_bayes['prob_b_dado_a']:.2%}")
    c2.metric(f"P(A|B): {evento_si} dado Evidencia", f"{res_bayes['prob_posterior']:.2%}")
    c3.metric("P(B): Prob. Evidencia", f"{res_bayes['prob_b']:.2%}")

def alerta_insight(prob_a):
    """Muestra un mensaje dependiendo de la probabilidad base."""
    if prob_a < 0.15:
        st.warning("⚠️ **Insight:** El evento es considerado 'Raro'.")
    else:
        st.success("✅ **Insight:** El evento tiene una frecuencia normal.")