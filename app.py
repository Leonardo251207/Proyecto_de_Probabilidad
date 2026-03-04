import streamlit as st
import pandas as pd

# Configuración visual
st.set_page_config(page_title="Proyecto Probabilidad", layout="wide")
st.title("📊 Analizador de Teorema de Bayes")

# 1. Carga de Archivo 
archivo = st.file_uploader("Sube tu archivo CSV de prueba", type=["csv"])

if archivo is not None:
    # Leer datos
    df = pd.read_csv(archivo)
    
    # 2. Detección automática de tipos 
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    col_fechas = df.select_dtypes(include=['datetime64']).columns.tolist()
    col_numericas = df.select_dtypes(include=['number']).columns.tolist()
    col_categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    st.sidebar.header("🔎 Columnas Detectadas")
    st.sidebar.write(f"📅 Fechas: {len(col_fechas)}")
    st.sidebar.write(f"🔢 Numéricas: {len(col_numericas)}")
    st.sidebar.write(f"🗂️ Categóricas: {len(col_categoricas)}")

    st.subheader("📋 Vista previa del Dataset")
    st.dataframe(df.head(10))

    # --- 3. Configuración del Evento (A) ---
    st.divider()
    st.header("🎯 Selección del Evento Objetivo (A)")
    
    col_interes = st.selectbox("Selecciona la columna del Evento (Variable Objetivo):", col_categoricas + col_numericas)
    
    # Inicializamos variables para evitar errores
    prob_a = 0
    evento_si = None

    if col_interes:
        opciones = df[col_interes].unique()
        evento_si = st.selectbox(f"¿Qué valor en '{col_interes}' representa el fallo/anomalía?", opciones)
        
        total = len(df)
        anomalos = len(df[df[col_interes] == evento_si])
        prob_a = anomalos / total

        c1, c2 = st.columns(2)
        with c1:
            st.metric(label=f"Probabilidad Base P({evento_si})", value=f"{prob_a:.2%}")
        with c2:
            if prob_a < 0.15:
                st.warning("⚠️ **Insight:** El evento es considerado 'Raro'.") [cite: 2]
            else:
                st.success("✅ **Insight:** El evento tiene una frecuencia normal.")

        # --- PASO 5: Probabilidades Condicionales (B) ---
        # Solo mostramos esto si ya se definió el Evento A
        st.divider()
        st.header("🔍 Análisis de Evidencia (B)")

        col_evidencia = st.selectbox("3. Selecciona una columna de Evidencia (B):", col_categoricas + col_numericas)

        if col_evidencia:
            if col_evidencia in col_numericas:
                umbral = st.number_input(f"Define el umbral para '{col_evidencia}':", value=float(df[col_evidencia].mean()))
                evidencia_mask = df[col_evidencia] > umbral
                texto_evidencia = f"{col_evidencia} > {umbral}"
            else:
                valor_ev = st.selectbox(f"¿Qué valor de '{col_evidencia}' usamos como evidencia?", df[col_evidencia].unique())
                evidencia_mask = df[col_evidencia] == valor_ev
                texto_evidencia = f"{col_evidencia} == {valor_ev}"

            # Cálculos de Bayes
            df_fallo = df[df[col_interes] == evento_si]
            
            # P(B|A)
            if len(df_fallo) > 0:
                if col_evidencia in col_numericas:
                    prob_evidencia_dado_fallo = (df_fallo[col_evidencia] > umbral).mean()
                else:
                    prob_evidencia_dado_fallo = (df_fallo[col_evidencia] == valor_ev).mean()
            else:
                prob_evidencia_dado_fallo = 0
            
            # P(B)
            prob_b = evidencia_mask.mean() 
            
            # Teorema de Bayes: P(A|B) = [P(B|A) * P(A)] / P(B)
            if prob_b > 0:
                prob_posterior = (prob_evidencia_dado_fallo * prob_a) / prob_b
            else:
                prob_posterior = 0

            # Resultados visuales
            c1, c2, c3 = st.columns(3)
            c1.metric(f"P(B|A): Evidencia dado {evento_si}", f"{prob_evidencia_dado_fallo:.2%}")
            c2.metric(f"P(A|B): {evento_si} dado Evidencia", f"{prob_posterior:.2%}")
            c3.metric("P(B): Prob. Evidencia", f"{prob_b:.2%}")
            
            st.success(f"**Resultado:** Si ocurre '{texto_evidencia}', la probabilidad de '{evento_si}' es de **{prob_posterior:.2%}**.")

            # --- PASO 6: Visualizaciones Obligatorias ---
            st.divider()
            st.header("📊 Visualización de Probabilidades")

            import plotly.graph_objects as go

            fig = go.Figure(data=[
                go.Bar(name='Probabilidad', 
                       x=['Base P(A)', 'Posterior P(A|B)'], 
                       y=[prob_a, prob_posterior],
                       marker_color=['#636EFA', '#EF553B'])
            ])
            fig.update_layout(title_text=f'Impacto de la Evidencia en la Probabilidad de {evento_si}',
                              yaxis=dict(tickformat=".2%"),
                              yaxis_range=[0, 1])
            
            st.plotly_chart(fig, use_container_width=True)

            if col_evidencia in col_numericas:
                st.subheader(f"Distribución de {col_evidencia}")
                st.bar_chart(df[col_evidencia].value_counts())

            # --- PASO 7: Clasificador y Matriz de Confusión (CORREGIDO) ---
            st.divider()
            st.header("🏁 Evaluación del Clasificador")

            from sklearn.metrics import confusion_matrix, accuracy_score
            import seaborn as sns
            import matplotlib.pyplot as plt

            # Generar predicciones para la sección de Metodología
            df['Prediccion'] = df[col_evidencia].apply(
                lambda x: evento_si if (
                    (col_evidencia in col_numericas and x > umbral) or 
                    (col_evidencia in col_categoricas and x == valor_ev)
                ) else "No Anomalía"
            )

            y_real = df[col_interes].apply(lambda x: evento_si if x == evento_si else "No Anomalía")
            y_pred = df['Prediccion']

            # Métricas de clasificación para la sección de Resultados
            acc = accuracy_score(y_real, y_pred)
            cm = confusion_matrix(y_real, y_pred, labels=[evento_si, "No Anomalía"])

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Exactitud", f"{acc:.2%}")
            
            fig_cm, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Pred: Si', 'Pred: No'], 
                        yticklabels=['Real: Si', 'Real: No'], ax=ax)
            ax.set_title("Matriz de Confusión")
            col_m2.pyplot(fig_cm)