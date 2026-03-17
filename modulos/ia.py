import google.generativeai as genai
import streamlit as st
import json
import time 

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- FUNCIÓN 1: TOOLTIPS DE BAYES ---
@st.cache_data(show_spinner=False, ttl=3600)
def generar_explicaciones_bayes(prob_b_dado_a, prob_posterior, prob_b, evento_si, df_muestra):
    time.sleep(1.5) # Pausa para no saturar la API
    try:
        modelo = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Actúa como un analista de datos experto pero muy amigable. 
        Contexto del dataset (primeras filas):
        {df_muestra}
        
        Basado en el tema de ese dataset, interpreta estos resultados del Teorema de Bayes:
        - Evento analizado: '{evento_si}'
        - Evidencia Condicional: {prob_b_dado_a:.2%}
        - Probabilidad Posterior: {prob_posterior:.2%}
        - Probabilidad Global: {prob_b:.2%}
        
        Escribe 3 frases muy breves y naturales (máximo 15 palabras cada una) que sirvan de "tooltip" de ayuda. 
        Usa la temática del dataset para dar los ejemplos. No uses fórmulas matemáticas complejas.
        
        Devuelve ÚNICAMENTE un JSON válido con estas claves exactas: "evidencia", "posterior", "global".
        """
        
        respuesta = modelo.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(respuesta.text)
        
    except Exception as e:
        # Si falla, limpiamos la caché para que el error no se quede pegado 1 hora
        st.cache_data.clear()
        st.toast(f"⚠️ Error IA (Métricas): {str(e)[:100]}")
        return {
            "evidencia": f"Frecuencia de la evidencia cuando ocurre {evento_si}.",
            "posterior": f"Probabilidad de {evento_si} considerando esta evidencia.",
            "global": "Qué tan común es esta evidencia en el dataset general."
        }

# --- FUNCIÓN 2: INSIGHTS DINÁMICOS ---
@st.cache_data(show_spinner=False, ttl=3600)
def generar_insight_ia(prob_a, evento_si, df_muestra):
    time.sleep(1.5) # Pausa para no saturar la API
    try:
        modelo = genai.GenerativeModel('gemini-2.5-flash')
        es_raro = "raro (baja frecuencia)" if prob_a < 0.15 else "normal (frecuencia esperada)"
        
        prompt = f"""
        Actúa como un analista de datos. Contexto del dataset:
        {df_muestra}
        
        El evento '{evento_si}' tiene una probabilidad base de {prob_a:.2%}. Estadísticamente, esto se considera {es_raro}.
        
        Escribe UNA sola frase amigable y directa (máximo 20 palabras) dando una conclusión sobre esta frecuencia, usando la temática del dataset. No uses formato Markdown ni comillas.
        """
        
        respuesta = modelo.generate_content(prompt)
        return respuesta.text.strip()
        
    except Exception as e:
        st.cache_data.clear()
        st.toast(f"⚠️ Error IA (Insight): {str(e)[:100]}")
        if prob_a < 0.15:
            return f"El evento '{evento_si}' es considerado de baja frecuencia (Raro)."
        else:
            return f"El evento '{evento_si}' tiene una frecuencia dentro de los parámetros normales."

# --- FUNCIÓN 3: ACORTADOR DE TEXTOS PARA GRÁFICOS ---
@st.cache_data(show_spinner=False, ttl=3600)
def acortar_texto_grafica(texto, limite=25):
    texto_str = str(texto)
    if len(texto_str) <= limite:
        return texto_str
        
    try:
        modelo = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        Resume el siguiente texto para que sirva como etiqueta o título en un gráfico.
        Debe mantener el sentido original pero ser muy conciso (máximo {limite} caracteres).
        No uses comillas, puntos finales ni explicaciones.
        
        Texto original: "{texto_str}"
        Texto resumido:
        """
        
        respuesta = modelo.generate_content(prompt)
        texto_corto = respuesta.text.strip().strip('\'"')
        
        if len(texto_corto) > limite + 5:
            return texto_str[:limite-3] + "..."
            
        return texto_corto
        
    except Exception:
        return texto_str[:limite-3] + "..."
    
# --- FUNCIÓN 4: SUGERENCIAS DE ANÁLISIS ---
@st.cache_data(show_spinner=False, ttl=3600)
def generar_sugerencias_analisis(df_muestra):
    # Aquí no ponemos pausa porque suele ser la primera en ejecutarse
    try:
        modelo = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Actúa como un analista de datos experto. Aquí tienes una muestra de un dataset:
        {df_muestra}
        
        Basándote en estas columnas y el contexto de los datos, sugiere 3 preguntas analíticas breves y muy interesantes 
        que el usuario podría investigar en este dashboard usando el Teorema de Bayes o probabilidad condicional.
        Ejemplo de formato: "¿Cómo afecta [Columna X] a la probabilidad de [Columna Y]?"
        
        Devuelve ÚNICAMENTE un JSON válido que sea una lista de 3 strings. Ejemplo: ["Pregunta 1", "Pregunta 2", "Pregunta 3"]
        """
        
        respuesta = modelo.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(respuesta.text)
        
    except Exception as e:
        st.cache_data.clear()
        st.toast(f"⚠️ Error IA (Sugerencias): {str(e)[:100]}")
        return [
            "¿Qué variable tiene el mayor impacto en tu objetivo?",
            "¿Cómo cambia la probabilidad al filtrar por diferentes categorías?",
            "Explora la relación entre tus variables de evidencia."
        ]