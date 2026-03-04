from sklearn.metrics import confusion_matrix, accuracy_score

def calcular_probabilidades_bayes(df, col_objetivo, evento_si, col_evidencia, valor_o_umbral, es_numerica):
    """Realiza todos los cálculos del Teorema de Bayes."""
    total = len(df)
    prob_a = len(df[df[col_objetivo] == evento_si]) / total
    
    if es_numerica:
        evidencia_mask = df[col_evidencia] > valor_o_umbral
    else:
        evidencia_mask = df[col_evidencia] == valor_o_umbral
    
    prob_b = evidencia_mask.mean()
    
    df_fallo = df[df[col_objetivo] == evento_si]
    if len(df_fallo) > 0:
        if es_numerica:
            prob_b_dado_a = (df_fallo[col_evidencia] > valor_o_umbral).mean()
        else:
            prob_b_dado_a = (df_fallo[col_evidencia] == valor_o_umbral).mean()
    else:
        prob_b_dado_a = 0
        
    prob_posterior = (prob_b_dado_a * prob_a) / prob_b if prob_b > 0 else 0
    
    return {
        "prob_a": prob_a,
        "prob_b": prob_b,
        "prob_b_dado_a": prob_b_dado_a,
        "prob_posterior": prob_posterior
    }

def evaluar_modelo(df, col_objetivo, evento_si, col_evidencia, valor_o_umbral, es_numerica):
    """Genera etiquetas de predicción y calcula métricas."""
    def predecir(x):
        condicion = (x > valor_o_umbral) if es_numerica else (x == valor_o_umbral)
        return evento_si if condicion else "No Anomalía"
    
    y_real = df[col_objetivo].apply(lambda x: evento_si if x == evento_si else "No Anomalía")
    y_pred = df[col_evidencia].apply(predecir)
    
    acc = accuracy_score(y_real, y_pred)
    cm = confusion_matrix(y_real, y_pred, labels=[evento_si, "No Anomalía"])
    
    return acc, cm