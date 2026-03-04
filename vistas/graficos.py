import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def grafico_comparativo_bayes(prob_a, prob_posterior, evento_si):
    """Crea el gráfico de barras comparando prob. base vs posterior."""
    fig = go.Figure(data=[
        go.Bar(name='Probabilidad', 
               x=['Base P(A)', 'Posterior P(A|B)'], 
               y=[prob_a, prob_posterior],
               marker_color=['#636EFA', '#EF553B'])
    ])
    fig.update_layout(
        title_text=f'Impacto de la Evidencia en {evento_si}',
        yaxis=dict(tickformat=".2%", range=[0, 1]),
        template="plotly_white"
    )
    return fig

def grafico_matriz_confusion(cm, etiquetas):
    """Genera el heatmap de la matriz de confusión usando Seaborn."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Pred: {etiquetas[0]}', f'Pred: {etiquetas[1]}'], 
                yticklabels=[f'Real: {etiquetas[0]}', f'Real: {etiquetas[1]}'], 
                ax=ax)
    ax.set_title("Matriz de Confusión")
    return fig