import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Importamos la función desde tu módulo
from modulos.ia import acortar_texto_grafica

# ==========================================
# GRÁFICAS DE BAYES (CORREGIDAS)
# ==========================================

def grafico_comparativo_bayes(prob_a, prob_posterior, evento_si):
    """Genera comparativa de barras con espacio para etiquetas superiores."""
    evento_si_corto = acortar_texto_grafica(evento_si, limite=25)
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Base P(A)', 'Posterior P(A|B)'],
            y=[prob_a, prob_posterior],
            marker_color=['#643DF2', '#F27F3D'],
            text=[f"{prob_a:.1%}", f"{prob_posterior:.1%}"],
            textposition='outside',  # Muestra el % arriba de la barra
            cliponaxis=False         # Evita que el texto se corte arriba
        )
    ])
    
    fig.update_layout(
        title_text=f'Impacto en {evento_si_corto}',
        font=dict(family="'Inter', sans-serif", color="#B0B0B0", size=14),
        title_font=dict(color="#FFFFFF", size=18),
        # Rango a 1.2 (120%) para que el texto de la barra no pegue con el techo
        yaxis=dict(tickformat=".0%", range=[0, 1.2], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        template="none",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=100, b=40) # Margen superior amplio
    )
    return fig

def grafico_evolucion_posterior(prob_inicial, evidencias, probs_posteriores):
    """Muestra la evolución con puntos y líneas sin cortes en los bordes."""
    # Acortamos etiquetas para evitar que choquen entre sí en el eje X
    etiquetas_x = ['Prob. Inicial'] + [acortar_texto_grafica(e, limite=12) for e in evidencias]
    valores_y = [prob_inicial] + probs_posteriores
    
    fig = go.Figure(data=[
        go.Scatter(
            x=etiquetas_x,
            y=valores_y,
            mode='lines+markers+text',
            line=dict(color='#643DF2', width=3, dash='dot'),
            marker=dict(size=10, color='#F27F3D'),
            text=[f"{v:.1%}" for v in valores_y],
            textposition="top center",
            cliponaxis=False # Permite que el texto flote fuera del área de trazado
        )
    ])
    
    fig.update_layout(
        title_text='Actualización de la Prob. Posterior',
        font=dict(family="'Inter', sans-serif", color="#B0B0B0", size=13),
        title_font=dict(color="#FFFFFF", size=18),
        # Rango extendido a 1.2 para dar aire a los porcentajes superiores
        yaxis=dict(tickformat=".0%", range=[0, 1.2], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(tickangle=-30), # Inclinación para que los textos no se monten
        template="none",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=80, b=60)
    )
    return fig

# ==========================================
# OTRAS GRÁFICAS (ESTADÍSTICAS)
# ==========================================

def grafico_matriz_confusion(cm, etiquetas):
    etiquetas_cortas = [acortar_texto_grafica(etiq, limite=15) for etiq in etiquetas]
    
    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'Arial'],
        'text.color': '#B0B0B0',
        'axes.labelcolor': '#B0B0B0',
        'xtick.color': '#B0B0B0',
        'ytick.color': '#B0B0B0',
    }):
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=[f'P: {etiquetas_cortas[0]}', f'P: {etiquetas_cortas[1]}'],
            yticklabels=[f'R: {etiquetas_cortas[0]}', f'R: {etiquetas_cortas[1]}'],
            ax=ax, cbar=True
        )
        ax.set_xlabel("Predicción (P)", color="#FFFFFF", fontweight='bold', labelpad=10)
        ax.set_ylabel("Real (R)", color="#FFFFFF", fontweight='bold', labelpad=10)
        
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='#B0B0B0', labelcolor='#B0B0B0')
        ax.set_title("Matriz de Confusión", color="#FCFCFC", fontweight='bold', pad=15)
        
        plt.tight_layout()
        return fig

def grafico_histograma(datos, nombre_variable):
    nombre_corto = acortar_texto_grafica(nombre_variable, limite=40)
    fig = go.Figure(data=[
        go.Histogram(
            x=datos,
            marker_color='#643DF2',
            opacity=0.8,
            nbinsx=20,
            texttemplate='%{y}',
            textposition='outside',
            cliponaxis=False
        )
    ])
    
    fig.update_layout(
        title=dict(text=f'Distribución de {nombre_corto}', font=dict(color="#FFFFFF", size=20), pad=dict(b=20)),
        font=dict(family="'Inter', sans-serif", color="#B0B0B0", size=14),
        xaxis_title="Rango de Valores",
        yaxis_title="Frecuencia (Cantidad)",
        template="none",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=90, b=40),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, max(datos)*1.2 if datos else 1])
    )
    return fig

def grafico_temporal(fechas, valores, nombre_metrica):
    nombre_corto = acortar_texto_grafica(nombre_metrica, limite=40)
    fig = go.Figure(data=[
        go.Scatter(
            x=fechas,
            y=valores,
            mode='lines+markers+text',
            line=dict(color='#F27F3D', width=3),
            marker=dict(size=8, color='#FFFFFF', line=dict(width=2, color='#F27F3D')),
            text=[f"{v:.1f}" if isinstance(v, (int, float)) else str(v) for v in valores],
            textposition="top center",
            textfont=dict(size=12, color="#E0E0E0"),
            cliponaxis=False
        )
    ])
    
    max_val = max(valores) if len(valores) > 0 else 1
    fig.update_layout(
        title=dict(text=f'Evolución temporal: {nombre_corto}', font=dict(color="#FFFFFF", size=20), pad=dict(b=20)),
        font=dict(family="'Inter', sans-serif", color="#B0B0B0", size=14),
        xaxis_title="Tiempo (Secuencia)",
        yaxis_title="Valor Medido",
        template="none",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=90, b=50),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, max_val * 1.25])
    )
    return fig

# ==========================================
# FUNCIONES DE EXPORTACIÓN
# ==========================================

def exportar_plotly_jpg(fig):
    """Convierte gráfica Plotly a bytes JPG."""
    fig_export = go.Figure(fig)
    fig_export.update_layout(paper_bgcolor='#0E1117', plot_bgcolor='#0E1117')
    return fig_export.to_image(format="jpg", width=800, height=500)

def exportar_matplotlib_jpg(fig):
    """Convierte gráfica Matplotlib a bytes JPG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', facecolor='#0E1117', edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()