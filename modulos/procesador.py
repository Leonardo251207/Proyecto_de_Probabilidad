import pandas as pd

def cargar_datos(archivo):
    """Lee el archivo y retorna un DataFrame."""
    return pd.read_csv(archivo)

def optimizar_tipos(df):
    """Detecta fechas y organiza las columnas por tipos."""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    tipos = {
        'fechas': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'numericas': df.select_dtypes(include=['number']).columns.tolist(),
        'categoricas': df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    }
    return df, tipos