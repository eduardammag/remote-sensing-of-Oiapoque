import pandas as pd

# CONFIGURAÇÕES
CSV_PATH = r"output/metricas_fragstats_por_celula.csv"
OUT_PATH = "output/verificacao_metricas_fragstats.csv"
EPS = 1e-6

# FUNÇÕES AUXILIARES
def validate_shape(shape, ed, label, alerts):
    """
    Fragstats:
    - SHAPE < 0 → inválido
    - SHAPE = 0 permitido apenas se ED = 0
    """
    if pd.isna(shape):
        return

    if shape < 0:
        alerts.append(f"{label}: SHAPE_MN negativo")
    elif shape == 0 and ed > 0:
        alerts.append(f"{label}: SHAPE_MN = 0 com ED > 0")

