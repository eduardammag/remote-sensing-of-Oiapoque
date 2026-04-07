import pandas as pd
from src.config import EPS, OUTPUT_RELATORIO

# CONFIGURAÇÕES
CSV_PATH = r"output/metricas_fragstats_por_celula.csv"

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

def validar_metricas():
    print("\n[ETAPA 3] Validação Fragstats-like")
    df = pd.read_csv(CSV_PATH)
    print(f"Células carregadas: {len(df)}")

    # detectar classes automaticamente
    class_ids = sorted({
        int(c.split("_")[1])
        for c in df.columns
        if c.startswith("cls_") and c.endswith("_CA")
    })

    print(f"Classes detectadas: {class_ids}\n")

    alerts_all = []

    for _, row in df.iterrows():
        alerts = []

        # -------- PAISAGEM --------
        TA = row.get("land_TA")
        NP = row.get("land_NP")
        ED = row.get("land_ED")
        SHAPE = row.get("land_SHAPE_MN")

        if pd.isna(TA) or TA <= 0:
            alerts.append("land_TA inválido")

        if NP < 0:
            alerts.append("land_NP negativo")

        if ED < 0:
            alerts.append("land_ED negativo")

        validate_shape(SHAPE, ED, "land", alerts)

        # -------- CLASSES --------
        for cls in class_ids:
            CA = row.get(f"cls_{cls}_CA")
            NPc = row.get(f"cls_{cls}_NP")
            EDc = row.get(f"cls_{cls}_ED")
            SHAPEc = row.get(f"cls_{cls}_SHAPE_MN")

            if pd.isna(CA) or CA == 0:
                continue

            if CA < 0:
                alerts.append(f"cls_{cls}: CA negativo")

            if CA > TA + EPS:
                alerts.append(f"cls_{cls}: CA > land_TA")

            if NPc < 0:
                alerts.append(f"cls_{cls}: NP negativo")

            if EDc < 0:
                alerts.append(f"cls_{cls}: ED negativo")

            validate_shape(SHAPEc, EDc, f"cls_{cls}", alerts)

        alerts_all.append(alerts)

    # -------- RESULTADOS --------
    df["alerts"] = alerts_all
    df["n_alerts"] = df["alerts"].apply(len)

    total = len(df)
    with_alerts = (df["n_alerts"] > 0).sum()

    print("\nRESULTADO DA VALIDAÇÃO")
    print("-" * 80)
    print(f"Células totais: {total}")
    print(f"Células com alertas: {with_alerts} ({100*with_alerts/total:.2f}%)")

    print("\nExemplos de alertas:")
    for i, r in df[df["n_alerts"] > 0].head(10).iterrows():
        print(f"Cell {i}: {r['alerts']}")

    # -------- ESTATÍSTICAS --------
    metrics_land = [
        "land_TA", "land_NP", "land_PD",
        "land_AREA_MN", "land_LPI",
        "land_ED", "land_SHAPE_MN"
    ]

    print("\nRESUMO ESTATÍSTICO (PAISAGEM)")
    print(df[metrics_land].describe())

    print("\nCORRELAÇÕES")
    print(df[metrics_land].corr())

    # -------- EXPORTAÇÃO --------
    df.to_csv(OUTPUT_RELATORIO, index=False)
    print(f"\nRelatório salvo em: {OUTPUT_RELATORIO}")
