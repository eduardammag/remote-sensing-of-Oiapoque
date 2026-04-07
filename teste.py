import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def run_c5_boost(features_csv, classes_csv, shp_path, target_col="classv0"):

    print("[ETAPA] Preparando dados...")

    # =========================
    # 1. Carregar CSVs
    # =========================
    df_features = pd.read_csv(features_csv)
    df_classes = pd.read_csv(classes_csv)

    # Criar chave compatível (cell_index começa em 0 → fid começa em 1)
    df_features = df_features.assign(fid=df_features["cell_index"] + 1)

    # =========================
    # 2. Merge tabular
    # =========================
    df = df_features.merge(df_classes, on="fid")

    print("Shape após merge:", df.shape)

    # Limpeza
    df = df.dropna()

    print("Shape final após limpeza:", df.shape)

    # =========================
    # 3. Ler shapefile
    # =========================
    print("[ETAPA] Lendo SHP...")

    gdf = gpd.read_file(shp_path)

    print("Colunas do SHP:", gdf.columns.tolist())
    print("Tipos de geometria:", gdf.geom_type.unique())

    # =========================
    # 4. Garantir chave 'fid'
    # =========================
    if "fid" not in gdf.columns:
        print("Criando coluna fid a partir do índice...")
        gdf = gdf.reset_index().rename(columns={"index": "fid"})
        gdf["fid"] = gdf["fid"] + 1

    # =========================
    # 5. Merge espacial
    # =========================
    gdf_merged = gdf.merge(df[["fid", target_col]], on="fid")

    print("Shape geográfico:", gdf_merged.shape)

    # =========================
    # 6. Plot geográfico
    # =========================
    print("[ETAPA] Gerando mapa geográfico...")

    fig, ax = plt.subplots(figsize=(10, 10))

    gdf_merged.plot(
        column=target_col,
        cmap="Set3",
        legend=True,
        edgecolor="black",
        linewidth=0.2,
        ax=ax
    )

    ax.set_title("Classificação por Células - Oiapoque (2x2 km)")
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()



features_csv = r"C:\Users\dudda\Downloads\remote-sensing-of-Oiapoque\output\metricas_fragstats_por_celula.csv"
classes_csv = r"C:\Users\dudda\Downloads\remote-sensing-of-Oiapoque\input\tipologia_classificada.csv"
shp_path = r"C:\Users\dudda\Downloads\remote-sensing-of-Oiapoque\input\OIAPOQUE_2x2km.shp"

run_c5_boost(
        features_csv,
        classes_csv,
        shp_path,
        target_col="classv0"
    )