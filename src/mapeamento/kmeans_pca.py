""" Mapeamento por clusterização (unsupervised)"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# FEATURES IMPORTANTES
def selecionar_features(df):

    features = [
        # PAISAGEM (estrutura)
        "land_ED",      # fragmentação (ESSENCIAL)
        "land_NP",      # número de fragmentos
        "land_LPI",     # dominância

        # USO DO SOLO (resumido)
        "cls_1_PLAND",  # floresta primária
        "cls_6_PLAND",  # floresta secundária

        # ACESSO
        "dist_m_estrada",
        "dist_m_hidrografia",
        "dist_m_mineracao",

        # PRESSÃO HUMANA
        "built_mean",
        "pop_mean"]
    df = df.copy()
    # preencher NaN de classes com 0
    for col in ["cls_1_PLAND", "cls_6_PLAND"]:
        df[col] = df[col].fillna(0)

    df = df.dropna(subset=["land_ED","land_NP","land_LPI","dist_m_estrada","dist_m_hidrografia",
        "dist_m_mineracao","built_mean","pop_mean"])
    return df, features


# CLUSTERIZAÇÃO
def rodar_cluster(df, features, n_clusters=5):

    # ESCALAR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # PCA (redução de dimensionalidade)
    pca = PCA(n_components=5)  # você pode testar 3–8
    X_pca = pca.fit_transform(X_scaled)

    print("\nVariância explicada pelo PCA:")
    print(pca.explained_variance_ratio_)
    print("Total:", pca.explained_variance_ratio_.sum())

    # CLUSTER
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df["cluster"] = model.fit_predict(X_pca)

    return df, model

# ROTULAR CLUSTERS (INTELIGENTE)
def rotular_clusters(df):

    # pegar apenas colunas numéricas
    df_num = df.select_dtypes(include="number")

    summary = df_num.groupby(df["cluster"]).mean()

    # lógica de risco
    summary["score"] = (
        summary["land_ED"] * 2 +
        summary["built_mean"] * 2 -
        summary["dist_m_estrada"] * 0.5
    )

    ranking = summary["score"].rank()

    label_map = {}

    for cluster, rank in ranking.items():
        if rank == ranking.max():
            label_map[cluster] = "high_risk"
        elif rank >= ranking.quantile(0.75):
            label_map[cluster] = "proximity"
        elif rank >= ranking.quantile(0.5):
            label_map[cluster] = "access"
        else:
            label_map[cluster] = "low"

    df["vulnerability_class"] = df["cluster"].map(label_map)
    return df, summary

# VISUALIZAÇÃO
def plotar(gdf):
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(column="vulnerability_class",legend=True,ax=ax)
    plt.title("Vulnerabilidade (Cluster)")
    plt.axis("off")
    plt.savefig("output")
    plt.show()


# PIPELINE
def run_clusters(csv_path, grid_path, out_path):
    print("\n[ETAPA 4] Clusterização de vulnerabilidade")
    df = pd.read_csv(csv_path)
    grid = gpd.read_file(grid_path)

    # FEATURES
    df, features = selecionar_features(df)

    # CLUSTER
    df, model = rodar_cluster(df, features, n_clusters=5)

    # ROTULAR
    df, summary = rotular_clusters(df)

    print("\nResumo dos clusters:")
    print(summary)

    # GEO
    gdf = grid.merge(df, left_index=True, right_index=True)

    # SALVAR
    gdf.to_file(out_path, driver="GPKG")

    print(f"\nMapa salvo em: {out_path}")

    # PLOT
    plotar(gdf)
    return gdf