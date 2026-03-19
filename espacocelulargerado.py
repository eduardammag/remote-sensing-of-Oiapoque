import geopandas as gpd
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- FUNÇÕES DE GRADE ----------------

def load_limite(path: str) -> gpd.GeoDataFrame:
    """
    Carrega o limite da área de interesse e garante que o CRS seja métrico (UTM).

    Parâmetros:
    - path: caminho para o arquivo do limite (shapefile, geopackage, etc.)

    Retorna:
    - GeoDataFrame do limite com CRS métrico
    """
    limite = gpd.read_file(path)

    # Se o CRS estiver em coordenadas geográficas (graus), converte para UTM
    if limite.crs.is_geographic:
        limite = limite.to_crs(limite.estimate_utm_crs())
    return limite


def create_grid(limite: gpd.GeoDataFrame, cell_size: int = 2000) -> gpd.GeoDataFrame:
    """
    Gera uma grade regular de polígonos sobre o limite fornecido.
    Cada célula tem tamanho 'cell_size' (em metros).

    Parâmetros:
    - limite: GeoDataFrame com a área de interesse
    - cell_size: tamanho da célula da grade em metros (padrão: 2000 m)

    Retorna:
    - GeoDataFrame com polígonos da grade
    """
    # Obtém os limites da área (min/max x e y)
    minx, miny, maxx, maxy = limite.total_bounds

    # Cria arrays com coordenadas dos vértices da grade
    x_coords = np.arange(minx, maxx + cell_size, cell_size)
    y_coords = np.arange(miny, maxy + cell_size, cell_size)

    # Gera os polígonos da grade
    grid_cells = [
        box(x, y, x + cell_size, y + cell_size)
        for x in x_coords[:-1]  # exclui o último para não criar célula extra
        for y in y_coords[:-1]
    ]

    # Retorna GeoDataFrame com a grade e CRS igual ao limite
    return gpd.GeoDataFrame({"geometry": grid_cells}, crs=limite.crs)


def clip_grid(grid: gpd.GeoDataFrame, limite: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Recorta a grade para se ajustar exatamente ao limite da área de interesse.
    Adiciona um identificador único para cada célula.

    Parâmetros:
    - grid: GeoDataFrame da grade completa
    - limite: GeoDataFrame do limite da área

    Retorna:
    - GeoDataFrame da grade recortada com coluna 'cell_id'
    """
    # Realiza overlay (interseção) entre grade e limite
    grid_clip = gpd.overlay(grid, limite, how="intersection")

    # Cria identificador único para cada célula
    grid_clip["cell_id"] = [f"CELL_{i+1}" for i in range(len(grid_clip))]
    return grid_clip


def save_grid(grid: gpd.GeoDataFrame, path: str):
    """
    Salva a grade final em um arquivo GPKG.
    Se o arquivo já existir, tenta remover antes de salvar.

    Parâmetros:
    - grid: GeoDataFrame da grade a salvar
    - path: caminho de saída do arquivo
    """
    import gc

    if os.path.exists(path):
        # Tenta liberar memória antes de remover
        gc.collect()
        try:
            os.remove(path)
        except PermissionError:
            print(f"Atenção: não foi possível remover '{path}', talvez esteja aberto.")
            return

    # Salva o arquivo como GeoPackage
    grid.to_file(path, driver="GPKG", mode="w")
    print(f"Grade gerada com {len(grid)} células e salva em '{path}'")


def plot_grid(limite: gpd.GeoDataFrame, grid: gpd.GeoDataFrame):
    """
    Plota o limite da área e a grade sobreposta.

    Parâmetros:
    - limite: GeoDataFrame do limite
    - grid: GeoDataFrame da grade
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plota o limite em preto
    limite.boundary.plot(ax=ax, color="black", linewidth=1)

    # Plota a grade em azul transparente
    grid.plot(ax=ax, facecolor="none", edgecolor="blue", linewidth=0.3)

    # Adiciona título e rótulos
    plt.title("Grade 2x2 km sobre o Limite de Oiapoque", fontsize=14)
    plt.xlabel("Coordenada X (m)")
    plt.ylabel("Coordenada Y (m)")
    plt.show()
