import os
import geopandas as gpd
import warnings
import gc  # Garbage collector, usado para liberar memória antes de remover arquivos
import rasterio
from rasterio.mask import mask
import numpy as np
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
# Ignora warnings para não poluir a saída do terminal
warnings.filterwarnings("ignore")

# UTILITÁRIOS

def safe_remove(path):
    """
    Remove um arquivo de forma segura.
    Antes de remover, força a coleta de lixo para liberar possíveis
    locks sobre o arquivo. Se o arquivo estiver aberto, captura PermissionError.
    
    Parâmetros:
    - path: caminho do arquivo a ser removido
    """
    if os.path.exists(path):
        # Força a coleta de memória para liberar handles de arquivos
        gc.collect()
        try:
            os.remove(path)
        except PermissionError:
            print(f"Atenção: não foi possível remover '{path}', talvez esteja aberto.")


def read_unique_classes(layer_path, class_col):
    """
    Lê uma camada vetorial e retorna a lista de classes únicas presentes em uma coluna específica.
    
    Parâmetros:
    - layer_path: caminho para o arquivo da camada (shapefile, geopackage, etc.)
    - class_col: nome da coluna que contém as classes
    
    Retorna:
    - lista de valores únicos na coluna de classe
    """
    try:
        gdf = gpd.read_file(layer_path)  # Lê o arquivo geoespacial usando GeoPandas
        if class_col not in gdf.columns:
            # Verifica se a coluna existe, lança erro se não
            raise ValueError(f"Coluna {class_col} não existe em {layer_path}")
        
        # Extrai classes únicas, ignorando valores nulos
        classes = gdf[class_col].dropna().unique().tolist()
        
        # Debug: mostra quantas classes foram encontradas
        print(f"  [DEBUG] Camada {os.path.basename(layer_path)} tem {len(classes)} classes: {classes}")
        return classes
    except Exception as e:
        # Captura qualquer erro durante a leitura e retorna lista vazia
        print(f"[ERRO] lendo classes de {layer_path}: {e}")
        return []


def build_class_code_map(layer_list):
    """
    Constrói um mapa de códigos únicos para todas as classes de todas as camadas.
    Útil para unificar classes de diferentes fontes antes de rasterizar ou calcular métricas.
    
    Parâmetros:
    - layer_list: lista de tuplas (caminho_da_camada, coluna_de_classe)
    
    Retorna:
    - dicionário {classe_original: código_inteiro}
    """
    code_map = {}  # dicionário final de mapeamento
    next_code = 1  # contador de códigos únicos
    print("[DEBUG] Construindo mapa de códigos de classe...")

    # Itera sobre todas as camadas
    for path, col in layer_list:
        vals = read_unique_classes(path, col)  # lê classes únicas da camada
        for v in vals:
            key = str(v)  # converte a classe em string (para consistência)
            if key not in code_map:
                code_map[key] = next_code  # atribui código único
                print(f"  [DEBUG] Mapeando classe '{key}' -> código {next_code}")
                next_code += 1

    print(f"[DEBUG] Total de {len(code_map)} classes únicas mapeadas")
    return code_map






def zonal_stats_raster(raster_path, geometry, src_crs=None, stats=['mean', 'sum']):
    """
    Extrai estatísticas de um raster para uma geometria (polígono).
    
    Parâmetros:
    - raster_path: caminho para o arquivo raster
    - geometry: uma geometria Shapely ou GeoDataFrame/GeoSeries
    - src_crs: CRS da geometria (obrigatório se geometry não tiver atributo crs)
    - stats: lista de estatísticas desejadas ('mean', 'sum')
    
    Retorna:
    - dicionário com as estatísticas calculadas
    """
    with rasterio.open(raster_path) as src:
        # Determinar o CRS da geometria
        if hasattr(geometry, 'crs'):
            geom_crs = geometry.crs
            # Se for GeoDataFrame, extrair a primeira geometria (ou iterar?)
            # Para simplificar, assumimos que é uma geometria única.
            if hasattr(geometry, 'geometry'):
                # É um GeoDataFrame/GeoSeries, pegar a geometria do primeiro elemento
                geom = geometry.geometry.iloc[0]
            else:
                geom = geometry
        else:
            geom_crs = src_crs
            geom = geometry

        # Reprojetar geometria para o CRS do raster se necessário
        if geom_crs and geom_crs != src.crs:
            geom = shape(transform_geom(geom_crs, src.crs, mapping(geom)))

        try:
            out_image, out_transform = mask(src, [geom], crop=True, nodata=np.nan)
            data = out_image[0]
            results = {}
            if 'mean' in stats:
                results['mean'] = np.nanmean(data)
            if 'sum' in stats:
                results['sum'] = np.nansum(data)
            return results
        except Exception as e:
            # Em caso de erro (ex.: geometria fora da extensão do raster)
            return {stat: np.nan for stat in stats}