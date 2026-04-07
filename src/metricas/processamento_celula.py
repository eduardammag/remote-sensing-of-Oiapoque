import math
import numpy as np
from rasterio.features import rasterize
import rasterio.transform
import traceback
from src.metricas.calculo_metricas import compute_metrics

def process_cell(i, cell_geom, layer_gdfs, class_map, resolution):
    """ Processa uma célula individual da grade, rasterizando as camadas e calculando métricas.
    
    Args:
        i (int): Índice da célula
        cell_geom (shapely.geometry): Geometria da célula (polígono)
        layer_gdfs (list): Lista de tuplas (GeoDataFrame, nome_coluna_classe) para cada camada
        class_map (dict): Mapeamento de valores de classe para códigos inteiros
        resolution (float): Resolução espacial do raster em unidades do CRS
    
    Returns:
        dict: Dicionário com métricas calculadas e status do processamento"""
    print(f"\n[DEBUG Célula {i}] Iniciando processamento...")
    
    # Obter bounding box da célula
    minx, miny, maxx, maxy = cell_geom.bounds
    width, height = maxx - minx, maxy - miny
    
    # Verificar se geometria não é vazia ou degenerada
    if width == 0 or height == 0:
        return {"cell_index": i, "status": "empty_geom"}
    
    # Calcular dimensões do raster em pixels
    pix = float(resolution)
    nx = max(1, int(math.ceil(width / pix)))   # Número de colunas
    ny = max(1, int(math.ceil(height / pix)))  # Número de linhas
    
    # Criar transformação de coordenadas para rasterização
    # from_origin: minx, maxy definem o canto superior esquerdo
    transform = rasterio.transform.from_origin(minx, maxy, pix, pix)
    
    # Array para armazenar o raster resultante (inicializado com zeros)
    arr = np.zeros((ny, nx), dtype=np.int32)
    
    # Conjuntos para rastreamento de classes presentes na célula
    classes_in_cell = set()
    shapes_count = 0
    
    # Processar cada camada vetorial
    for layer_idx, (layer_gdf, class_col) in enumerate(layer_gdfs):
        try:
            # Filtrar feições que intersectam a célula
            layer_subset = layer_gdf[layer_gdf.geometry.intersects(cell_geom)]
            if layer_subset.empty:
                continue  # Pular se não houver interseção
            
            # Preparar formas para rasterização
            shapes = []
            for idx, row in layer_subset.iterrows():
                # Clippar geometria para os limites da célula
                geom_inter = row.geometry.intersection(cell_geom)
                if geom_inter.is_empty:
                    continue  # Pular geometrias vazias após interseção
                
                # Obter código da classe usando o mapeamento
                key = str(row[class_col])
                code = class_map.get(key)
                if code is None:
                    continue  # Pular se classe não estiver mapeada
                
                # Adicionar forma (geometria, código) para rasterização
                shapes.append((geom_inter, int(code)))
                classes_in_cell.add(f"{key}({code})")  # Registro para debug
                shapes_count += 1
            
            # Rasterizar se houver formas nesta camada
            if shapes:
                # Rasteriza sobrepondo valores (usando máximo para preservar prioridades)
                tmp = rasterize(shapes, out_shape=(ny, nx), transform=transform,
                                fill=0, dtype=np.int32, all_touched=True)
                arr = np.maximum(arr, tmp)  # Sobrescreve com valores maiores (sobreposição)
        
        except Exception as e:
            print(f"[ERRO Célula {i}] erro processando camada {layer_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Verificar se há valores não-zero no raster
    unique_vals = np.unique(arr)
    if len(unique_vals[unique_vals > 0]) == 0:
        return {"cell_index": i, "status": "empty_cell", "classes": None}
    
    # Definir valor para áreas sem dados (nodata)
    # Se apenas uma classe presente, usar -1 para background
    if len(unique_vals[unique_vals > 0]) == 1:
        arr[arr == 0] = -1
        nodata_val = -1
    else:
        nodata_val = 0  # Zero representa background/sem classe
    
    arr = arr.astype(np.int32)
    
    # Calcular métricas usando função externa
    try:
        class_metrics, landscape_metrics = compute_metrics(arr, pix)
    except Exception as e:
        return {"cell_index": i, "status": f"metrics_failed: {e}", "classes": list(classes_in_cell)}
    
    # Montar dicionário de resultados
    out = {"cell_index": i, "status": "ok", "classes": list(classes_in_cell)}
    
    # Adicionar métricas de paisagem (nível geral)
    for k, v in landscape_metrics.items():
        out[f"land_{k}"] = float(v) if v is not None else None
    
    # Adicionar métricas por classe (nível de classe)
    for cls, metrics in class_metrics.items():
        for k, v in metrics.items():
            out[f"cls_{int(cls)}_{k}"] = float(v) if v is not None else None
    
    return out
