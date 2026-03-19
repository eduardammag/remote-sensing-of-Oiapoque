import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
from config import RASTER_RESOLUTION, CRS_TARGET, TEST_N, N_WORKERS
from utils import safe_remove, build_class_code_map, zonal_stats_raster  # nova função
from processamento_celula import process_cell


# ------ FUNÇÃO PRINCIPAL ------
def run_all(layer_list, grade_path, outdir, resolution=RASTER_RESOLUTION, test_n=TEST_N,
            crs_target=CRS_TARGET, road_path=None, mining_path=None,
            hydro_path=None, indigenous_path=None, dump_path=None,
            pop_raster=None, built_raster=None, n_workers=N_WORKERS):
    """
    Função principal que orquestra o processamento de todas as células da grade.

    Args:
        layer_list (list): Lista de tuplas (caminho_arquivo, nome_coluna_classe) para cada camada
        grade_path (str): Caminho para arquivo da grade (shapefile/geopackage)
        outdir (str): Diretório de saída para resultados
        resolution (float): Resolução espacial para rasterização
        test_n (int or None): Número de células para teste (None = processa todas)
        crs_target (str): CRS de destino para reprojeção
        road_path (str): Caminho para camada de estradas (opcional)
        mining_path (str): Caminho para camada de mineração (opcional)
        hydro_path (str): Caminho para camada de hidrografia (opcional)
        indigenous_path (str): Caminho para camada de áreas indígenas (opcional)
        dump_path (str): Caminho para camada de lixão (opcional)
        pop_raster (str): Caminho para raster de população (opcional)
        built_raster (str): Caminho para raster de área construída (opcional)
        n_workers (int): Número de processos paralelos (1 = sequencial)

    Returns:
        dict: Dicionário com caminhos dos arquivos gerados e DataFrames
    """

    print("=" * 80)
    print("INICIANDO PROCESSAMENTO FRAGSTATS")
    print("=" * 80)

    # CARREGAR E PREPARAR GRADE
    grade = gpd.read_file(grade_path)

    # Reprojetar grade se necessário
    if crs_target is not None:
        grade = grade.to_crs(crs_target)

    # Verificar se CRS está em unidades métricas (área > 10k m²)
    if grade.geometry.iloc[0].area < 1e4:
        raise RuntimeError("Grade não está em CRS métrico. Reprojete e tente novamente.")

    # Construir mapa de códigos de classe a partir das camadas
    class_map = build_class_code_map(layer_list)

    # CARREGAR E PREPARAR CAMADAS
    layer_gdfs = []
    for path, class_col in layer_list:
        gdf = gpd.read_file(path)
        # Reprojetar para mesmo CRS da grade
        if gdf.crs != grade.crs:
            gdf = gdf.to_crs(grade.crs)
        layer_gdfs.append((gdf, class_col))

    # PROCESSAR CAMADAS AUXILIARES (distâncias)
    road_union, mining_union = None, None
    hydro_union, indigenous_union, dump_union = None, None, None

    # Unificar geometrias de estradas
    if road_path and os.path.exists(road_path):
        r = gpd.read_file(road_path).to_crs(grade.crs)
        road_union = unary_union(r.geometry.values)

    # Unificar geometrias de mineração
    if mining_path and os.path.exists(mining_path):
        m = gpd.read_file(mining_path).to_crs(grade.crs)
        mining_union = unary_union(m.geometry.values)

    # Unificar geometrias de hidrografia
    if hydro_path and os.path.exists(hydro_path):
        h = gpd.read_file(hydro_path).to_crs(grade.crs)
        hydro_union = unary_union(h.geometry.values)

    # Unificar geometrias de áreas indígenas
    if indigenous_path and os.path.exists(indigenous_path):
        ind = gpd.read_file(indigenous_path).to_crs(grade.crs)
        indigenous_union = unary_union(ind.geometry.values)

    # Unificar geometrias de lixão
    if dump_path and os.path.exists(dump_path):
        d = gpd.read_file(dump_path).to_crs(grade.crs)
        dump_union = unary_union(d.geometry.values)

    # PREPARAR TAREFAS DE PROCESSAMENTO
    grade_reset = grade.reset_index().rename(columns={"index": "cell_index"})
    tasks = [(row.cell_index, row.geometry) for _, row in grade_reset.iterrows()]

    # Limitar número de células para modo de teste
    if test_n is not None:
        tasks = tasks[:test_n]
        print(f"[MODO TESTE] Processando apenas {test_n} células")

    # PROCESSAR CÉLULAS (PARALELO OU SEQUENCIAL)
    results = []
    if n_workers > 1:
        print(f"[PARALELO] Usando {n_workers} processos")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_cell = {
                executor.submit(process_cell, idx, geom, layer_gdfs, class_map, resolution): idx
                for idx, geom in tasks
            }
            for future in as_completed(future_to_cell):
                results.append(future.result())
    else:
        print("[SEQUENCIAL] Processando uma célula por vez")
        for idx, geom in tasks:
            results.append(process_cell(idx, geom, layer_gdfs, class_map, resolution))

    # CONSOLIDAR RESULTADOS DAS MÉTRICAS FRAGSTATS
    df = pd.DataFrame(results).sort_values("cell_index").reset_index(drop=True)

    # -----------------------------------------------------------------
    # CALCULAR DISTÂNCIAS E ESTATÍSTICAS DE RASTER PARA CADA CÉLULA
    # -----------------------------------------------------------------
    print("[EXTRA] Calculando distâncias e estatísticas adicionais...")

    # Criar GeoDataFrame com geometrias indexadas por cell_index
    geo_indexed = grade_reset.set_index("cell_index").geometry

    # Preparar listas para as novas colunas
    dist_road = []
    dist_mining = []
    dist_hydro = []
    dist_indigenous = []
    dist_dump = []

    pop_mean = []
    pop_sum = []
    built_mean = []
    built_sum = []

    # Iterar sobre as células na mesma ordem do df (já ordenado por cell_index)
    for cell_idx in df["cell_index"]:
        geom = geo_indexed.loc[cell_idx]
        cen = geom.centroid

        # Distâncias
        dist_road.append(float(cen.distance(road_union)) if road_union else np.nan)
        dist_mining.append(float(cen.distance(mining_union)) if mining_union else np.nan)
        dist_hydro.append(float(cen.distance(hydro_union)) if hydro_union else np.nan)
        dist_indigenous.append(float(cen.distance(indigenous_union)) if indigenous_union else np.nan)
        dist_dump.append(float(cen.distance(dump_union)) if dump_union else np.nan)

        # Estatísticas zonais (população)
        if pop_raster and os.path.exists(pop_raster):
            # A função zonal_stats_raster deve retornar dict com 'mean' e 'sum' para uma única geometria
            # Vamos chamar uma versão que processa uma geometria por vez (simples)
            # Aqui estamos assumindo que a função aceita uma geometria e o caminho do raster
            stats = zonal_stats_raster(pop_raster, geom, src_crs=grade.crs)
            pop_mean.append(stats.get('mean', np.nan))
            pop_sum.append(stats.get('sum', np.nan))
        else:
            pop_mean.append(np.nan)
            pop_sum.append(np.nan)

        # Estatísticas zonais (área construída)
        if built_raster and os.path.exists(built_raster):
            stats = zonal_stats_raster(built_raster, geom, src_crs=grade.crs)
            built_mean.append(stats.get('mean', np.nan))
            built_sum.append(stats.get('sum', np.nan))
        else:
            built_mean.append(np.nan)
            built_sum.append(np.nan)

    # Adicionar colunas ao DataFrame
    df["dist_m_estrada"] = dist_road
    df["dist_m_mineracao"] = dist_mining
    df["dist_m_hidrografia"] = dist_hydro
    df["dist_m_indigena"] = dist_indigenous
    df["dist_m_lixao"] = dist_dump
    df["pop_mean"] = pop_mean
    df["pop_sum"] = pop_sum
    df["built_mean"] = built_mean
    df["built_sum"] = built_sum

    # -----------------------------------------------------------------
    # SALVAR CSV (COM TODAS AS MÉTRICAS + NOVAS COLUNAS)
    # -----------------------------------------------------------------
    csv_out = os.path.join(outdir, "metricas_fragstats_por_celula.csv")
    df.to_csv(csv_out, index=False, encoding='utf-8')
    print(f"\n[SAÍDA] Métricas salvas em: {csv_out}")

    # -----------------------------------------------------------------
    # CRIAR GEOGEOPACKAGE COM RESULTADOS
    # -----------------------------------------------------------------
    # Juntar métricas à grade pelo cell_index
    df_indexed = df.set_index("cell_index")
    grade_out = grade_reset.join(df_indexed, on="cell_index", how="left")

    # Salvar como GeoPackage
    gpkg_out = os.path.join(outdir, "grade_2km_metricas_fragstats.gpkg")
    safe_remove(gpkg_out)
    grade_out.to_file(gpkg_out, driver="GPKG")
    print(f"[SAÍDA] Grade georreferenciada salva em: {gpkg_out}")

    # RETORNAR RESULTADOS
    return {
        "csv": csv_out,
        "gpkg": gpkg_out,
        "df": df,
        "grade": grade_out
    }