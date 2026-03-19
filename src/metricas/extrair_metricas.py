import os
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor

from src.utils import safe_remove, build_class_code_map, zonal_stats_raster
from src.metricas.processamento_celula import process_cell

from src.config import (
    GRADE_PATH, OUTDIR, LAYER_LIST,
    ROAD_PATH, MINING_PATH, HYDRO_PATH,
    INDIGENOUS_PATH, DUMP_PATH,
    POP_RASTER, BUILT_RASTER,
    RASTER_RESOLUTION, CRS_TARGET,
    TEST_N, N_WORKERS, csv_out
)


# ==========================================================
# CARREGAMENTO DE DADOS
# ==========================================================

def carregar_grade(grade_path, crs_target):

    grade = gpd.read_file(grade_path)

    if crs_target:
        grade = grade.to_crs(crs_target)

    if grade.geometry.iloc[0].area < 1e4:
        raise RuntimeError("Grade não está em CRS métrico.")

    return grade


def carregar_camadas(layer_list, grade_crs):

    layer_gdfs = []

    for path, class_col in layer_list:

        gdf = gpd.read_file(path)

        if gdf.crs != grade_crs:
            gdf = gdf.to_crs(grade_crs)

        layer_gdfs.append((gdf, class_col))

    return layer_gdfs


# ==========================================================
# CAMADAS AUXILIARES
# ==========================================================

def carregar_union(path, crs):

    if path and os.path.exists(path):
        gdf = gpd.read_file(path).to_crs(crs)
        return unary_union(gdf.geometry.values)

    return None


def carregar_unions_auxiliares(grade_crs):

    return {
        "road": carregar_union(ROAD_PATH, grade_crs),
        "mining": carregar_union(MINING_PATH, grade_crs),
        "hydro": carregar_union(HYDRO_PATH, grade_crs),
        "indigenous": carregar_union(INDIGENOUS_PATH, grade_crs),
        "dump": carregar_union(DUMP_PATH, grade_crs)
    }


# ==========================================================
# PROCESSAMENTO DAS CÉLULAS (THREADS)
# ==========================================================

def processar_celulas(tasks, layer_gdfs, class_map, resolution, n_workers):

    if n_workers > 1:

        print(f"[THREADS] usando {n_workers} workers")

        def worker(task):
            idx, geom = task
            return process_cell(idx, geom, layer_gdfs, class_map, resolution)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:

            results = list(
                executor.map(worker, tasks)
            )

    else:

        print("[SEQUENCIAL]")

        results = [
            process_cell(idx, geom, layer_gdfs, class_map, resolution)
            for idx, geom in tasks
        ]

    return results


# ==========================================================
# MÉTRICAS EXTRAS
# ==========================================================

def calcular_metricas_extras(df, grade_reset, unions):

    geo_indexed = grade_reset.set_index("cell_index").geometry

    dist_cols = {
        "dist_m_estrada": [],
        "dist_m_mineracao": [],
        "dist_m_hidrografia": [],
        "dist_m_indigena": [],
        "dist_m_lixao": []
    }

    pop_mean, pop_sum = [], []
    built_mean, built_sum = [], []

    for cell_idx in df["cell_index"]:

        geom = geo_indexed.loc[cell_idx]
        cen = geom.centroid

        dist_cols["dist_m_estrada"].append(
            cen.distance(unions["road"]) if unions["road"] else np.nan
        )

        dist_cols["dist_m_mineracao"].append(
            cen.distance(unions["mining"]) if unions["mining"] else np.nan
        )

        dist_cols["dist_m_hidrografia"].append(
            cen.distance(unions["hydro"]) if unions["hydro"] else np.nan
        )

        dist_cols["dist_m_indigena"].append(
            cen.distance(unions["indigenous"]) if unions["indigenous"] else np.nan
        )

        dist_cols["dist_m_lixao"].append(
            cen.distance(unions["dump"]) if unions["dump"] else np.nan
        )

        # Raster stats
        if POP_RASTER and os.path.exists(POP_RASTER):
            stats = zonal_stats_raster(POP_RASTER, geom)
            pop_mean.append(stats.get("mean", np.nan))
            pop_sum.append(stats.get("sum", np.nan))
        else:
            pop_mean.append(np.nan)
            pop_sum.append(np.nan)

        if BUILT_RASTER and os.path.exists(BUILT_RASTER):
            stats = zonal_stats_raster(BUILT_RASTER, geom)
            built_mean.append(stats.get("mean", np.nan))
            built_sum.append(stats.get("sum", np.nan))
        else:
            built_mean.append(np.nan)
            built_sum.append(np.nan)

    for k, v in dist_cols.items():
        df[k] = v

    df["pop_mean"] = pop_mean
    df["pop_sum"] = pop_sum
    df["built_mean"] = built_mean
    df["built_sum"] = built_sum

    return df


# ==========================================================
# SALVAR RESULTADOS
# ==========================================================

def salvar_resultados(df, grade_reset):

    df.to_csv(csv_out, index=False)

    df_indexed = df.set_index("cell_index")

    grade_out = grade_reset.join(df_indexed, on="cell_index")

    gpkg_out = os.path.join(OUTDIR, "grade_2km_metricas_fragstats.gpkg")

    safe_remove(gpkg_out)

    grade_out.to_file(gpkg_out, driver="GPKG")

    return gpkg_out


# ==========================================================
# PIPELINE PRINCIPAL
# ==========================================================

def run_all():

    print("="*80)
    print("PIPELINE FRAGSTATS")
    print("="*80)

    grade = carregar_grade(GRADE_PATH, CRS_TARGET)

    class_map = build_class_code_map(LAYER_LIST)

    layer_gdfs = carregar_camadas(LAYER_LIST, grade.crs)

    unions = carregar_unions_auxiliares(grade.crs)

    grade_reset = grade.reset_index().rename(columns={"index": "cell_index"})

    tasks = [(row.cell_index, row.geometry) for _, row in grade_reset.iterrows()]

    if TEST_N:
        tasks = tasks[:TEST_N]

    results = processar_celulas(
        tasks,
        layer_gdfs,
        class_map,
        RASTER_RESOLUTION,
        N_WORKERS
    )

    df = pd.DataFrame(results).sort_values("cell_index")

    df = calcular_metricas_extras(df, grade_reset, unions)

    gpkg_out = salvar_resultados(df, grade_reset)

    print("CSV:", csv_out)
    print("GPKG:", gpkg_out)

    return df