"""Script principal do pipeline geoespacial.

Etapas:
1) Geração de grade celular
2) Extração de métricas espaciais
3) Validação estilo Fragstats"""
from mapeamento import run_mapping
from verificacao_metricas import OUT_PATH
from config import GRADE_PATH
import pandas as pd
from espacocelulargerado import (load_limite, create_grid, clip_grid, save_grid, plot_grid)
from extrair_metricas import run_all
from verificacao_metricas import validate_shape, CSV_PATH, OUT_PATH, EPS
from config import (GRADE_PATH, OUTDIR, LAYER_LIST, ROAD_PATH, MINING_PATH, RASTER_RESOLUTION,
    CRS_TARGET, TEST_N, N_WORKERS, HYDRO_PATH, INDIGENOUS_PATH, DUMP_PATH, POP_RASTER, BUILT_RASTER)


# =========================================================
# ETAPA 1 — GRADE
# =========================================================
def gerar_grade():
    print("\n[ETAPA 1] Gerando grade celular...")

    limite = load_limite("input/Limite_oiapoque_certo_dissolv.gpkg")

    grid = create_grid(limite, cell_size=2000)
    grid_clip = clip_grid(grid, limite)

    save_grid(grid_clip, "grade_2km_oiapoque.gpkg")
    plot_grid(limite, grid_clip)

    print("Grade gerada com sucesso.\n")


# =========================================================
# ETAPA 2 — MÉTRICAS
# =========================================================
def extrair_metricas():
    print("\n[ETAPA 2] Extraindo métricas espaciais...")

    out = run_all(
        layer_list=LAYER_LIST,
        grade_path=GRADE_PATH,
        outdir=OUTDIR,
        resolution=RASTER_RESOLUTION,
        test_n=TEST_N,
        crs_target=CRS_TARGET,
        road_path=ROAD_PATH,
        mining_path=MINING_PATH,
        n_workers=N_WORKERS,
        hydro_path=HYDRO_PATH,
        indigenous_path=INDIGENOUS_PATH,
        dump_path=DUMP_PATH,
        pop_raster=POP_RASTER,
        built_raster=BUILT_RASTER,
    )

    print("Extração finalizada.")
    print("CSV:", out["csv"])
    print("GPKG:", out["gpkg"])
    return out


# =========================================================
# ETAPA 3 — VALIDAÇÃO
# =========================================================
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
    df.to_csv(OUT_PATH, index=False)
    print(f"\nRelatório salvo em: {OUT_PATH}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("PIPELINE GEOESPACIAL")
    print("=" * 80)

    try:
        print("\n[1/4] Gerando grade...")
        gerar_grade()
        print("[OK] Grade gerada com sucesso.")

    except Exception as e:
        print("[ERRO] Falha ao gerar grade.")
        print("Detalhes:", e)
        return

    try:
        print("\n[2/4] Extraindo métricas da paisagem...")
        extrair_metricas()
        print("[OK] Métricas extraídas com sucesso.")

    except Exception as e:
        print("[ERRO] Falha na extração de métricas.")
        print("Detalhes:", e)
        return

    try:
        print("\n[3/4] Validando métricas...")
        validar_metricas()
        print("[OK] Métricas validadas.")

    except Exception as e:
        print("[ERRO] Falha na validação das métricas.")
        print("Detalhes:", e)
        return

    try:
        print("\n[4/4] Gerando mapa de clusters...")
        print("CSV:", OUT_PATH)
        print("GRID:", GRADE_PATH)
        print("OUTPUT:", "mapa_cluster.gpkg")

        run_mapping(
            csv_path=OUT_PATH,
            grid_path=GRADE_PATH,
            out_path="mapa_cluster.gpkg"
        )

        print("[OK] Mapa gerado com sucesso.")

    except Exception as e:
        print("[ERRO] Falha na geração do mapa.")
        print("Detalhes:", e)
        return

    print("\nPipeline finalizado com sucesso.")


if __name__ == "__main__":
    main()