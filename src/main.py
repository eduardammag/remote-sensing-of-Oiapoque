from src.config import (GRADE_PATH, OUTDIR)
from src.grade_celular.geracao_grade import gerar_grade
from src.metricas.extrair_metricas import extrair_metricas
from src.metricas.verificacao_metricas import validar_metricas
from src.mapeamento.kmeans_pca import run_clusters


# MAIN

def main():
    print("PIPELINE GEOESPACIAL")

    print("\n[1/4] Gerando grade...")
    gerar_grade()
    print("[OK] Grade gerada com sucesso.")

    print("\n[2/4] Extraindo métricas da paisagem...")
    extrair_metricas()
    print("[OK] Métricas extraídas com sucesso.")


    print("\n[3/4] Validando métricas...")
    validar_metricas()
    print("[OK] Métricas validadas.")

    print("\n[4/4] Gerando mapa de clusters...")
    run_clusters(csv_path= OUTDIR, grid_path= GRADE_PATH,out_path="mapa_cluster.gpkg")
    print("[OK] Mapa gerado com sucesso.")

    print("\nPipeline finalizado com sucesso.")


if __name__ == "__main__":
    main()