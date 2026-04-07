from src.config import (CSV_PATH, CLASSES_CSV)
from src.grade_celular.geracao_grade import gerar_grade
from src.metricas.extrair_metricas import extrair_metricas
from src.metricas.verificacao_metricas import validar_metricas
from src.mapeamento.c5_boost import run_c5_boost

features_csv = r"C:\Users\dudda\Downloads\remote-sensing-of-Oiapoque\output\metricas_fragstats_por_celula.csv"
classes_csv = r"C:\Users\dudda\Downloads\remote-sensing-of-Oiapoque\input\tipologia_classificada.csv"

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

    print("\n[4/4] Gerando a classificação...")
    run_c5_boost(features_csv, classes_csv, target_col="classv0")
    print("[OK] Mapa gerado com sucesso.")

    print("\nPipeline finalizado com sucesso.")


if __name__ == "__main__":
    main()