""" Classificação supervisionada (C5-like + Boosting usando TODAS as features)
    COM LOGS PARA ACOMPANHAMENTO
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def preparar_dados(df, target_col):
    print("\n[ETAPA] Preparando dados...")

    # cria cópia para não alterar o original
    df = df.copy()

    print(f"Shape original: {df.shape}")

    # remove linhas onde a variável alvo (target) é nula
    df = df.dropna(subset=[target_col])
    print(f"Após remover NA no target ({target_col}): {df.shape}")

    # separa variáveis explicativas (X) removendo a coluna alvo
    X = df.drop(columns=[target_col])

    # mantém apenas colunas numéricas (modelos do sklearn precisam disso)
    X = X.select_dtypes(include=["number"])
    print(f"Número de features numéricas: {X.shape[1]}")

    # remove colunas completamente vazias
    X = X.dropna(axis=1, how="all")
    print(f"Após remover colunas vazias: {X.shape[1]} features")

    # remove linhas que ainda possuem algum NA
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = df.loc[mask, target_col]

    print(f"Shape final após limpeza: {X.shape}")

    # retorna:
    # X -> features
    # y -> target
    # df_limpo -> dataframe correspondente (alinhado com X)
    return X, y, df.loc[mask].copy()


def run_c5_boost(caminho_csv, target_col="classe"):

    print("\n===================================")
    print("INICIANDO CLASSIFICAÇÃO C5 + BOOST")
    print("===================================")

    # mostra qual arquivo está sendo usado
    print(f"\n[INFO] Lendo CSV: {caminho_csv}")

    # carregar dataset
    df = pd.read_csv(caminho_csv)

    print(f"[INFO] Dataset carregado com shape: {df.shape}")
    print(f"[INFO] Coluna alvo: {target_col}")

    # preparar dados
    X, y, df_limpo = preparar_dados(df, target_col)

    print("\n[ETAPA] Normalização dos dados...")

    # normalização (importante para boosting funcionar melhor)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[OK] Dados normalizados")

    print("\n[ETAPA] Divisão treino/teste...")

    # divide em treino (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.3,
        random_state=42,
        stratify=y  # mantém proporção das classes
    )

    print(f"Tamanho treino: {X_train.shape}")
    print(f"Tamanho teste: {X_test.shape}")

    print("\n[ETAPA] Criando modelo base (árvore)...")

    # árvore de decisão (aproximação do C4.5)
    base_tree = DecisionTreeClassifier(
        max_depth=5,          # limita profundidade (evita overfitting)
        min_samples_leaf=10,  # mínimo de amostras por folha
        random_state=42
    )

    print("[OK] Árvore criada")

    print("\n[ETAPA] Aplicando Boosting...")

    # AdaBoost = várias árvores fracas combinadas
    model = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=100,   # número de árvores
        learning_rate=0.1,
        random_state=42
    )

    print("[OK] Boosting configurado")

    print("\n[ETAPA] Treinando modelo...")

    # treino do modelo
    model.fit(X_train, y_train)

    print("[OK] Modelo treinado")

    print("\n[ETAPA] Avaliação...")

    # previsão no conjunto de teste
    y_pred = model.predict(X_test)

    print("\n=== RESULTADOS ===")
    print(classification_report(y_test, y_pred))

    print("\n[ETAPA] Gerando predição para TODO o dataset...")

    # previsão para todo o dataset limpo
    df_limpo["classe_predita"] = model.predict(X_scaled)

    print("[OK] Predição concluída")

    print("\n===================================")
    print("PROCESSO FINALIZADO")
    print("===================================")

    # retorna:
    # df_limpo -> com a coluna nova 'classe_predita'
    # model -> modelo treinado
    return df_limpo, model