"""
Classificação supervisionada (C5-like + Boosting) usando métricas de fragstats
e avaliando desempenho com o CSV de tipologia_classificada manualmente.
Gera também uma imagem colorida para visualização das classes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# mapa de cores para visualização
COLOR_MAP = {
    "Risco_Mode": "red",
    "Risco_Baixo": "green",
    "Risco_Aloc": "orange",
    "Risco_Auto": "blue"
}

def preparar_dados(df_features, df_classes, target_col):
    print("\n[ETAPA] Preparando dados...")

    # 🔗 merge entre features e classes
    df = df_features.merge(df_classes, on="id")  # ajuste "id" se necessário

    print(f"Shape após merge: {df.shape}")

    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=["number"])

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    mask = X.notna().all(axis=1)
    X = X[mask]
    y = df.loc[mask, target_col]

    print(f"Shape final após limpeza: {X.shape}")

    return X, y, df.loc[mask].copy()

def run_c5_boost(features_csv, classes_csv, target_col="classv0"):
    print("\n===================================")
    print("INICIANDO CLASSIFICAÇÃO C5 + BOOST")
    print("===================================")

    df_features = pd.read_csv(features_csv)
    df_classes = pd.read_csv(classes_csv)

    # preparar dados
    X, y, df_limpo = preparar_dados(df_features, df_classes, target_col)

    # normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # converte target para números
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # árvore base (C5-like)
    base_tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)

    # AdaBoost
    model = AdaBoostClassifier(estimator=base_tree, n_estimators=100,
                               learning_rate=0.1, random_state=42)

    # treino
    model.fit(X_train, y_train)

    # avaliação
    y_pred_test = model.predict(X_test)
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred_test)

    print("\n=== RESULTADOS ===")
    print(classification_report(y_test_labels, y_pred_labels))
    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_))

    # predição para todo dataset
    df_limpo["classe_predita"] = le.inverse_transform(model.predict(X_scaled))

    # salvar CSV final
    df_limpo.to_csv(r"C:\Users\dudda\Downloads\remote-sensing-of-Oiapoque\output\metricas_com_predicao.csv", index=False)
    print("\nCSV com predições salvo!")

    # criar visualização
    plt.figure(figsize=(12, 8))
    colors = df_limpo["classe_predita"].map(COLOR_MAP)
    plt.scatter(df_limpo["cell_index"], np.zeros_like(df_limpo["cell_index"]), c=colors, s=50)
    # criar legenda
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cls,
                          markerfacecolor=color, markersize=10)
               for cls, color in COLOR_MAP.items()]
    plt.legend(handles=handles, title="Classe predita")
    plt.title("Classificação Células - Predição")
    plt.xlabel("cell_index")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(r"C:\Users\dudda\Downloads\remote-sensing-of-Oiapoque\output\mapa_predicao.png")
    plt.show()
    print("Mapa de predição salvo!")

    return df_limpo, model


