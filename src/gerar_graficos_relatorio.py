import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gerar_grafico_matrizes():
    caminho_base = os.path.join(os.path.dirname(__file__), "..", "reports", "modelos", "matrizes_confusao")
    arquivos = [
        ("KNN_A_5_Vizinhos.csv", "KNN (k=5)"),
        ("KNN_B_11_Vizinhos.csv", "KNN (k=11)"),
        ("SVM_A_Linear.csv", "SVM (Linear)"),
        ("SVM_B_RBF.csv", "SVM (RBF)"),
        ("RF_A_Baseline.csv", "Random Forest (Baseline)"),
        ("RF_B_Regularized.csv", "Random Forest (Regularizado)"),
        ("XGB_A_Baseline.csv", "XGBoost (Baseline)"),
        ("XGB_B_Regularized.csv", "XGBoost (Regularizado)")
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=300)
    axes = axes.flatten()

    for idx, (arq, titulo) in enumerate(arquivos):
        path = os.path.join(caminho_base, arq)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        matrix = df.values.astype(int)

        ax = axes[idx]
        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

        # Rótulos
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Não Churn (0)", "Churn (1)"], fontsize=9)
        ax.set_yticklabels(["Não Churn (0)", "Churn (1)"], fontsize=9)
        ax.set_title(titulo, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predição", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)

        # Valores internos com percentual sobre o total
        total = matrix.sum()
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                pct = (val / total) * 100
                color = "white" if val > matrix.max() / 2 else "black"
                ax.text(j, i, f"{val:,}\n({pct:.1f}%)".replace(",", "."),
                        ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "..", "reports", "modelos", "fig_matrizes_confusao.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {output_path}")

def gerar_grafico_metricas():
    # Coletar resumos
    caminho_modelos = os.path.join(os.path.dirname(__file__), "..", "reports", "modelos")

    dados = []

    # KNN
    df_knn = pd.read_csv(os.path.join(caminho_modelos, "knn_resumo.csv"))
    for _, row in df_knn.iterrows():
        dados.append({
            "modelo": "KNN (k=5)" if "5" in row["configuration"] else "KNN (k=11)",
            "acc": row["accuracy_mean"], "acc_std": row["accuracy_std"],
            "prec": row["precision_mean"], "prec_std": row["precision_std"],
            "rec": row["recall_mean"], "rec_std": row["recall_std"],
            "f1": row["f1_score_mean"], "f1_std": row["f1_score_std"],
        })

    # SVM
    df_svm = pd.read_csv(os.path.join(caminho_modelos, "svm_resumo.csv"))
    for _, row in df_svm.iterrows():
        dados.append({
            "modelo": "SVM (Linear)" if "Linear" in row["configuration"] else "SVM (RBF)",
            "acc": row["accuracy_mean"], "acc_std": row["accuracy_std"],
            "prec": row["precision_mean"], "prec_std": row["precision_std"],
            "rec": row["recall_mean"], "rec_std": row["recall_std"],
            "f1": row["f1_score_mean"], "f1_std": row["f1_score_std"],
        })

    # RF
    df_rf = pd.read_csv(os.path.join(caminho_modelos, "random_forest_resumo.csv"))
    for _, row in df_rf.iterrows():
        # Handle col names which may have accents
        cfg = row.iloc[0]
        dados.append({
            "modelo": "RF (Baseline)" if "Baseline" in cfg else "RF (Regularizado)",
            "acc": row.iloc[1], "acc_std": row.iloc[2],
            "prec": row.iloc[3], "prec_std": row.iloc[4],
            "rec": row.iloc[5], "rec_std": row.iloc[6],
            "f1": row.iloc[7], "f1_std": row.iloc[8],
        })

    # XGBoost
    df_xgb = pd.read_csv(os.path.join(caminho_modelos, "xgboost_resumo.csv"))
    for _, row in df_xgb.iterrows():
        dados.append({
            "modelo": "XGBoost (Baseline)" if "Baseline" in row["configuration"] else "XGBoost (Regularizado)",
            "acc": row["accuracy_mean"], "acc_std": row["accuracy_std"],
            "prec": row["precision_mean"], "prec_std": row["precision_std"],
            "rec": row["recall_mean"], "rec_std": row["recall_std"],
            "f1": row["f1_score_mean"], "f1_std": row["f1_score_std"],
        })

    df_res = pd.DataFrame(dados)

    # Gráfico de barras comparativo
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    x = np.arange(len(df_res))
    width = 0.2

    ax.bar(x - 1.5*width, df_res["acc"], width, yerr=df_res["acc_std"], capsize=3, label="Acurácia", color="#1f77b4")
    ax.bar(x - 0.5*width, df_res["prec"], width, yerr=df_res["prec_std"], capsize=3, label="Precisão", color="#ff7f0e")
    ax.bar(x + 0.5*width, df_res["rec"], width, yerr=df_res["rec_std"], capsize=3, label="Revocação", color="#2ca02c")
    ax.bar(x + 1.5*width, df_res["f1"], width, yerr=df_res["f1_std"], capsize=3, label="F1-Score", color="#d62728")

    # Linha base da classe majoritária (73,46%)
    ax.axhline(0.7346, color="gray", linestyle="--", linewidth=1.2, label="Linha de Base Majoritária (73,46%)")

    ax.set_ylabel("Pontuação (0 a 1)", fontsize=11)
    ax.set_title("Comparação de Desempenho dos Classificadores — Telco Customer Churn (Média ± Desvio)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["modelo"], rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0.4, 0.9)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "..", "reports", "modelos", "fig_comparativo_metricas.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {output_path}")

if __name__ == "__main__":
    gerar_grafico_matrizes()
    gerar_grafico_metricas()
