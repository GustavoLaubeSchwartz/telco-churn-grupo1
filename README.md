# Telco Customer Churn — Grupo 1

Trabalho prático da disciplina de Inteligência de Negócios 2026/2 sobre previsão de cancelamento de clientes (*churn*) com a base Telco Customer Churn. O repositório contém todo o pipeline desenvolvido pelo grupo: análise exploratória, pré-processamento sem vazamento de dados, particionamento estratificado e o treinamento/avaliação dos quatro classificadores do Grupo 1: **KNN, SVM, Random Forest e XGBoost**.

## Resultados verificados

### Preparação dos dados (Itens 3.1 e 3.2)

| Verificação | Resultado |
|---|---:|
| Clientes na base principal | 7.043 |
| Atributos originais | 33 |
| Preditores selecionados | 19: 3 numéricos e 16 categóricos |
| Variáveis após transformação em cada partição | 46: 3 padronizadas e 43 indicadores |
| Cancelamento, classe 1 | 1.869 (26,54%) |
| Não cancelamento, classe 0 | 5.174 (73,46%) |
| Ausências em Total Charges | 11 |
| Linhas integralmente duplicadas na base principal | 0 |
| Partições estratificadas | 5 |

A razão entre a classe majoritária e a minoritária é de aproximadamente 2,77:1.

### Desempenho dos modelos (Itens 3.3 a 3.6)

Resultados obtidos na validação cruzada estratificada em 5 partições (média ± desvio-padrão):

| Modelo / Configuração | Acurácia | Precisão | Revocação | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **KNN (A: k=5)** | 0,7789 ± 0,0114 | 0,5875 ± 0,0249 | 0,5645 ± 0,0252 | 0,5754 ± 0,0189 |
| **KNN (B: k=11)** | 0,7907 ± 0,0133 | 0,6118 ± 0,0259 | 0,5784 ± 0,0316 | 0,5945 ± 0,0271 |
| **SVM (A: Linear)** | 0,8063 ± 0,0097 | 0,6592 ± 0,0210 | 0,5597 ± 0,0248 | 0,6052 ± 0,0211 |
| **SVM (B: RBF)** | 0,8090 ± 0,0085 | 0,6824 ± 0,0238 | 0,5254 ± 0,0172 | 0,5935 ± 0,0168 |
| **RF (A: Baseline, 300 árvores)** | 0,7964 ± 0,0073 | 0,6489 ± 0,0188 | 0,5083 ± 0,0308 | 0,5696 ± 0,0210 |
| **RF (B: Regularizado, max_depth=12)** | 0,8093 ± 0,0085 | 0,6743 ± 0,0211 | 0,5452 ± 0,0246 | 0,6027 ± 0,0190 |
| **XGBoost (A: Baseline)** | 0,8079 ± 0,0096 | 0,6649 ± 0,0248 | 0,5581 ± 0,0218 | 0,6065 ± 0,0186 |
| **XGBoost (B: Regularizado)** | 0,8055 ± 0,0096 | 0,6613 ± 0,0250 | 0,5490 ± 0,0253 | 0,5995 ± 0,0199 |

*Linha de base da classe majoritária: 73,46% de acurácia.*

## Arquivos principais

```text
notebooks/experimentos_modelos.ipynb      Treinamento e avaliação de KNN, SVM, RF e XGBoost
notebooks/analise_preprocessamento.ipynb  Notebook de análise exploratória e pré-processamento
src/analise_preprocessamento.py           Código de análise, tratamento e exportação dos folds
src/gerar_graficos_relatorio.py           Script de geração das figuras e matrizes de confusão
data/raw/                                Sete planilhas originais fornecidas
data/processed/telco_base_limpa.csv       Base limpa com 19 preditores e alvo (7.043 linhas)
data/processed/rastreabilidade.csv       Correspondência entre posição de linha e CustomerID
data/processed/particoes.csv             Partição de validação atribuída a cada cliente
data/processed/folds/fold_N.npz           Matrizes pré-processadas de cada partição (1 a 5)
data/processed/folds/fold_N_parametros.json Parâmetros aprendidos no treino de cada partição
reports/modelos/                         Resultados em CSV por fold, resumo e matrizes
reports/modelos/fig_comparativo_metricas.png Gráfico comparativo de desempenho dos modelos
reports/modelos/fig_matrizes_confusao.png    Visualização das 8 matrizes de confusão
reports/analise.json                     Estatísticas detalhadas e hashes das fontes
reports/dicionario_atributos.json        Tipo e papel de cada atributo original
docs/TRATAMENTOS.md                      Decisões e justificativas dos tratamentos
tests/test_outputs.py                   Verificações automatizadas dos arquivos gerados
```

Os CSVs usam **UTF-8, vírgula como delimitador e ponto como separador decimal**. Células vazias representam ausência. `row_index` é a posição de linha, iniciando em zero, na base limpa; não deve ser incluído como preditor.

## Como executar

Recomendado: Python 3.12. Na raiz do repositório:

```bash
python -m venv .venv
```

Ative o ambiente:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux ou macOS
source .venv/bin/activate
```

Instale as dependências e execute os passos necessários:

```bash
# 1. Instalar dependências
python -m pip install -r requirements.txt

# 2. Gerar a base tratada e os folds (opcional se já clonados)
python src/analise_preprocessamento.py

# 3. Rodar testes automatizados de integridade
python -m unittest discover -s tests -v

# 4. Gerar os gráficos e matrizes de confusão
python src/gerar_graficos_relatorio.py
```

O treinamento e avaliação de todos os classificadores está documentado passo a passo no notebook `notebooks/experimentos_modelos.ipynb`, que pode ser executado no Jupyter Notebook ou VS Code.

## Qual base usar

### Base limpa

`data/processed/telco_base_limpa.csv` contém os **19 preditores selecionados e Churn Value**. As categorias estão preservadas, Total Charges foi convertido para numérico e os 11 espaços em branco foram convertidos em ausências.

**Essas 11 ausências são intencionais nesta camada.** Imputar e padronizar toda a base antes da validação cruzada faria a validação influenciar os parâmetros de treino (vazamento de dados). Portanto, o tratamento dependente de estatísticas é aplicado estritamente dentro de cada partição.

### Matrizes transformadas

Os cinco arquivos `fold_N.npz` contêm dados já imputados, codificados e padronizados, sem valores ausentes ou infinitos. Cada arquivo possui:

- `X_train` e `X_valid`: matrizes com 46 atributos numéricos, sem alvo ou identificadores.
- `y_train` e `y_valid`: classes 0 e 1.
- `train_indices` e `valid_indices`: posições na base limpa para auditoria.
- `feature_names`: nomes das 46 colunas após codificação one-hot e padronização.

Exemplo de leitura em Python:

```python
import numpy as np

with np.load("data/processed/folds/fold_1.npz", allow_pickle=False) as dados:
    X_train = dados["X_train"]
    X_valid = dados["X_valid"]
    y_train = dados["y_train"]
    y_valid = dados["y_valid"]
    nomes = dados["feature_names"]
```

Cada cliente participa da validação exatamente uma vez. As três primeiras partições possuem 5.634 clientes de treino e 1.409 de validação; as duas últimas, 5.635 e 1.408. Foi usado `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

## Tratamentos aplicados

1. Leitura com `keep_default_na=False`, preservando categorias válidas como `None` em Offer.
2. Auditoria de número de registros, tipos, ausências, duplicatas, chaves e constantes nas sete fontes.
3. Uso de `Telco_customer_churn.xlsx` como versão principal, sem misturar valores divergentes das outras versões.
4. Separação de `Churn Value` como alvo e exclusão de identificadores, constantes, campos geográficos fora do recorte e campos com potencial de vazamento de desfecho.
5. Conversão de Total Charges para numérico e remoção de espaços nas extremidades das categorias selecionadas.
6. Imputação numérica pela mediana do **treino de cada partição**.
7. `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, mantendo todos os níveis categóricos.
8. `StandardScaler` nas três variáveis numéricas, após imputação. Os indicadores one-hot permanecem em 0/1.
9. Exportação das matrizes transformadas das 5 partições para treinamento e avaliação dos classificadores.

Consulte [todas as decisões em TRATAMENTOS.md](docs/TRATAMENTOS.md).

## Origem e limites

A IBM descreve a amostra como dados de uma empresa **fictícia** de telecomunicações na Califórnia. As fontes e o dicionário estão na [documentação Telco Customer Churn da IBM](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

A base é transversal: os experimentos realizados não comprovam previsão prospectiva de cancelamento, servindo para comparação metodológica dos algoritmos sob as mesmas condições experimentais.

## Referências e uso de IA

- [IBM — Telco customer churn](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
- [Scikit-learn — Prevenção de vazamento de informação](https://scikit-learn.org/stable/common_pitfalls.html)
- [Scikit-learn — OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Scikit-learn — StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

Foi utilizado ChatGPT como ferramenta auxiliar para apoio à inspeção dos dados, elaboração do código de análise/modelagem, documentação e organização do projeto. Todos os cálculos, matrizes e métricas foram gerados a partir da execução do código sobre as bases fornecidas.
