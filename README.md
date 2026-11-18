# Telco Customer Churn — Grupo 1

Análise exploratória e pré-processamento para os tópicos **3.1 e 3.2** do trabalho de Inteligência de Negócios 2026/2. O projeto contém o código Python, as sete planilhas fornecidas, a base limpa, as cinco partições transformadas e os registros das verificações.

**Escopo:** preparação dos dados. Ainda não foram treinados nem comparados KNN, SVM, Random Forest e XGBoost, exigidos nas etapas seguintes para o Grupo 1. Não há métricas de classificadores neste repositório.

## Resultados verificados

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

Os números decorrem dos arquivos em `data/raw`, não de resultados de um modelo. A razão entre a classe majoritária e a minoritária é de aproximadamente 2,77:1.

## Arquivos principais

```text
src/analise_preprocessamento.py       Código de análise, tratamento e exportação
notebooks/analise_preprocessamento.ipynb  Interface de notebook para o mesmo código
data/raw/                           Sete planilhas originais
data/processed/telco_base_limpa.csv   7.043 linhas, 19 preditores e alvo
data/processed/rastreabilidade.csv   Correspondência entre posição e CustomerID
data/processed/particoes.csv         Partição de validação de cada linha
data/processed/folds/fold_1.npz       Matrizes da primeira partição (há cinco)
data/processed/folds/fold_1_parametros.json  Parâmetros aprendidos no treino
reports/analise.json                 Estatísticas, auditoria e hashes das fontes
reports/dicionario_atributos.json    Tipo e papel de cada atributo original
docs/TRATAMENTOS.md                  Decisões e justificativas dos tratamentos
tests/test_outputs.py               Verificações dos arquivos produzidos
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

Instale as dependências e execute:

```bash
python -m pip install -r requirements.txt
python src/analise_preprocessamento.py
python -m unittest discover -s tests -v
```

`requirements-lock.txt` registra as versões utilizadas na geração conferida, caso seja necessário reproduzir esse ambiente específico. O notebook pode ser aberto no Jupyter ou VS Code, com o mesmo ambiente Python. Não contém dependências de caminhos pessoais absolutos.

Para usar outra pasta com as mesmas sete planilhas ou guardar os resultados em outro diretório:

```bash
python src/analise_preprocessamento.py --data-dir caminho/das/planilhas --output-dir saida
```

A execução gera novamente os arquivos de `data/processed` e `reports` dentro do diretório de saída; não altera as planilhas de entrada. Os testes usam os arquivos entregues na raiz deste repositório.

## Qual base usar

### Base limpa

`data/processed/telco_base_limpa.csv` contém os **19 preditores selecionados e Churn Value**. As categorias estão preservadas, Total Charges foi convertido para numérico e os 11 espaços em branco foram convertidos em ausências.

**Essas 11 ausências são intencionais nesta camada.** Imputar e padronizar toda a base antes da validação cruzada faria a validação influenciar os parâmetros de treino. Portanto, o tratamento dependente de estatísticas é aplicado dentro de cada partição.

### Matrizes transformadas

Os cinco arquivos `fold_N.npz` contêm dados já imputados, codificados e padronizados, sem valores ausentes ou infinitos. Cada arquivo possui:

- `X_train` e `X_valid`: matrizes com 46 atributos, sem alvo ou identificadores.
- `y_train` e `y_valid`: classes 0 e 1.
- `train_indices` e `valid_indices`: posições na base limpa para auditoria.
- `feature_names`: nomes das 46 colunas.

Exemplo de leitura:

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

**Não concatene as matrizes das cinco partições como se fossem uma nova base de clientes.** Os treinos se sobrepõem e possuem estatísticas próprias. Para futuras buscas de hiperparâmetros com divisões internas, ajuste o pré-processamento novamente dentro de cada treino interno, usando a base limpa e um `Pipeline`. Os arquivos exportados atendem somente às partições registradas.

## Tratamentos aplicados

1. Leitura com `keep_default_na=False`, preservando categorias válidas como `None` em Offer.
2. Auditoria de número de registros, tipos, ausências, duplicatas, chaves e constantes nas sete fontes.
3. Uso de `Telco_customer_churn.xlsx` como versão principal, sem misturar valores divergentes das outras versões.
4. Separação de `Churn Value` como alvo e exclusão de identificadores, constantes, campos geográficos fora do recorte e campos que poderiam vazar o desfecho.
5. Conversão de Total Charges para numérico e remoção de espaços nas extremidades das categorias selecionadas.
6. Imputação numérica pela mediana do **treino de cada partição**.
7. `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, mantendo todos os níveis categóricos.
8. `StandardScaler` nas três variáveis numéricas, após imputação. Os indicadores one-hot permanecem em 0/1.
9. Conferência de dimensões, finitude e médias padronizadas. Exportação das matrizes, índices, nomes e parâmetros.

Não foram removidos clientes, aplicados SMOTE ou outro reamostramento, nem criados resultados de classificação. Consulte [todas as decisões em TRATAMENTOS.md](docs/TRATAMENTOS.md).

## Origem e limites

A IBM descreve a amostra como dados de uma empresa **fictícia** de telecomunicações na Califórnia. As fontes e o dicionário estão na [documentação Telco Customer Churn da IBM](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

Foram recebidos arquivos em versões diferentes. Os conjuntos de identificadores coincidem, mas há 551 divergências em Dependents entre a base principal e CustomerChurn, 264 em CEP frente a location e 11 em permanência/cobrança frente a services. Por isso, as planilhas não foram concatenadas nem combinadas silenciosamente.

Os arquivos foram incluídos para reprodução acadêmica com sua atribuição original. Este repositório não reivindica autoria sobre o dataset nem atribui uma nova licença aos materiais da IBM. A publicação pública do repositório não substitui os termos da fonte.

A base é transversal: os experimentos posteriores não demonstrarão, por si só, previsão prospectiva de cancelamento. A disponibilidade temporal das variáveis deve ser considerada antes de qualquer aplicação real. A seleção de variáveis é uma decisão metodológica, não uma comprovação de que as excluídas seriam irrelevantes.

## Referências e uso de IA

- [IBM — Telco customer churn](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
- [Scikit-learn — prevenção de vazamento de informação](https://scikit-learn.org/stable/common_pitfalls.html)
- [Scikit-learn — OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Scikit-learn — StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

Foi utilizado ChatGPT para apoiar a inspeção das fontes, a elaboração e execução do código, a documentação e a organização do repositório. Os resultados registrados foram calculados sobre os arquivos fornecidos. O código foi adaptado da análise inicial para receber caminhos por argumentos e exportar as bases, sem mudar o recorte de 19 preditores.
