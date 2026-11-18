# Tratamentos e decisões metodológicas

## Fonte principal e auditoria

Unidade de análise: um cliente. Fonte principal: `Telco_customer_churn.xlsx`, 7.043 clientes e 33 atributos. Nenhuma planilha foi alterada. Os hashes SHA-256 das fontes estão em `reports/analise.json`.

| Arquivo | Registros | Colunas | Papel |
|---|---:|---:|---|
| Telco_customer_churn.xlsx | 7.043 | 33 | Fonte principal |
| CustomerChurn.xlsx | 7.043 | 21 | Auditoria de outra versão |
| Telco_customer_churn_demographics.xlsx | 7.043 | 9 | Auditoria |
| Telco_customer_churn_location.xlsx | 7.043 | 10 | Auditoria |
| Telco_customer_churn_services.xlsx | 7.043 | 31 | Auditoria |
| Telco_customer_churn_status.xlsx | 7.043 | 12 | Auditoria |
| Telco_customer_churn_population.xlsx | 1.671 | 3 | Áreas de CEP; não são clientes |

Todos os arquivos apresentam zero linhas integralmente duplicadas. As chaves de cliente são únicas em suas tabelas; Zip Code é único em population. As divergências entre versões são registradas no relatório e não foram corrigidas por substituição arbitrária. A base principal tem zero duplicatas mesmo quando CustomerID é desconsiderado.

## Alvo

`Churn Value`: 0 = não cancelamento (5.174); 1 = cancelamento (1.869). Churn Label coincide com essa representação em todas as linhas. O alvo não entra em `X`.

## Preditores mantidos

**Numéricos:** Tenure Months, Monthly Charges, Total Charges.

**Categóricos:** Gender, Senior Citizen, Partner, Dependents, Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies, Contract, Paperless Billing e Payment Method.

São 19 preditores antes da codificação. Gender e os demais campos de perfil são atributos do dataset didático; nenhum resultado individual é usado para decisões sobre pessoas reais.

## Exclusões e justificativas

| Campo | Decisão |
|---|---|
| CustomerID | Identificador sem função preditiva; guardado separadamente para rastreabilidade |
| Count | Constante igual a 1 |
| Country e State | Constantes |
| City e Zip Code | Fora do recorte inicial, evitando cardinalidade elevada; CEP é código, não distância |
| Lat Long | Representação textual redundante das coordenadas |
| Latitude e Longitude | Fora do recorte geográfico inicial |
| Churn Label | Duplicata categórica do alvo |
| Churn Reason | Informação sobre a saída; presença/ausência revela o desfecho |
| Churn Score | Saída de outro modelo; não deve substituir o problema de classificação |
| CLTV | Exclusão conservadora: fórmula e disponibilidade temporal não verificadas |

Customer Status, Churn Category, Satisfaction Score e identificadores Location ID, Service ID, Status ID e LoyaltyID não foram incorporados, porque pertencem a fontes auxiliares que não foram integradas. Sua ausência da matriz não implica que tenham sido colunas removidas da base principal.

## Ausências e inconsistências

- Leitura com `keep_default_na=False` para não confundir a categoria textual `None` com ausência.
- Auditoria de células vazias, NaN e strings contendo somente espaços em todas as fontes.
- Total Charges: 11 espaços em branco convertidos em NaN; todos associados a Tenure Months = 0. Não se assumiu cobrança acumulada zero.
- Churn Reason: 5.174 vazios estruturais na base principal, todos em não cancelados. A coluna foi excluída, e não imputada para uso preditivo.
- CustomerChurn apresenta os mesmos 11 vazios em Total Charges. Status apresenta 5.174 vazios em Churn Category e Churn Reason.
- Não foram encontrados outros vazios nas fontes auditadas, sob essa definição de ausência.
- As categorias selecionadas receberam `str.strip()`. Ausências categóricas futuras são previstas com imputação pela moda do treino; nenhum preenchimento categórico foi necessário no conjunto atual.
- Não foram encontrados valores negativos nas três variáveis numéricas selecionadas.
- No phone service e No internet service foram mantidos como categorias válidas e conferidos contra o serviço principal correspondente. Não houve divergências nessas condições.
- Não foram removidos extremos apenas pelo tamanho do valor, nem excluídos clientes.

## Codificação e escala

OneHotEncoder com todas as categorias e `handle_unknown="ignore"`. Categorias inéditas na validação geram zeros no bloco correspondente, sem aprender com a validação. Não foi atribuída ordem numérica artificial às categorias.

As três variáveis numéricas passam por mediana e StandardScaler: `z = (x - media_treino) / desvio_treino`. É padronização, não escala min-max entre zero e um. O desvio usado pelo StandardScaler corresponde a `ddof=0`. Não se padronizam os 43 indicadores one-hot.

## Separação e artefatos

Cinco partições estratificadas, embaralhadas, semente 42. Cada pré-processador é ajustado de novo no treino de sua partição. As medianas de Total Charges foram 1.387,40; 1.388,45; 1.414,45; 1.412,40; 1.390,60, respectivamente. Não existe uma mediana global aplicada a todos os folds.

`telco_base_limpa.csv` preserva as 11 ausências para permitir novas divisões sem contaminação. `fold_N.npz` contém as matrizes efetivamente transformadas e finitas. O arquivo `fold_N_parametros.json` registra medianas, médias, escalas, categorias e nomes das colunas. O formato NPZ dispensa pickle na leitura.

Não se deve usar a concatenação dos folds como base única. Para ajuste adicional por validação interna, o pipeline deve ser reaprendido em cada divisão interna. Reamostramento, se adotado posteriormente, também deve ocorrer somente no treino. Os cinco splits registrados devem ser compartilhados pelos classificadores a comparar.

## Verificações executadas

O script confere valores finitos e médias aproximadamente zero nas variáveis padronizadas do treino. Os testes adicionais verificam as contagens, a integridade das fontes por hash, as classes, a ausência de interseção entre treino/validação, a cobertura única de cada cliente em validação, a reprodução independente de imputação/escala e a consistência dos rótulos e índices exportados.

Estas verificações dizem respeito à preparação. Não foram avaliados classificadores, hiperparâmetros, acurácia, precisão, recall ou F1-score de modelos treinados.
