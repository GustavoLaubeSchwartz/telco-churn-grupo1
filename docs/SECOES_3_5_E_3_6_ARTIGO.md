# Seções 3.5 e 3.6 — Artigo Científico

Este documento contém o texto das seções 3.5 e 3.6 redigido em tom direto, sem rodeios ou excesso de termos artificiais, refletindo uma postura natural de estudante de graduação e alinhado exatamente ao código executado pelo grupo no repositório.

Para quem for colar direto no Overleaf, o código LaTeX formatado está em `docs/secoes_3_5_e_3_6_overleaf.tex`.

---

## 3.5 Validação cruzada e ajuste de hiperparâmetros

Para avaliar os quatro modelos de forma justa e reproduzível, utilizamos uma validação cruzada estratificada em cinco partições (*Stratified 5-Fold Cross-Validation*) com semente aleatória (`random_state=42`) e embaralhamento dos dados (`shuffle=True`). O uso de estratificação foi indispensável para preservar a proporção de 26,54% de clientes que cancelaram o serviço em todas as partições de treino e teste.

A fim de evitar qualquer vazamento de dados (*data leakage*), todas as etapas de pré-processamento definidas anteriormente — imputação da mediana nos valores ausentes numéricos, codificação *One-Hot* das variáveis categóricas e padronização com *StandardScaler* — foram ajustadas estritamente sobre os dados de treino de cada partição, aplicando-se apenas a transformação no conjunto de validação.

Em conformidade com os requisitos da disciplina para o Grupo 1, foram avaliados quatro classificadores: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest e XGBoost. Para cada um, testamos duas configurações distintas:

* **KNN:** Na configuração A, utilizamos $k=5$ vizinhos, ponderação uniforme e distância euclidiana. Na configuração B, ampliamos para $k=11$ vizinhos mantendo os demais parâmetros, com o intuito de verificar se uma vizinhança maior conseguiria atenuar o ruído da alta dimensionalidade (46 atributos após o *One-Hot*).
* **SVM:** Na configuração A, aplicamos o kernel linear com $C=1,0$, estabelecendo uma fronteira direta de decisão. Na configuração B, utilizamos o kernel RBF com $C=1,0$ e `gamma='scale'`, buscando identificar padrões não lineares entre as variáveis.
* **Random Forest:** A configuração A foi um *baseline* com 300 árvores sem restrição de profundidade (`max_depth=None`). Já a configuração B manteve as 300 árvores, mas adicionou regularização limitando a profundidade a 12 níveis (`max_depth=12`) e exigindo pelo menos 2 amostras por folha (`min_samples_leaf=2`), visando conter o sobreajuste (*overfitting*).
* **XGBoost:** Na configuração A (*baseline*), usamos 200 árvores, profundidade máxima 3, taxa de aprendizado de 0,10, subamostragem de 80% de linhas e colunas e regularização L2 padrão ($\lambda=1$). Na configuração B, aumentamos para 400 árvores, reduzindo a taxa de aprendizado para 0,05 e elevando a regularização L2 para $\lambda=2$, testando se um aprendizado mais conservador traria ganho de generalização.

---

## 3.6 Métricas de avaliação e análise dos resultados

Como a base Telco Customer Churn possui desbalanceamento (73,46% de não cancelamentos contra 26,54% de cancelamentos), a acurácia isolada é uma métrica insuficiente: um classificador ingênuo que simplesmente previsse que nenhum cliente cancelaria alcançaria 73,46% de acerto sem detectar nenhum caso real. Dessa forma, consideramos também Precisão, Revocação (*Recall*), F1-Score e as matrizes de confusão consolidadas ao longo das cinco partições.

A Tabela I apresenta a média e o desvio-padrão de cada métrica obtida nas cinco partições. Já a Tabela II consolida a contagem total de acertos e erros sobre todos os 7.043 clientes da base (sendo 5.174 da classe negativa e 1.869 da classe positiva).

**TABELA I — Desempenho dos classificadores na validação cruzada de 5 partições (média ± desvio-padrão)**

| Modelo / Configuração | Acurácia | Precisão | Revocação | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **KNN ($k=5$)** | $0{,}7789 \pm 0{,}0114$ | $0{,}5875 \pm 0{,}0249$ | $0{,}5645 \pm 0{,}0252$ | $0{,}5754 \pm 0{,}0189$ |
| **KNN ($k=11$)** | $0{,}7907 \pm 0{,}0133$ | $0{,}6118 \pm 0{,}0259$ | $0{,}5784 \pm 0{,}0316$ | $0{,}5945 \pm 0{,}0271$ |
| **SVM (Linear)** | $0{,}8063 \pm 0{,}0097$ | $0{,}6592 \pm 0{,}0210$ | $0{,}5597 \pm 0{,}0248$ | $0{,}6052 \pm 0{,}0211$ |
| **SVM (RBF)** | $0{,}8090 \pm 0{,}0085$ | **$0{,}6824 \pm 0{,}0238$** | $0{,}5254 \pm 0{,}0172$ | $0{,}5935 \pm 0{,}0168$ |
| **RF (Baseline, $n=300$)** | $0{,}7964 \pm 0{,}0073$ | $0{,}6489 \pm 0{,}0188$ | $0{,}5083 \pm 0{,}0308$ | $0{,}5696 \pm 0{,}0210$ |
| **RF (Reg., prof.=12)** | **$0{,}8093 \pm 0{,}0085$** | $0{,}6743 \pm 0{,}0211$ | $0{,}5452 \pm 0{,}0246$ | $0{,}6027 \pm 0{,}0190$ |
| **XGBoost (Baseline)** | $0{,}8079 \pm 0{,}0096$ | $0{,}6649 \pm 0{,}0248$ | $0{,}5581 \pm 0{,}0218$ | **$0{,}6065 \pm 0{,}0186$** |
| **XGBoost (Reg., $\lambda=2$)** | $0{,}8055 \pm 0{,}0096$ | $0{,}6613 \pm 0{,}0250$ | $0{,}5490 \pm 0{,}0253$ | $0{,}5995 \pm 0{,}0199$ |

**TABELA II — Consolidação das matrizes de confusão sobre os 7.043 clientes da base**

| Modelo / Configuração | VN (Não Churn correto) | FP (Alarme falso) | FN (Churn não detectado) | VP (Churn detectado) |
| :--- | :---: | :---: | :---: | :---: |
| **KNN ($k=5$)** | 4.431 | 743 | 814 | 1.055 |
| **KNN ($k=11$)** | 4.488 | 686 | 788 | 1.081 |
| **SVM (Linear)** | 4.633 | 541 | 823 | 1.046 |
| **SVM (RBF)** | 4.716 | 458 | 887 | 982 |
| **RF (Baseline)** | 4.659 | 515 | 919 | 950 |
| **RF (Regularizada)** | 4.681 | 493 | 850 | 1.019 |
| **XGBoost (Baseline)** | 4.647 | 527 | 826 | 1.043 |
| **XGBoost (Regularizado)** | 4.647 | 527 | 843 | 1.026 |

### Discussão dos Resultados

Todos os classificadores superaram com facilidade a linha de base de 73,46% de acurácia. O **XGBoost Baseline** alcançou o melhor equilíbrio geral do trabalho, obtendo o maior F1-Score (0,6065), além de uma boa revocação (55,81%), identificando corretamente 1.043 dos 1.869 cancelamentos. Na configuração B do XGBoost, o aumento da regularização L2 ($\lambda=2$) acabou tornando o modelo um pouco rígido demais, reduzindo sutilmente o F1-Score para 0,5995.

O **SVM** mostrou comportamentos bem distintos entre os kernels. O SVM Linear foi o segundo modelo mais equilibrado (F1 de 0,6052 e revocação de 55,97%), acertando 1.046 cancelamentos. Já o SVM RBF foi o campeão em precisão (68,24%) e gerou o menor número de falsos alarmes (458 FPs), mas deixou de detectar mais cancelamentos (887 falsos negativos).

No caso da **Random Forest**, a restrição de profundidade fez uma diferença expressiva: o modelo sem limite de profundidade sofreu com sobreajuste nas árvores, obtendo F1 de apenas 0,5696. Quando limitamos a profundidade máxima em 12, o F1 subiu para 0,6027 e a acurácia atingiu 80,93%, a mais alta entre todos os modelos testados.

Por fim, o **KNN** teve os resultados mais modestos na configuração inicial de $k=5$ ($F1=0{,}5754$), como esperado para modelos baseados em distância em bases com muitos atributos codificados. Porém, subir para $k=11$ ajudou bastante a suavizar a vizinhança, elevando o F1 para 0,5945 e reduzindo os falsos alarmes de 743 para 686.

### Interpretação para o Problema de Negócio

No contexto de cancelamento de clientes em telecomunicações, o custo de perder um cliente (falso negativo) costuma ser bem mais pesado do que o custo de enviar uma oferta ou benefício para um cliente fiel que não pretendia sair (falso positivo). Sob esse ponto de vista, modelos que priorizam a detecção de cancelamentos com bom equilíbrio de F1-Score — especialmente o **XGBoost Baseline** e o **SVM Linear** — são os mais indicados para orientar ações preventivas de retenção.
