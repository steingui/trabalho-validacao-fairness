# Trabalho 2 - Validação de Dados, Testes Estatísticos e Fairness

Este repositório contém a implementação do Trabalho 2 da disciplina de Validação de Dados, Testes Estatísticos e Fairness (Aulas 3 e 4).

## 1. Visão Geral

O objetivo do projeto é construir um pipeline preditivo sobre o dataset **Adult Census Income**, aplicando validações rigorosas na entrada de dados, testes de estabilidade estatística e verificação de equidade (fairness) entre grupos demográficos.

- **Dataset:** Adult Census Income (UCI Machine Learning Repository)
- **Atributo Sensível:** `sex` (Male / Female)
- **Variável Alvo (Target):** `income` (`>50K` / `<=50K`)

---

## 2. Estrutura do Repositório

```
.
├── trabalho_validacao_fairness.ipynb  # Notebook principal com explicações e código executável
├── test_trabalho.py                   # Suíte de testes automatizados com pytest
├── requirements.txt                   # Dependências do projeto
└── README.md                          # Documentação detalhada
```

---

## 3. Metodologia e Implementação

### Entregável 1: Validação de Dados (Padrão Aula 3)
Funções utilitárias escritas em Python puro para validar integridade antes da modelagem:
- `validate_schema`: garante a presença das colunas esperadas e valida os tipos de dados (`int64`, `object`).
- `check_no_missing_values`: verifica ausência de nulos em campos obrigatórios.
- `check_range`: valida se colunas numéricas (`age`, `hours_per_week`) respeitam limites físicos plausíveis.
- `check_unique_ids`: identifica registros duplicados na base.

### Entregável 2: Testes Estatísticos (Padrão Aula 4)
- **Intervalo de Confiança via Bootstrap (`bootstrap_ci`):** realiza 1.000 reamostragens com reposição (`np.random.default_rng`) para estimar o intervalo de confiança de 95% da acurácia.
- **Comparação entre Modelos (`compare_models`):** avalia se a diferença de acurácia entre o **Modelo A** (Regressão Logística com One-Hot Encoding) e o **Modelo B** (Regressão Logística somente com variáveis numéricas) é estatisticamente significante.

### Entregável 3: Testes de Fairness (Padrão Aula 4)
- **Paridade Demográfica (`check_demographic_parity`):** compara a taxa de aprovação predita ($P(\hat{Y}=1)$) entre os grupos de gênero (`Male` vs `Female`), definindo um limiar máximo aceitável de tolerância de 10% (0.10).
- **Disparate Impact Ratio:** calcula a razão $P(\hat{Y}=1 | Female) / P(\hat{Y}=1 | Male)$ e verifica o enquadramento na regra dos 80% (EEOC).

---

## 4. Resultados e Interpretação (Entregável 4)

### Validação de Dados
- O dataset original possui 32.561 registros.
- Foram detectados **2.399 registros com valores ausentes** (representados originalmente por `?`) nas colunas `workclass`, `occupation` e `native_country`.
- Foram encontradas **24 linhas duplicadas**, removidas durante o saneamento.
- Os limites numéricos (`age` entre 17 e 90 anos; `hours_per_week` entre 1 e 99) foram atendidos sem anomalias extremas.

### Testes Estatísticos
- **Acurácia Pontual (Modelo A):** 84,76%
- **IC 95% (Bootstrap 1.000 iterações):** [84,02%, 85,50%]
- **Comparação A vs B:** A diferença de acurácia média foi de +6,2% a favor do modelo completo. O intervalo de confiança de 95% para a diferença [5,2%, 7,1%] exclui o zero, confirmando ganho de performance estatisticamente significante.

### Teste de Fairness
- **Taxa de Predição Positiva por Gênero:**
  - Homens (`Male`): 26,1%
  - Mulheres (`Female`): 8,1%
- **Gap de Paridade Demográfica:** 18,0% (Reprovado no teste de limite de 10%).
- **Disparate Impact Ratio:** 0,31 (Abaixo do mínimo recomendado de 0,80).
- **Conclusão:** O modelo treinado sem pós-processamento reproduz e amplifica a disparidade de renda histórica contida na base original. Para utilização em ambiente de produção, seria necessário adotar estratégias de mitigação de viés (como *re-weighting* na perda ou ajuste de limiar de decisão por subgrupo).

---

## 5. Como Executar

### Via Terminal (Pytest)

```bash
# Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Executar a suíte de testes
pytest test_trabalho.py -v
```

### Via Jupyter Notebook

Abra o arquivo `trabalho_validacao_fairness.ipynb` no VS Code ou Jupyter Lab e execute todas as células para visualizar o fluxo sequencial de validação, treinamento e saídas de teste.
