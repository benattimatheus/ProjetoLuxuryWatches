# Dataset Luxury Watches
Matheus Mitsuo Yamafuku Benatti<br>
RA: 22066197-2


```markdown
# ProjetoLuxuryWatches 🕰️

Pipeline completo para análise, pré-processamento, edição e modelagem preditiva de datasets com foco em valores de relógios de luxo.  
Utiliza AutoML (TPOT), visualizações EDA (AutoViz, Sweetviz, D-Tale, YData), PyCaret.

---

## ✅ Pré-requisitos

- Python **3.11.7 (recomendado)
- Instalar as dependências com:

pip install -r requirements.txt
```

---

## Como usar

O sistema pode ser executado em **três modos principais**:

### 1. Modo de Edição (`edit`)

Permite editar visualmente um arquivo `.csv` com o D-Tale:

```bash
python main.py edit Watches.csv
```

### 2. Modo de Treinamento (`train`)

Treina um modelo baseado no tipo da tarefa (`regression`, `classification`, `clustering`):

```bash
python main.py train Watches.csv price regression
```


### 3. Pipeline Completa e EDA

Executa o pipeline completo: download, geração de relatórios EDA, treinamento com TPOT e geração de análise exploratória de dados.

```bash
python main.py
```

Este modo utiliza um dataset padrão e assume que a variável `price` é o target.

---

## 📌 Observações

- O target `price` deve ser numérico para regressão.

---

## 🐍 Versão recomendada do Python

- **Python 3.11.7

---

## 🧠 Tecnologias utilizadas

- TPOT (AutoML)
- PyCaret
- SHAP
- D-Tale
- Sweetviz
- AutoViz
- YData Profiling

