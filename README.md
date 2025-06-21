# Offensive Language Reformulation and Detection using LLMs

This project investigates whether large language models (LLMs) introduce semantic bias when reformulating offensive language and whether such reformulated content remains detectable as offensive by other models.

## 🧠 Project Overview

- **Goal:** Evaluate the consistency and bias of LLMs in processing offensive language.
- **Pipeline:**
  1. **LLM1** receives an original sentence, classifies it as offensive or not, reformulates it (if offensive), and reclassifies the reformulation.
  2. **LLM2** receives both the original and the LLM1 reformulation and classifies them independently. It also reformulates the original sentence and classifies its own output.
  3. **Comparison step** analyzes agreement/divergence between the two models in both reformulation and classification.

## 📁 Project Structure

```
project-root/
│
├── data/                  # Dataset inputs and outputs
│   ├── original_dataset.csv
│   ├── llm1_output.csv
│   └── llm2_output.csv
│
├── scripts/               # Python scripts
│   ├── run_llm1.py
│   ├── run_llm2.py
│   └── compare_results.py
│
├── prompts/               # Prompt templates
│   ├── reformulation.txt
│   └── classification.txt
│
├── results/               # Generated analysis and JSONs
│   └── classification_log.json
│
├── report/                # LaTeX report files
│   ├── main.tex
│   └── references.bib
│
└── README.md
```

## 🧪 Requirements

- Python 3.10+
- `transformers`
- `datasets`
- `pandas`
- Access to at least one LLM (locally or via API)

## 🚀 Running the Project

1. Prepare your dataset (`data/original_dataset.csv`)
2. Run the first pipeline:
   ```bash
   python scripts/run_llm1.py
   ```
3. Run the second model:
   ```bash
   python scripts/run_llm2.py
   ```
4. Analyze results:
   ```bash
   python scripts/compare_results.py
   ```

## 📊 Output Format

Each output CSV contains:

| id | original_sentence | original_reason | reform_sentence | reform_reason |
|----|-------------------|------------------|------------------|----------------|
| 1  | "example"         | "why offensive"  | "rephrased"      | "why offensive"|

Additional classification metadata (e.g., `classification_changed`, `reform_type`) are stored in a separate JSON log.

## 📚 Citation

If you use this project in your research, please cite:

> Freitas, G. (2025). _Investigating Semantic Shifts in LLM Reformulations of Offensive Language_.
