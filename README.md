<h1 align="center">AAVGen: Precision Engineering of Adeno-associated Virus for Renal Selective Targeting</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  </a>
  <a href="https://huggingface.co/datasets/Moreza009/AAV_datasets">
    <img src="https://img.shields.io/badge/🤗-HuggingFace-yellow.svg" alt="HuggingFace">
  </a>  
  <a href="https://arxiv.org/abs/2602.18915">
    <img src="https://img.shields.io/badge/arXiv-2508.18579-b31b1b.svg" alt="arXive">
  </a>
</p>

<p align="center">
  <img src="/assets/Logo.png" alt="Logo" width="500">
</p>

## TL;DR

AAVGen is an AI-driven framework that designs novel AAV capsids optimized simultaneously for kidney targeting, production efficiency, and thermal stability, enabling next-generation renal gene therapy vectors.

---

## Abstract

Adeno-associated viruses (AAVs) are promising vectors for gene therapy, but their native serotypes face limitations in tissue tropism, immune evasion, and production efficiency. Engineering capsids to overcome these hurdles is challenging due to the vast sequence space and the difficulty of simultaneously optimizing multiple functional properties. The complexity also adds when it comes to the kidney, which presents unique anatomical barriers and cellular targets that require precise and efficient vector engineering. Here, we present AAVGen, a generative artificial intelligence framework for de novo design of AAV capsids with enhanced multi-trait profiles. AAVGen integrates a protein language model (PLM) with supervised fine-tuning (SFT) and a reinforcement learning technique termed Group Sequence Policy Optimization (GSPO). The model is guided by a composite reward signal derived from three ESM-2-based regression predictors, each trained to predict a key property: production fitness, kidney tropism, and thermostability. Our results demonstrate that AAVGen produces a diverse library of novel VP1 protein sequences. In silico validations revealed that the majority of the generated variants have superior performance across all three employed indices, indicating successful multi-objective optimization. Furthermore, structural analysis via AlphaFold3 confirms that the generated sequences preserve the canonical capsid folding despite sequence diversification. AAVGen establishes a foundation for data-driven viral vector engineering, accelerating the development of next-generation AAV vectors with tailored functional characteristics.


![General flow image](/assets/Fig_1.png)

*Figure 1. Schematic overview of the AAVGen framework. Three ESM-2 regression models predict production fitness, kidney tropism, and thermostability, serving as reward functions within a GSPO-based reinforcement learning pipeline to fine-tune a ProtGPT2 generative model.*

---

## Key Results

- **Regression models**: Production fitness predictor achieved Spearman ρ = 0.91; kidney tropism ρ = 0.35; thermostability ρ = 0.26.
- **Generative diversity**: AAVGen generated 500,000 sequences with ~96% uniqueness; median sequence identity to AAV2 WT of 99.18%, with ~13% edit distance, confirming biologically plausible novelty.
- **Functional quality**: 99.7% of generated sequences classified as "Best" for production fitness; 98.27% as "Good" or better for kidney tropism; 88.57% as "Good" for thermostability.
- **Multi-property co-optimization**: Strong positive Spearman correlations across all three properties, confirming no property trade-off.
- **Structural fidelity**: AlphaFold3 analysis showed median RMSD of ~0.42–0.47 Å vs. AAV2 WT, far superior to randomly generated baselines (median RMSD 0.48 Å, median fitness −4.65).

---

## Repository Structure

```
AAVGen/
├── assets/                    # Figures and visual assets
├── datasets/                  # Input/output datasets (.csv)
├── src/
│   ├── data_processing/       # Data preprocessing scripts
│   │   ├── get_deep_diversification_production_fitness.py
│   │   ├── get_fit4function_production_datasets.py
│   │   └── get_landscape_production_fitness.py
│   ├── train/                 # Model training scripts
│   │   ├── aav2_regs.py       # Regression model training
│   │   ├── sft.py             # Supervised fine-tuning
│   │   └── gspo.py            # GSPO reinforcement learning
│   └── inference/             # Sequence generation and scoring
│       ├── AAVGen.py
│       ├── AAVGen_CLI.py
│       └── regressions_inferences.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/AAVGen.git
cd AAVGen
pip install -r requirements.txt
```
---

## Data

Raw and processed datasets are available on Hugging Face:

👉 [Moreza009/AAV_datasets](https://huggingface.co/datasets/Moreza009/AAV_datasets)

The training corpus integrates data from three independent studies:
- **Ogden et al.** — AAV2 deep mutational scanning: 31,579 VP1 sequences (production fitness), 24,984 (kidney tropism), 30,889 (thermostability).
- **Bryant et al.** — AAV2 VP1 residues 561–588: 296,896 multi-mutation sequences with fitness scores.
- **Eid et al.** — AAV9: 100,000 sequences with production fitness and liver biodistribution data.

### Data Preprocessing

```bash
cd src/data_processing
python get_deep_diversification_production_fitness.py
python get_fit4function_production_datasets.py
python get_landscape_production_fitness.py
```

> A preprocessed version is also available directly on the [Hugging Face dataset hub](https://huggingface.co/datasets/Moreza009/AAV_datasets).

---

## Model Training

### 1. Regression Models (Reward Functions)

Three ESM-2-based regression models are trained sequentially via transfer learning to predict production fitness, kidney tropism, and thermostability. The fitness model (trained for ~11.4 hours on an NVIDIA V100 32 GB) serves as the initialization checkpoint for the other two models.

```bash
cd src/train
python aav2_regs.py
```

### 2. Supervised Fine-Tuning (SFT)

ProtGPT2 (~738M parameters) is fine-tuned on 192,199 non-redundant VP1 sequences from AAV2 and AAV9 to learn residue–residue relationships across serotypes (~9 hours on NVIDIA V100).

```bash
cd src/train

python sft.py
```

### 3. Group Sequence Policy Optimization (GSPO)

The SFT model is further refined via GSPO using a composite reward signal from the three regression models, plus auxiliary rewards for sequence length diversity and intra-batch uniqueness (~9.6 hours on NVIDIA V100).

```bash
cd src/train
python gspo.py
```

---

## Inference

### Sequence Generation

**Via a graphical UI**: [Launch the web interface](link) to generate AAV VP1 sequences interactively.

**Via Python API**:

```python
from AAVGen import generate

input_sequences = ["M", "MAAG"]

generated_sequences = generate(
    input_sequences,
    temperature=0.8,
    top_p=0.95,
    max_length=300,
)

print(generated_sequences)
```

**Via CLI** (recommended for large-scale generation):

Quick test run:
```bash
cd src/inference

python AAVGen_CLI.py -n 1000 -b 32 --name test-run
```

Full example with all options:
```bash
python AAVGen_CLI.py \
  -n 100000 \
  --batch-size 64 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 40 \
  --max-length 600 \
  --output-dir ./results \
  --name my-aav2-sequences
```

### Functional Property Scoring

Score your generated sequences for production fitness, kidney tropism, and thermostability using the trained regression models. Place your sequences in a `.csv` file with a column named `generate_seqs` inside the `datasets/` folder, then run:

```bash
cd src/inference

python regressions_inferences.py \
    --df_name my_dataset\
    --batch_size 256
```

> Do not include the `.csv` extension in `--df_name`. Output will be saved as `my_dataset_out.csv`.

> **Note**: The preprocessing step (see [Data Preprocessing](#data-preprocessing)) must be completed before running regression inference.

---

## Methods Overview

### Architecture

| Component | Base Model | Parameters | Purpose |
|-----------|-----------|------------|---------|
| Regression models (×3) | ESM-2 (UR50D) | 8M | Reward functions for fitness, tropism, thermostability |
| Generative model | ProtGPT2 | 738M | VP1 sequence generation |

### Training Pipeline

1. **Regression models** are fine-tuned from ESM-2 using sequential transfer learning (fitness → tropism, fitness → thermostability), supervised with MSE loss and AdamW.
2. **SFT** fine-tunes ProtGPT2 on a curated corpus of 192,199 high-fitness AAV2 and AAV9 VP1 sequences.
3. **GSPO** applies sequence-level policy gradient optimization. For each training step, G=32 candidate sequences are generated and evaluated; advantage estimates are computed via group-wise normalization, and the policy is updated using a clipped surrogate objective (ε=0.2).

### Reward Functions

| Reward | Description |
|--------|-------------|
| Production fitness | ESM-2 regressor score vs. AAV2 WT threshold |
| Kidney tropism | ESM-2 regressor score vs. AAV2 WT threshold |
| Thermostability | ESM-2 regressor score vs. AAV2 WT threshold |
| Length controller | Gaussian penalty discouraging WT-length sequences |
| Intra-batch uniqueness | Binary reward penalizing duplicate sequences within a batch |

### Structural Validation

500 sequences sampled from "Good"/"Best" categories were folded using **AlphaFold3** (5 structures each) and compared to the AAV2 WT PDB structure via RMSD in PyMOL (v3.1.1). A baseline of 250 randomly mutated sequences was generated for comparison.

---

## Hardware

All models were trained on a dedicated server equipped with:
- **GPU**: NVIDIA V100 (32 GB VRAM)
- **CPU**: AMD EPYC 7502
- **RAM**: 32 GB

| Model | Training Time |
|-------|--------------|
| Production fitness regression | 11 h 25 min |
| Kidney tropism regression | 3 h 24 min |
| Thermostability regression | 3 h 29 min |
| SFT (ProtGPT2) | 9 h 05 min |
| GSPO fine-tuning | 9 h 38 min |

---

## Citation

If you use AAVGen in your research, please cite:

```bibtex
@misc{ghaffarzadehesfahani2026aavgenprecisionengineeringadenoassociated,
      title={AAVGen: Precision Engineering of Adeno-associated Viral Capsids for Renal Selective Targeting}, 
      author={Mohammadreza Ghaffarzadeh-Esfahani and Yousof Gheisari},
      year={2026},
      eprint={2602.18915},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2602.18915}, 
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please open an issue or contact [mreghafarazadeh@gmail.com](mailto:mreghafarazadeh@gmail.com).
