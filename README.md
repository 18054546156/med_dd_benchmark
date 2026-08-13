# Medical Dataset Distillation Benchmark

A comprehensive benchmark for evaluating dataset distillation methods on medical imaging datasets.

---

## 📊 Datasets

### PathMNIST
- **Classes**: 9 (colorectal tissue types)
- **Original size**: 28×28×3
- **Preprocessed size**: 32×32×3
- **Split**: 89,996 train / 10,004 val / 7,180 test
- **Source**: [MedMNIST](https://medmnist.com/)

### COVID-19
- **Classes**: 4 (COVID, Lung_Opacity, Normal, Viral_Pneumonia)
- **Original size**: varies
- **Preprocessed size**: 112×112×3
- **Split**: train / val / test
- **Source**: Kaggle COVID-19 Image Dataset

### Kvasir
- **Classes**: 8 (gastrointestinal diseases)
- **Original size**: varies
- **Preprocessed size**: 128×128×3
- **Split**: train / val / test
- **Source**: [Kvasir-Capsule](https://datasets.simula.no/kvasir-capsule/)

---

## 🚀 Quick Start

### 1. Data Download & Preprocessing

#### PathMNIST
```bash
# Auto download via medmnist
python -c "
import medmnist
medmnist.PathMNIST(split='train', download=True, root='data/raw')
"

# Preprocess (if using archived script)
python scripts/archived_scripts/prepare_medical_data.py \
    --data-root data prepare --dataset PathMNIST
```

#### COVID-19
```bash
# Download from Kaggle (requires API key)
kaggle datasets download -d pranavraikokte/covid19-image-dataset
unzip covid19-image-dataset.zip -d data/raw/COVID/

# Preprocess
python scripts/archived_scripts/prepare_medical_data.py \
    --data-root data prepare --dataset COVID
```

---

### 2. Run NCFM

#### Three-Stage Pipeline (Recommended)
```bash
# Stage 1: Pretrain 20 teachers (~2 hours, 4 GPUs)
sbatch scripts/ncfm_pretrain_fixed.sbatch

# Stage 2: Condense synthetic data (~4 hours, 4 GPUs)
sbatch scripts/ncfm_condense_fixed.sbatch

# Stage 3: Evaluation (~40 mins, 4 GPUs)
sbatch scripts/ncfm_eval_dsa_only.sbatch
```

**Results**: PathMNIST IPC=10: **80.4% ± 0.6%** (DSA only, fair comparison)

---

### 3. Run HoP

#### Complete Pipeline (Recommended)
```bash
# Complete: Buffer → Distill → Evaluation (~6-7 hours, 4 GPUs)
sbatch --export=DATASET=PathMNIST scripts/hop_tm_pipeline_4gpu.sbatch
```

---

## 📐 Model Architectures

| Dataset | Image Size | Backbone | Depth | Parameters |
|---------|-----------|----------|-------|------------|
| PathMNIST | 32×32 | ConvNet | D3 | - |
| COVID | 112×112 | ConvNet | D5 | - |
| Kvasir | 128×128 | ConvNet | D5 | - |

**Note**: Both NCFM and HoP use the same image sizes and backbones for each dataset.

---

## 🔬 Methods

### NCFM (Neutral Characteristic Function Matching)
- **Core**: Characteristic Function (CF) Loss with 4096 random frequencies
- **Teachers**: 20 pre-trained models
- **Augmentation**: DSA + CutMix (pretrain), DSA only (condense/eval)
- **Optimizer**: AdamW (lr=0.001) for evaluation

### HoP (High-Order Trajectory Matching)
- **Core**: Trajectory matching across expert checkpoints
- **Experts**: 100 expert trajectories (4 workers × 25 each)
- **Augmentation**: DSA only
- **Optimizer**: SGD (lr=0.01) for evaluation

---

## ✅ Code Verification

### Key Features
- ✅ **Correct Normalization**: Avoids double normalization bug
- ✅ **Unified Image Sizes**: NCFM and HoP use same sizes
- ✅ **Fair Evaluation**: DSA only, no CutMix
- ✅ **Core Algorithms**: Unchanged from original implementations

### Verification Status
```
✅ NCFM core algorithm: matches GitHub original
✅ HoP core algorithm: matches GitHub original
✅ Normalization handling: train_skip_normalize=True
✅ Backbone configuration: PathMNIST D3, COVID/Kvasir D5
✅ Image sizes: PathMNIST 32×32, COVID 112×112, Kvasir 128×128
```

---

## 📊 Results

### PathMNIST (IPC=10)

| Method | Teachers/Experts | Augmentation | Accuracy |
|--------|-----------------|--------------|----------|
| NCFM | 20 | DSA only | 80.4% ± 0.6% |
| HoP | 100 | DSA only | (running) |

**Evaluation Protocol**: 
- Same for both methods
- DSA augmentation only (no CutMix)
- 10 independent runs
- Test on official test split

---

## 📁 Directory Structure

```
dd_benchmark/
├── adapted/                  # Algorithm implementations
│   ├── ncfm/                # NCFM code
│   └── hop_tm/              # HoP code
├── configs/                 # Configuration files
│   ├── ncfm/
│   │   ├── pathmnist/
│   │   │   ├── ipc10_full_fixed.yaml
│   │   │   └── ipc10_eval_dsa_only.yaml
│   │   ├── covid/
│   │   └── kvasir/
│   └── hop_tm/
│       ├── pathmnist/
│       └── covid/
├── data/                    # Data directory
│   ├── raw/                # Raw downloads
│   └── prepared/           # Preprocessed data
│       ├── PathMNIST/
│       ├── COVID/
│       └── Kvasir/
├── scripts/                 # Run scripts (7 core scripts)
│   ├── ncfm_pretrain_fixed.sbatch
│   ├── ncfm_condense_fixed.sbatch
│   ├── ncfm_eval_dsa_only.sbatch
│   ├── hop_tm_pipeline_4gpu.sbatch
│   ├── hop_tm_pipeline.sbatch
│   ├── hop_tm_worker.sbatch
│   ├── hop_tm_finalize.sbatch
│   └── archived_scripts/   # Old validation/test scripts
├── utils/                   # Shared utilities
└── results/                 # Experiment results
```

---

## 🔧 Configuration Files

### NCFM Configuration Source

Most parameters come from the original NCFM CIFAR-10 configuration:

| Parameter | Value | Source |
|-----------|-------|--------|
| `num_freqs` | 4096 | Raw NCFM paper |
| `num_premodel` | 20 | Raw NCFM CIFAR-10 |
| `ipc` | 10 | Raw NCFM CIFAR-10 |
| `niter` | 20000 | Raw NCFM CIFAR-10 |
| `eval_optimizer` | adamw | Raw NCFM |
| `adamw_lr` | 0.001 | Raw NCFM |

### HoP Configuration Source

Parameters from the original HoP (MTT) implementation:

| Parameter | Value | Source |
|-----------|-------|--------|
| `num_experts` | 100 | Raw HoP (MTT) |
| `train_epochs` | 100 | Raw HoP (MTT) |
| `save_interval` | 25 | Raw HoP (MTT) |
| `ipc` | 10 | Raw HoP (MTT) |
| `Iteration` | 10000 | Raw HoP (MTT) |

---

## 📝 Key Implementation Details

### Normalization Handling (Critical)

**NCFM**:
```python
# Load data with skip_normalize=True
load_medical_splits(dataset, data_dir, train_skip_normalize=True)

# Normalize only once in diffaug
images = dsa_augment(images)
images = (images - mean) / std  # Only normalization
```

**HoP**:
```python
# Load data with normal normalization
load_medical_splits(dataset, data_dir, train_skip_normalize=False)

# Already normalized in loader, apply DSA
images = dsa_augment(images)
```

**Result**: Train and val distributions are consistent ✅

---

### Fair Comparison Protocol

To ensure fair comparison between NCFM and HoP:

1. **Same Image Sizes**: 
   - PathMNIST: 32×32
   - COVID: 112×112
   - Kvasir: 128×128

2. **Same Backbones**:
   - PathMNIST: ConvNet-D3
   - COVID: ConvNet-D5
   - Kvasir: ConvNet-D5

3. **Same Augmentation (Evaluation)**:
   - DSA only
   - No CutMix
   - No other augmentations

4. **Same Test Set**:
   - Official test split
   - Same preprocessing

---

## 🎯 Core Scripts

### Active Scripts (7)
```
scripts/ncfm_pretrain_fixed.sbatch   - NCFM Stage 1
scripts/ncfm_condense_fixed.sbatch   - NCFM Stage 2
scripts/ncfm_eval_dsa_only.sbatch    - NCFM Stage 3
scripts/hop_tm_pipeline_4gpu.sbatch  - HoP complete (4 GPU)
scripts/hop_tm_pipeline.sbatch       - HoP complete (1 GPU)
scripts/hop_tm_worker.sbatch         - HoP worker (manual)
scripts/hop_tm_finalize.sbatch       - HoP finalize (manual)
```

### Archived Scripts (23)
```
scripts/archived_scripts/            - Validation/test scripts
  ├── validate_*.py                 - Configuration validation
  ├── test_*.py                     - Model testing
  ├── prepare_medical_data.py       - Data preprocessing
  └── ... (20 more)
```

---

## 🐛 Known Issues

### Fixed
- ✅ Train/val normalization inconsistency (double normalization bug)
- ✅ COVID backbone mismatch (now unified to D5)
- ✅ MEANS/STDS not synced to NCFM global dictionary

### Current Status
- ✅ All core issues resolved
- ✅ Code verified on HPC
- ✅ NCFM PathMNIST results validated (80.4% ± 0.6%)

---

## 📚 References

### NCFM
- Paper: "Dataset Distillation via Neutral Characteristic Function Matching"
- GitHub: (original NCFM repository)

### HoP (MTT)
- Paper: "Dataset Distillation by Matching Training Trajectories"
- GitHub: (original MTT/HoP repository)

### Medical Datasets
- MedMNIST: https://medmnist.com/
- COVID-19: Kaggle dataset
- Kvasir: https://datasets.simula.no/kvasir-capsule/

---

## 🤝 Contributing

This is a research benchmark. For questions or issues, please contact the project maintainers.

---

## 📄 License

(To be determined based on original implementations)

---

## 🙏 Acknowledgments

- Original NCFM authors
- Original HoP (MTT) authors
- MedMNIST team
- Medical dataset providers

---

**Last Updated**: 2026-08-14

**Status**: 
- ✅ NCFM PathMNIST complete (80.4% ± 0.6%)
- 🔄 HoP PathMNIST running (60%)
- 🔄 HoP COVID running (12%)
