## 🧠 Assignment 3 – CNNs for Image Classification (CIFAR-10)

**Course**: CMPUT 328 — Visual Recognition  
**Student**: Linh (`<lvle>`)  
**Parts**: A. In-lab (3 hrs), B. Take-home (1 week)

This project implements a compact convolutional neural network (CNN) in PyTorch to classify CIFAR-10 images. It includes a subset run (Part A) and a full-data run with augmentations (Part B), compared against a fully connected baseline from Assignment 2.

---

## 🛠️ Setup Instructions

### Environment
- Python 3.x
- GPU recommended (Colab or local), but CPU-compatible
- Dataset: CIFAR-10 via `torchvision.datasets.CIFAR10`

### Installation
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Code

### Notebook
1. Open `Assignment3_Template.ipynb`
2. Run top-to-bottom in a fresh runtime
3. Ensure internet access to download CIFAR-10

### Optional GPU
```python
use_gpu = torch.cuda.is_available()
```

---

## 🔁 Reproducibility Settings

Set these seeds before data loading and model initialization:
```python
import torch, random, numpy as np
torch.manual_seed(328)
random.seed(328)
np.random.seed(328)

# For DataLoader
from torch.utils.data import DataLoader
g = torch.Generator()
g.manual_seed(328)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=g)

# For deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 📂 Folder Structure

```
A3_CNN_<CCID>.zip/
├── Assignment3_Template.ipynb
├── README.md
├── requirements.txt
├── report.pdf
├── model.pt (optional)
```

---

## 📊 Results Summary

| Part | Accuracy | Epochs | Augmentations Used |
|------|----------|--------|---------------------|
| A (Subset) | 52.3% | 10 | None |
| B (Full)   | 78.6% | 45 | RandomCrop, HorizontalFlip, ColorJitter,  |

- CNN outperformed A2 MLP by +12.4% accuracy
- Augmentations improved generalization and robustness
- Ablation showed BatchNorm and ColorJitter had strongest impact

---

## 📌 Notes for Reproducibility

We fix `seed=42`, use a `45k/5k/10k` split, and train **SmallCNN** for **15 epochs** with **AdamW** (`lr=1e-3`, `wd=5e-4`), a cosine LR schedule, and batch size **256**. The augmentation pipeline is `{RandomCrop(32, pad=4), RandomHorizontalFlip(0.5), ColorJitter (mild), RandomErasing (small)}`. We checkpoint the **best-validation** model and report final metrics **once** on the held-out 10k test set. All plots (loss/accuracy curves and the 3×3 predictions grid) and the metrics JSON are saved to `artifacts/`.


