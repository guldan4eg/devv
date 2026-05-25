# WikiArt Style Classifier

Classification of 10 artistic styles from the WikiArt dataset using ResNet50 and transfer learning (PyTorch).

## Styles

Impressionism, Realism, Romanticism, Expressionism, Baroque, Ukiyo-e, Art Nouveau, Surrealism, Symbolism, Cubism

## Project Structure

```
├── config.py        # Hyperparameters, paths, class names
├── data.py          # WikiArtDataset, split_dataset(), data loaders
├── model.py         # get_model(), Grad-CAM
├── train.py         # Training loop (transfer learning + fine-tuning)
├── evaluate.py      # Test evaluation, confusion matrix, per-class metrics
├── inference.py     # Single-image prediction with Grad-CAM
├── export_onnx.py   # ONNX export
├── utils.py         # EarlyStopping, metrics, visualization
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the [WikiArt dataset from Kaggle](https://www.kaggle.com/datasets/steubk/wikiart) and organize it so that each class has its own subdirectory:

```
data/
├── Impressionism/
├── Realism/
├── Romanticism/
...
```

Set the path in `config.py` or via the `--data_dir` CLI argument.

## Training

```bash
# Stage 1: transfer learning (frozen backbone)
# Stage 2: fine-tuning (unfrozen layer3, layer4, fc)
python train.py --data_dir /path/to/wikiart
```

## Evaluation

```bash
python evaluate.py --data_dir /path/to/wikiart --checkpoint best_model.pth
```

## Inference

```bash
python inference.py --image /path/to/image.jpg --checkpoint best_model.pth --gradcam
```

## ONNX Export

```bash
python export_onnx.py --checkpoint best_model.pth --output model.onnx
```
