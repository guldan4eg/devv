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
├── export_onnx.py       # ONNX export
├── download_dataset.py  # Download WikiArt from HuggingFace
├── utils.py             # EarlyStopping, metrics, visualization
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

A sample dataset (50 images per class, 500 total) is included in `data/`.

To download the full dataset from HuggingFace:

```bash
python download_dataset.py                          # all available images
python download_dataset.py --max_per_class 500      # 500 per class
```

You can also use the [WikiArt dataset from Kaggle](https://www.kaggle.com/datasets/steubk/wikiart) — organise it so each class has its own subdirectory matching the names in `config.py`.

Set the path via the `--data_dir` CLI argument.

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
