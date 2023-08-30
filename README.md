# Unsupervised multi-scale graph anomaly detection

Currently supported datasets: weibo, elliptic, yelpchi (untested), tfinance (untested)

Currently supported models: multi-scale-dominant, multi-scale-amnet

## Cross-model training & anomaly detection:
```
SCALES=3
DATASET=weibo
DEVICE=0
SAVE=True

./full_run.sh $SCALES $DATASET $DEVICE $SAVE
```

## MsGAD training & anomaly detection:
```
cd msgad
SCALES=3
DATASET=weibo
DEVICE=0
SAVE=True

./run_detection.sh $SCALES $DATASET $DEVICE $SAVE
```

Anomaly detection results plotted in ./msgad/output

## Code organization

In the folder `msgad`:
- `data/` – Scipy matrix files containing currently supported datasets
- `output/` – Figures/pickle files used for logging anomaly detection scores/ranking across methods
- `dgl_graphs/` – DGL graph yamls needed for DGL distributed dataloading
- `run_detection.sh` – Script to run experiments
- `run.py` – Script to run model training & anomaly detection
- `label_analysis.py` – Creates hierarchical labels used for multi-scale anomaly classification/learning
- `model.py` – Main graph reconstruction model
- `loss_utils.py/` – Contains loss calculations
- `utils.py` – Contains anomaly detection pipeline, dataloading, and tensorboard logging class.
- `plot_model_wise_hits.py` – Plots cross-model results to store in `msgad/output`

