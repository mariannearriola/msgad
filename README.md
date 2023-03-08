# Unsupervised multi-scale graph anomaly detection

Currently supported datasets: cora, weibo, tfinance

Currently supported models: multi-scale, dominant

## Model training & anomaly detection:
```
cd msgad

python run.py --dataset cora_triple_sc_all --sample_train False --sample_test False --model dominant
```

Anomaly detection results are printed to the console and saved in the ./msgad/output folder

## Code organization
- `data/` – Scipy matrix files containing currently supported datasets (link to T-Finance: TODO)
- `models/` – Reconstruction backbones
- `run.py` – Script to run model training & anomaly detection
- `model.py` – Main graph reconstruction model (TODO: make superclass)
- `loss_utils.py/` – Contains loss calculations
- `utils.py` – Contains anomaly detection pipeline & dataloading