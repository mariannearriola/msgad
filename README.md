Currently supported datasets: Cora, Weibo, T-Finance
Currently supported models: Multi-scale, DOMINANT, AnomalyDAE, MLPAE

Model training & anomaly detection:
```
python run.py --dataset cora --sample_train True --sample_test False --model multi-scale
```

Anomaly detection results are currently printed to the console and saved in the ./msgad/output folder