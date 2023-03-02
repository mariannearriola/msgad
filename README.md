Currently supported datasets: Cora, T-Finance

Currently supported models: Multi-scale, DOMINANT

Model training & anomaly detection:
```
python run.py --dataset cora_triple_sc_all --sample_train False --sample_test False --model dominant
```

Anomaly detection results are currently printed to the console and saved in the ./msgad/output folder
