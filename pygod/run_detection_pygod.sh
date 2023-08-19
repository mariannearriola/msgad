WEIGHTS=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# anomaly dae: 0.3
for weight in "${WEIGHTS[@]}"; do
    echo "weight $weight"
    taskset -c 0-5 python3 -m main --weight $weight --data_flag msgad_data --data_dir ../msgad/data --msgad_name weibo --scales 3 --epochs 200 --model anomalydae #--beta_1 1 --beta_2 0 --epochs 100 #--embedding_channels 128 --hidden_channels 256
done