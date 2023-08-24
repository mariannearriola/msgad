WEIGHTS=(0.3)
for weight in "${WEIGHTS[@]}"; do
    echo "weight $weight"
    taskset -c 0-5 python3 -m main --weight $weight --dataset $1 --scales $2 --epochs 150 --model $3 --data_dir ../msgad/data
done