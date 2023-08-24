ANCHORS=(25)
for num_anchors in "${ANCHORS[@]}"; do
    echo "anchors $num_anchors"
    taskset -c 0-5 python3 -m main --data_flag msgad_data --data_dir ../msgad/data --dataset $1 --num_anchors $num_anchors --scales $2 --epochs 150
done