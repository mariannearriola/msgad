ANCHORS=(25 75 125 175 200)
QS=(2 4 6 8 10 12)
for num_anchors in "${ANCHORS[@]}"; do
    for q in "${QS[@]}"; do
        echo "anchors $num_anchors, q $q"
        taskset -c 0-5 python3 -m main --data_flag msgad_data --data_dir ../msgad/data --msgad_name weibo --num_anchors $num_anchors --q $q --epochs 200 --scales 3 #--embedding_channels 128 --hidden_channels 256
    done
done