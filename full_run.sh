SCALES=$1
DATASET=$2
DEVICE=$3
SAVE=$4

export CUDA_VISIBLE_DEVICES=$DEVICE

./msgad/run_detection.sh $CONFIG $DEVICE $SAVE
./ANEMONE/run_detection_anemone.sh $DATASET $SCALES
./GRADATE/run_detection_gradate.sh $DATASET $SCALES
./subgraph_anomaly_detection_icdm22/run_detection_subgraph.sh $DATASET $SCALES
./pygod/run_detection_pygod.sh $DATASET $SCALES "dominant"
./pygod/run_detection_pygod.sh $DATASET $SCALES "anomalydae"
python msgad/plot_model_wise_hits.py --dataset $DATASET --scales $SCALES