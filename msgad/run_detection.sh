export CUDA_VISIBLE_DEVICES=$2
if [ $3 == "save" ]
then
    rm -r batch_data/$4
    taskset -c 0-5 python run.py --epoch 1 --datasave True --config $1
fi
taskset -c 0-15 python run.py --dataload True --config $1