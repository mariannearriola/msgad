export CUDA_VISIBLE_DEVICES=$2
if [ $3 == "True" ]
then
    taskset -c 0-1 python run.py --epoch 1 --datasave True --config $1
fi
taskset -c 0-1 python run.py --dataload True --config $1