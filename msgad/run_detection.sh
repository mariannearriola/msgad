export CUDA_VISIBLE_DEVICES=$3
if [ ${10} == "save" ]
then
    rm -r batch_data/$4
    taskset -c 0-5 python run.py --epoch 1 --datasave True --datadir batch_data/$4 --label_type $7 --model $6 --d 5 --batch_size $2 --dataset $1 --debug $8 --device $9 --lr 1e-4 --vis_filters True --sample_train True --sample_test True
fi
taskset -c 0-5 python run.py --epoch $5 --dataload True --datadir batch_data/$4 --label_type $7 --model $6 --d 5 --batch_size $2 --dataset $1 --debug $8 --device $9 --lr 1e-3 --vis_filters True  --sample_train True --sample_test True