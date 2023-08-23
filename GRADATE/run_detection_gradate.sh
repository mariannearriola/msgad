ALPHAS=(0.5)
BETAS=(0.3)
SUBGRAPH_SIZES=(2)


for alpha in "${ALPHAS[@]}"; do
    for beta in "${BETAS[@]}"; do
        for subgraph_size in "${SUBGRAPH_SIZES[@]}"; do
            echo "alpha $alpha, beta $beta, subgraph size $subgraph_size"
            python3 -m run --alpha $alpha --beta $beta --dataset $1 --auc_test_rounds 2 --scales $2 --num_epoch 1000 --subgraph_size $subgraph_size
        done
    done
done