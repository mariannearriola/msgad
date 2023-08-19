ALPHAS=(0.3 0.5 0.9)
BETAS=(0.3 0.5 0.9)
SUBGRAPH_SIZES=(2 3 4)


for alpha in "${ALPHAS[@]}"; do
    for beta in "${BETAS[@]}"; do
        for subgraph_size in "${SUBGRAPH_SIZES[@]}"; do
            echo "alpha $alpha, beta $beta, subgraph size $subgraph_size"
            python3 -m run --alpha $alpha --beta $beta --dataset weibo --auc_test_rounds 2 --scales 3 --num_epoch 200 --subgraph_size $subgraph_size
        done
    done
done