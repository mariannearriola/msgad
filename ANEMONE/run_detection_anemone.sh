ALPHAS=(1.0)
SUBGRAPH_SIZES=(2 3 4)


for alpha in "${ALPHAS[@]}"; do
    for subgraph_size in "${SUBGRAPH_SIZES[@]}"; do
        echo "alpha $alpha, subgraph size $subgraph_size"
        python3 -m run --alpha $alpha --subgraph_size $subgraph_size --dataset weibo --auc_test_rounds 2 --runs 1 --scales 3
    done
done