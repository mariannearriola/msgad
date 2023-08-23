#HS=(1 2)
HS=(2)
#PSIS=(2 4 8)
PSIS=(2)
#LAMBDAS=(0.5 .25 .125 .0625 .03215)
LAMBDAS=(0.5)


for h in "${HS[@]}"; do
    for psi in "${PSIS[@]}"; do
        for lambda in "${LAMBDAS[@]}"; do
            echo "h $h, psi $psi, lambda $lambda"
            python3 -m run --psi $psi --dataset $1 --h $h --lamda $lambda --scales $2
        done
    done
done