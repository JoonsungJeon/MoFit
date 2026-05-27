cd "$(dirname "$0")" || exit 1

MAX=1 ## Number of images

while [ "$#" -gt 0 ]; do
    case "$1" in
        --max) MAX="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


echo "Running COCO"

#####################################################
GT_ROOT="path/to/MSCOCO/clean/images/members" 
MEM="member"

# GT_ROOT="path/to/MSCOCO/clean/images/non_members"
# MEM="non_member"

SAVE_DIR="/home/test/MoFit/Results/COCO"

#####################################################


INIT_EMB="blip2"


ckpt_path="path/to/ckpts/sd-MSCOCO-checkpoint"


adv_rnd=1
emb_rnd=1

## timesteps
anchor=14
until=13

## CUDA_VISIBLE_DEVICES
cuda=0




type="Uncond" # "Cond" or or "Uncond"
eps=0.3
step_size=0.15 ## default: 0.15
OptimIter=1000 ## iterations needed for a surrogate
iters=300 ## iterations needed for an embedding
lr=6e-2 



count=0

for img_path in "$GT_ROOT"/*.jpg "$GT_ROOT"/*.png; do
    [ -e "$img_path" ] || continue

    ## Change the start point ##
    # if [ "$count" -lt 0 ]; then
    #     count=$((count + 1))
    #     continue
    # fi
    ## ---------------------- ##

    echo "[sh] Running blip2.py with image: $img_path"

    CUDA_VISIBLE_DEVICES=$cuda python MoFit_COCO.py --ckpt_path "$ckpt_path" --type "$type" --adv_rnd "$adv_rnd" --emb_rnd "$emb_rnd" \
    --anchor "$anchor" --until "$until" --OptimIter "$OptimIter" --iters "$iters" --lr "$lr" --eps "$eps" --img_path "$img_path" --save_dir "$SAVE_DIR" --mem "$MEM" --init "$INIT_EMB" --step_size "$step_size"
   
    count=$((count + 1))
    if [ "$MAX" -ge 0 ] && [ "$count" -ge "$MAX" ]; then
        echo "Reached max count ($MAX), exiting."
        break
    fi
done