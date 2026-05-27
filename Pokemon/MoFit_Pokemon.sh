cd "$(dirname "$0")" || exit 1

MAX=1 ## Number of images

while [ "$#" -gt 0 ]; do
    case "$1" in
        --max) MAX="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Running Pokemon"

#####################################################
GT_ROOT="path/to/Pokemon/clean/images/members"
MEM="member"

# GT_ROOT="path/to/Pokemon/clean/images/non_members"
# MEM="non_member"

SAVE_DIR="/home/test/MoFit/Results/Pokemon"

#####################################################

INIT_EMB="clip_interrogator"

ckpt_path="path/to/ckpts/sd-pokemon-checkpoint"


adv_rnd=1
emb_rnd=1


cuda=1


anchor=14
until=13

## ========== Default ========== ##
type="Uncond" # "Cond" or or "Uncond"
eps=0.3
step_size=0.15 
OptimIter=1000
iters=200
lr=6e-2



count=0

for img_path in "$GT_ROOT"/*.jpg "$GT_ROOT"/*.png; do
    [ -e "$img_path" ] || continue

    # if [ "$count" -lt 94 ]; then
    #     count=$((count + 1))
    #     continue
    # fi

    echo "[sh] Running blip2.py with image: $img_path"

    CUDA_VISIBLE_DEVICES=$cuda python MoFit_Pokemon.py --ckpt_path "$ckpt_path" --type "$type" --adv_rnd "$adv_rnd" --emb_rnd "$emb_rnd" \
    --anchor "$anchor" --until "$until" --OptimIter "$OptimIter" --iters "$iters" --lr "$lr" --eps "$eps" --img_path "$img_path" --save_dir "$SAVE_DIR" --mem "$MEM" --init "$INIT_EMB" --step_size "$step_size"
   
    
    count=$((count + 1))
    if [ "$MAX" -ge 0 ] && [ "$count" -ge "$MAX" ]; then
        echo "Reached max count ($MAX), exiting."
        break
    fi
done