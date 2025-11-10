export HF_HOME="/media/home4/free_space/lap_awlv/cache/huggingface"

PATHD_OUTPUT=/home/lap_awlv/LTX-Video/outputs/inference-8
mkdir -p $PATHD_OUTPUT

ls_prompts=(    
    "In a store aisle, a man seen from a distance holds a product at chest level, looks side to side cautiously, then bends slightly to unzip his shoulder bag and gently insert the item inside."
    
    "In a store aisle, a man positioned looks at a product in his hand, squats slightly to reach into his bag, opens it wide, and slowly lowers the item in while scanning his surroundings."
    
    "A man picks up a product, rotates slightly to face his bag, opens it with one hand, and places the item inside in a smooth, deliberate motion, maintaining his position."
    
    "A man observes a product, turns his shoulder slightly, lifts the flap of his bag, and tucks the product in with care, glancing around once before closing the bag."

    "A man, holds a product for a moment, glances to both sides, then turns slightly and lowers the item into his shoulder bag deliberately."
    
    "A man is standing in one spot. He picks up a product with both hands, glances around cautiously, then slowly slips it into the pocket of his trousers."

    "A man stands still in a store aisle. He inspects a product, lowers it to waist level, and gradually tucks it into his pants pocket while occasionally scanning his surroundings."

    "A man holds a product with one hand, rotates it, pretends to examine it, then stealthily slides it into his trouser pocket while glancing sideways."

    "A  man pretends to read a product label, looks over both shoulders, and discreetly inserts the item into the pocket of his trousers."

    "In a store, a man shifts his weight slightly, clutches a product close to his chest, then quickly and quietly tucks it into his trouser pocket while looking around."
)

idx_prompt=0
for prompt in "${ls_prompts[@]}"; do
    prompt="$prompt Static camera. High angle. Wide shot."
    echo "Prompt $idx_prompt: $prompt"

    pathd_output=$PATHD_OUTPUT/prompt_${idx_prompt}
    mkdir -p $pathd_output
    for count in {1..5}; do
        CUDA_VISIBLE_DEVICES=2 python inference.py \
            --prompt "$prompt" \
            --num_frames 161 \
            --seed 42 \
            --pipeline_config configs/ltxv-13b-0.9.7-distilled.yaml \
            --output_path $pathd_output \
            --frame_rate 16 \
            --height 1280 \
            --width 736 \
            # --conditioning_media_paths gen10.png \
            # --conditioning_start_frames 0 \
            # --offload_to_cpu \

            # --output_path /home/laptq/LTX-Video/outputs/lxtvideo-1.mp4 \

    done
    idx_prompt=$((idx_prompt + 1))
    echo
done