export HF_HOME="/media/home4/free_space/lap_awlv/cache/huggingface"

PATHD_OUTPUT=/home/lap_awlv/LTX-Video/outputs/inference-7
mkdir -p $PATHD_OUTPUT

ls_prompts=(    
    "In a store aisle, a man seen from a distance holds a product at chest level, looks side to side cautiously, then bends slightly to unzip his shoulder bag and gently insert the item inside. The camera remains static."
    
    "A man positioned mid-frame in a fixed, distant shot looks at a product in his hand, squats slightly to reach into his bag, opens it wide, and slowly lowers the item in while scanning his surroundings."
    
    "Filmed from a distant static angle, a man picks up a product, rotates slightly to face his bag, opens it with one hand, and places the item inside in a smooth, deliberate motion, maintaining his position."
    
    "Seen from afar in a still camera frame, a man observes a product, turns his shoulder slightly, lifts the flap of his bag, and tucks the product in with care, glancing around once before closing the bag."

    "A man, fully visible from a distance in a static frame, holds a product for a moment, glances to both sides, then turns slightly and lowers the item into his shoulder bag deliberately."
    
    "In a distant, unmoving shot, a man stands near a shelf, takes a product, opens his shoulder bag using one hand, and smoothly lowers the product inside with the other."
    
    "The camera remains stationary as a man grabs a product, stands still, adjusts his bag, and slowly fits the item into the open bag while glancing discreetly around."
    
    "Filmed from afar, a man inspects a product, then leans slightly to one side, opens his shoulder bag using both hands, and places the item inside without stepping away."
        
    "In a fixed wide shot, a man picks up a product, looks cautiously around, unzips his shoulder bag, and places the item in carefully, standing rooted in one spot."
    
    "Seen from a far static camera, a man grabs a product, examines it briefly, then shifts slightly to unzip his shoulder bag and insert the item carefully inside."
)

idx_prompt=0
for prompt in "${ls_prompts[@]}"; do
    prompt="$prompt The camera is far away and static across frames, always capturing the whole body of the person in every frame."
    echo "Prompt $idx_prompt: $prompt"

    pathd_output=$PATHD_OUTPUT/prompt_${idx_prompt}
    mkdir -p $pathd_output
    for count in {1..20}; do
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