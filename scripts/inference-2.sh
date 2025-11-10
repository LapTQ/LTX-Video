export HF_HOME="/media/home4/free_space/lap_awlv/cache/huggingface"

PATHD_OUTPUT=/home/lap_awlv/LTX-Video/outputs/inference-2
mkdir -p $PATHD_OUTPUT

ls_prompts=(
    "A man stands in the distance inside a store, visible full-body from a fixed camera. He picks up a product with one hand, glances around subtly, opens his shoulder bag with the other hand, and carefully slips the item inside while staying in place."
    
    "In a store aisle, a man seen from a distance holds a product at chest level, looks side to side cautiously, then bends slightly to unzip his shoulder bag and gently insert the item inside. The camera remains static."
    
    "A man positioned mid-frame in a fixed, distant shot looks at a product in his hand, squats slightly to reach into his bag, opens it wide, and slowly lowers the item in while scanning his surroundings."
    
    "Filmed from a distant static angle, a man picks up a product, rotates slightly to face his bag, opens it with one hand, and places the item inside in a smooth, deliberate motion, maintaining his position."
    
    "Seen from afar in a still camera frame, a man observes a product, turns his shoulder slightly, lifts the flap of his bag, and tucks the product in with care, glancing around once before closing the bag."
    
    "A full-body view from a fixed camera shows a man holding a product at waist height, pausing briefly, then slowly placing the item into his shoulder bag, using both hands while staying in the same spot."
    
    "In a wide shot, a man stands stationary. He examines a product, unzips his crossbody bag, shifts his weight to one leg, and slides the item in while keeping his posture relaxed."
    
    "From a distant, stable viewpoint, a man steadies a product in one hand, adjusts the strap of his bag, flips it open, and places the item inside with practiced ease."
    
    "Captured from afar, a man takes a product off a shelf, looks over his shoulder, carefully opens his shoulder bag, and slides the item in, all while staying in the same position."
    
    "A man, fully visible from a distance in a static frame, holds a product for a moment, glances to both sides, then turns slightly and lowers the item into his shoulder bag deliberately."
    
    "From a fixed camera angle, a man grabs a product, subtly checks his surroundings, shifts his torso to the left, opens his shoulder bag, and places the item inside with focused movements."
    
    "A man in a store, fully visible from a static camera placed far away, holds a product close to his chest, turns his upper body slightly, opens his shoulder bag, and drops the item in with care."
    
    "In a distant, unmoving shot, a man stands near a shelf, takes a product, opens his shoulder bag using one hand, and smoothly lowers the product inside with the other."
    
    "The camera remains stationary as a man grabs a product, stands still, adjusts his bag, and slowly fits the item into the open bag while glancing discreetly around."
    
    "Filmed from afar, a man inspects a product, then leans slightly to one side, opens his shoulder bag using both hands, and places the item inside without stepping away."
    
    "A full-body distant view captures a man retrieving a product, then carefully adjusting his bag’s strap, opening it, and placing the item inside while standing in place."
    
    "In a fixed wide shot, a man picks up a product, looks cautiously around, unzips his shoulder bag, and places the item in carefully, standing rooted in one spot."
    
    "A static, far-shot view shows a man holding a product in both hands before sliding it carefully into his shoulder bag, never moving from his standing position."
    
    "From a fixed, distant camera angle, a man lifts a product to inspect it, holds it in one hand while opening his bag, and places it inside, remaining stationary."
    
    "Seen from a far static camera, a man grabs a product, examines it briefly, then shifts slightly to unzip his shoulder bag and insert the item carefully inside."
    
    "A man seen fully in a distant wide shot takes a product from a shelf, holds it momentarily, opens his bag, and smoothly places the product in, remaining in place."
    
    "Captured from a far, motionless angle, a man lifts a product, leans a little, opens his shoulder bag flap with one hand, and places the product in while standing still."
    
    "A man fully in frame from a distance holds a product, slowly turns his upper body to the left, and places the item inside his shoulder bag in a calm motion."
    
    "In a static full-body shot, a man stands firm as he carefully inspects a product, loosens the flap of his shoulder bag, and gently puts the item inside."
    
    "From a wide unmoving frame, a man stands near a shelf, lifts a product with care, adjusts his bag strap, opens the bag, and lowers the item in slowly."
    
    "A man, captured entirely from a distant fixed camera, holds a product mid-air, glances over his shoulder, and places it precisely into his open shoulder bag."
    
    "In a long shot with no camera movement, a man calmly retrieves a product, checks his surroundings, and lowers the item into his shoulder bag without leaving his spot."
    
    "A far shot shows a man taking a product, shifting his weight, opening his bag with a practiced motion, and slipping the product inside while staying put."
    
    "In a still wide frame, a man reaches for a product, inspects it, slowly opens his shoulder bag with one hand, and carefully inserts the item while standing steady."
    
    "From a distant static view, a man stands upright, turns slightly, and with fluid motion places a product into his bag, never changing location."
)

idx_prompt=0
for prompt in "${ls_prompts[@]}"; do
    prompt="$prompt The camera is far away and static across frames, always capturing the whole body of the person in every frame."
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