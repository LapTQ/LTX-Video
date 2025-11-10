export HF_HOME="/media/home4/free_space/lap_awlv/cache/huggingface"

PATHD_OUTPUT=/home/lap_awlv/LTX-Video/outputs/inference-6
mkdir -p $PATHD_OUTPUT

ls_prompts=(
"Captured from a static, distant surveillance camera showing the man’s full body, a man stands still in a retail aisle. He picks up a product, looks around cautiously, opens his shoulder bag slightly, carefully places the product inside, then slowly removes his hand from the bag."

"A stationary overhead surveillance camera captures a full-body view of a man standing motionless. He holds a product in both hands, glances left and right, opens his bag flap, drops the product in, and gently pulls his hand out."

"From a fixed surveillance camera at a distance, a man in a retail aisle examines a product, subtly shifts his stance, opens the top zipper of his shoulder bag, places the item in with care, then withdraws his hand while scanning the area."

"A man is captured full-body by a high-mounted, still surveillance camera. He inspects a product, adjusts the strap of his shoulder bag, slips the product in carefully, then lets his hand linger briefly before pulling it out."

"A man, observed from a static surveillance camera, stands in place, opens the side of his shoulder bag, lowers a product in slowly, and then pulls his empty hand back out while glancing around the aisle."

"A wide view from a fixed surveillance camera shows a man standing in place with a shoulder bag. He holds a product, shifts it to one hand, unzips the bag, inserts the item smoothly, then removes his hand in a single motion."

"A full-body surveillance shot shows a man firmly rooted in place. He clutches a product, opens the top flap of his bag, places the product inside deliberately, and withdraws his hand without moving his feet."

"A distant surveillance camera captures a man pretending to inspect a product. He glances up, then opens his shoulder bag cautiously, inserts the item slowly, then takes his hand out as if nothing happened."

"A full-body view from a surveillance camera shows a man in a shoulder bag adjusting his grip on a product, pulling open his bag, dropping the item inside, and lifting his hand out naturally."

"A man in a retail aisle, seen from a fixed surveillance view, holds a product briefly, opens his bag flap with one hand, drops the item in, and slowly retracts his hand while subtly turning his head."

"From an elevated, static surveillance angle, a man standing still shifts the strap of his bag, places a product inside after a moment of hesitation, and slowly retracts his hand."

"Surveillance footage from a fixed position shows a man quietly shifting a product into his shoulder bag, watching around while doing so, then withdrawing his hand without making sudden movements."

"Captured by a fixed surveillance camera, a man inspects a product quietly, leans slightly, places the product into his shoulder bag’s inner compartment, then brings his hand back out."

"A static surveillance camera records a man pretending to browse. He slips a product into his personal bag slowly and smoothly removes his hand after ensuring it’s hidden."
)

idx_prompt=0
for prompt in "${ls_prompts[@]}"; do
    echo "Prompt: $prompt"

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