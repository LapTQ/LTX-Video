export HF_HOME="/media/home4/free_space/lap_awlv/cache/huggingface"

PATHD_OUTPUT=/home/lap_awlv/LTX-Video/outputs/inference-3
mkdir -p $PATHD_OUTPUT

ls_prompts=(
"A man is standing in a store aisle. From a full-body view, he picks up a product with his right hand, glances to his left, then slowly places the item into the front pocket of his trousers. The camera remains static and distant."
"From a fixed camera angle, a man at a distance holds a product, subtly inspects it while looking over his shoulder, and discreetly slips it into his trouser pocket without changing his position."
"A full-body view shows a man examining a product at waist level, hesitating, then smoothly sliding the item into his front pants pocket while scanning the surroundings. The camera is still."
"A man stands in one spot in a store. He lifts a product, pretends to read its label, checks both sides, and finally tucks it into the pocket of his trousers. The static camera captures the full action from a distance."
"Under a motionless camera view, a man is shown full-body. He picks up an object, turns slightly to his left, then carefully folds it into his right pants pocket while glancing around."
"A distant, fixed camera captures a man facing slightly sideways. He raises a product to chest level, lowers it, and slips it into his trouser pocket while his eyes scan left and right."
"A full-body view from a still camera shows a man holding a product with both hands, pretending to examine it, then shifting one hand to tuck it into his pants pocket while staying still."
"In a retail aisle, a man is seen fully in frame. He fiddles with a product for a few moments, then checks behind him before pushing it into his pocket. The camera doesn’t move."
"A man stands motionless in one place, filmed from afar. He raises an item, pretends to compare it to something unseen, then lowers it directly into his pants pocket. The frame remains unchanged."
"A camera positioned several meters away captures a full-body shot of a man. He gazes forward, casually places a product into his trouser pocket, and continues to glance around suspiciously."
"A static scene shows a man at a safe distance. He holds up a product, rubs his chin as if considering something, then stealthily places the product in his pants pocket while scanning the shelves."
"A distant, fixed camera shows a man standing still. He holds a product, shifts his body weight slightly, and carefully drops the item into his front trouser pocket without changing location."
"A man is fully visible in frame, standing in place. He picks up a product, examines it briefly, then slips it smoothly into his pocket while looking toward the aisle entrance."
"Filmed from a stable camera far away, a man inspects a product in both hands, then slips it discreetly into his pocket while turning slightly left."
"A static camera captures a man’s full body as he holds a product with a relaxed posture, surveys the area, then tucks the product into his pocket quickly but calmly."
"A man in the distance stands under a store light. He adjusts the product in his hand as if unsure, then, in one fluid motion, places it in his pants pocket while barely shifting his feet."
"Without leaving his spot, a man glances at shelves, picks up a product, and after a brief pause, places it inside his trouser pocket. The entire scene is captured by a fixed, distant camera."
"A full-body shot of a man shows him carefully holding an item with both hands, pretending to inspect it, and after a pause, sliding it down into his pocket while scanning the environment."
"A man is seen from far away. He plays with a product in his hand, holds it at waist level, and then carefully inserts it into his pocket without changing stance. The camera stays static."
"From a motionless viewpoint, a man handles a product briefly, pretends to read a label, checks around, then hides it in his pants pocket while standing in the same position."
"A camera captures the full body of a man who slowly picks up a product, weighs it in his hand, and after hesitating, places it inside his trousers pocket with a casual hand gesture."
"A man stands unmoving in a retail aisle. From a fixed, distant view, he picks up a product, inspects it from multiple angles, then hides it in his pocket while subtly looking over his shoulder."
"A static camera records a man standing still. He raises a product up toward his face, pretends to examine it, then quietly lowers it into his front pants pocket."
"A full-body shot from afar shows a man glancing around, lifting a product slowly, then inserting it into his pocket in one smooth and deliberate motion."
"A man stands in one spot, shown full body by a still camera. He examines a product in silence, then subtly moves it to his pocket while glancing in both directions."
"A stationary camera captures a distant man holding a product. He tilts his head as if thinking, then nonchalantly places the item in his pocket while remaining in place."
"From a far and static viewpoint, a man holds a product at waist level, rotates it once, then slides it quickly into his pocket while keeping his feet planted."
"A full-body wide shot captures a man hesitating with a product, briefly hiding it behind his back, and then shifting it into his pants pocket while appearing calm."
"A man, fully visible in the frame, picks up a product, fakes interest by pretending to read it, then moves it to his pocket swiftly. The camera does not follow."
"A far-off, static camera frames a man in the store. He lifts a product slowly, plays with it between his fingers, then moves his hand down and deposits it into his pocket while casually looking around."
)

idx_prompt=0
for prompt in "${ls_prompts[@]}"; do
    prompt="$prompt The camera is far away and static across frames, always capturing the whole body of the person in every frame."
    echo "Prompt: $prompt"

    pathd_output=$PATHD_OUTPUT/prompt_${idx_prompt}
    mkdir -p $pathd_output
    for count in {1..5}; do
        CUDA_VISIBLE_DEVICES=3 python inference.py \
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