export HF_HOME="/media/home4/free_space/lap_awlv/cache/huggingface"

PATHD_OUTPUT=/home/lap_awlv/LTX-Video/outputs/inference-5
mkdir -p $PATHD_OUTPUT

ls_prompts=(
"Captured from a distant static surveillance camera in a retail store, a man is standing in one spot. He picks up a product with both hands, glances around cautiously, then slowly slips it into the pocket of his trousers."

"A fixed-position surveillance camera captures a full-body view of a man standing still in a store aisle. He inspects a product, lowers it to waist level, and gradually tucks it into his pants pocket while occasionally scanning his surroundings."

"From a distant, unmoving surveillance perspective, a man holds a product with one hand, rotates it, pretends to examine it, then stealthily slides it into his trouser pocket while glancing sideways."

"A full-body shot from a static surveillance camera shows a man pretending to read a product label, looks over both shoulders, and discreetly inserts the item into the pocket of his trousers."

"In the static frame of a store surveillance camera, a man shifts his weight slightly, clutches a product close to his chest, then quickly and quietly tucks it into his trouser pocket while looking around."

"A man stands in one place in a store aisle, observed by a distant, fixed surveillance camera. He fidgets with a product nervously, pauses, and then slips it into his pocket, staying alert to nearby movement."

"A stationary surveillance camera at a distance records a man in a shop. The man examines a product, holds it close to his torso, then subtly pushes it into his pocket while watching his surroundings."

"Seen from a high-mounted, non-moving surveillance camera, a man clutches a product tightly, pauses, then pushes it down into his pants pocket while making brief glances in various directions."

"A man stands firmly on the same spot in a retail aisle. Under the distant view of a static surveillance camera, he inspects a product, then quietly inserts it into his trousers pocket while surveying the store."

"A wide-angle fixed surveillance shot shows a man holding a product, faking interest in its label, then calmly tucking it into his pocket with a slight twist of his body."

"A full-body surveillance view captures a man holding a product, tilting it as if inspecting it, then after a pause, slips it into his pocket while maintaining the same physical spot."

"From an overhead fixed surveillance view, a man clutches a product, leans slightly to the side, then with subtle hand movement, hides the product in his trouser pocket."

"A man stands motionless under a static surveillance camera, holding a product. He shifts his gaze left and right, then stealthily moves it into his pocket."

"A static surveillance camera shows a man pretending to inspect a product. He lifts it, pauses mid-air, then gradually hides it in his pants pocket without stepping away."

"From a fixed surveillance perspective, a man raises a product to eye level, then lowers it carefully and discreetly puts it into his pocket with deliberate hand motion."

"A surveillance camera captures a full-body view of a man pretending to compare a product. After a moment, he slowly hides it in his trousers pocket while looking toward the aisle ends."

"A man under a still overhead surveillance camera holds a product close to his hip, then with subtle finger movement, pushes it into his pants pocket while staying still."

"A surveillance camera captures a man from afar. He holds a product at chest level, surveys the area, then tucks it carefully into his trouser pocket with both hands."

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