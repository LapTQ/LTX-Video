export HF_HOME="/media/home4/free_space/lap_awlv/cache/huggingface"

mkdir -p /home/lap_awlv/LTX-Video/outputs

CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py \
    --prompt "A man strolls through the brightly lit aisles of a modern supermarket, his sleek shoulder bag slung casually across his body. With a focused gaze, he reaches up to a high shelf, plucking a neatly packaged item. Glancing furtively around, he then swiftly yet carefully tucks the product away into his personal bag, mindful of his surroundings as a very high camera captures the scene from an elevated vantage point several meters away." \
    --height 1280 \
    --width 736 \
    --num_frames 49 \
    --seed 42 \
    --pipeline_config configs/ltxv-13b-0.9.7-distilled.yaml \
    --output_path /home/lap_awlv/LTX-Video/outputs \
    --frame_rate 8 \
    --conditioning_media_paths gen10.png \
    --conditioning_start_frames 0 \
    # --offload_to_cpu \

    # --output_path /home/laptq/LTX-Video/outputs/lxtvideo-1.mp4 \