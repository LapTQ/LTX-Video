export HF_HOME="/mnt/hdd10tb/Users/laptq/cache/huggingface"

CUDA_VISIBLE_DEVICES=0,1,2 python inference.py \
    --prompt "A man strolls through the brightly lit aisles of a modern supermarket, his sleek shoulder bag slung casually across his body. With a focused gaze, he reaches up to a high shelf, plucking a neatly packaged item. Glancing furtively around, he then swiftly yet carefully tucks the product away into his personal bag, mindful of his surroundings as a very high camera captures the scene from an elevated vantage point several meters away." \
    --height 720 \
    --width 1280 \
    --num_frames 49 \
    --seed 42 \
    --pipeline_config configs/ltxv-2b-0.9.yaml \
    --output_path /home/laptq/LTX-Video/outputs/lxtvideo-1.mp4 \
    --frame_rate 8 \
    --offload_to_cpu \
    # --conditioning_media_paths IMAGE_PATH \
    # --conditioning_start_frames 0 \