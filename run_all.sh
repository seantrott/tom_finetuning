for m in Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B allenai/OLMo-2-0425-1B \
         allenai/OLMo-2-1124-7B Qwen/Qwen2.5-7B EleutherAI/pythia-6.9b \
         allenai/OLMo-2-1124-13B; do
  python src/finetune_transfer.py --model $m --data_dir data/raw \
      --out results/$(basename $m) --seeds 0 1 2 --tasks FB1 FB2 IR FB_attn --micro_batch 1
done