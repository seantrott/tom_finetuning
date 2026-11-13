for m in allenai/OLMo-2-0425-1B allenai/OLMo-2-1124-7B \
         Qwen/Qwen2.5-7B meta-llama/Llama-3.1-8B mistralai/Mistral-7B-v0.3 \
         EleutherAI/pythia-6.9b; do
  python src/finetune_transfer.py --model $m --data_dir data/raw \
      --out results/$(basename $m) --seeds 0 1 2 --tasks FB1 FB2 IR FB_attn --micro_batch 1
done