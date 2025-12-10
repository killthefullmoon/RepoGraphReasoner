## Finetuning using Llama Factory

Our specific config file is LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml

If you want recreate our results please run  llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

After the checkpoint is set, please run ： python eval_custom_checkpoint.py --checkpoint #Your Trained Checkpoint#
