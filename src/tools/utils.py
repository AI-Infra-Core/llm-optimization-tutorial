import torch
from tools.logging import logger
from functools import lru_cache

GPU_BF16_PERFORMANCE = {
    "NVIDIA A100-SXM4-80GB": 312
}

@lru_cache(maxsize=1)
def get_gpu_performance():
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name not in GPU_BF16_PERFORMANCE:
        raise ValueError(f"Unsupported GPU: {gpu_name}")
    return GPU_BF16_PERFORMANCE[gpu_name]

class MFUCalculator:
    def __init__(self, model):
        self.call_count = 0
        self.model_params_body = sum(p.numel() for n,p in model.named_parameters() if not ('embed_tokens' in n or "lm_head" in n ))
        self.model_params_final_layer = sum(p.numel() for n,p in model.named_parameters() if "lm_head" in n)
        self.attn_size = model.config.num_attention_heads * model.config.head_dim
        self.model_layer_num = model.config.num_hidden_layers

    def calculate(self, total_seq_length, seq_length_squared_sum, time_this_step):
        FLOPs_linear_body = 2 * self.model_params_body * total_seq_length
        FLOPs_linear_final_layer = 2 * self.model_params_final_layer * total_seq_length
        FLOPs_attn = 3 * seq_length_squared_sum * self.attn_size * self.model_layer_num

        if self.call_count < 8:
            logger.info(f"[MFU Calculator] FLOPs_attn/FLOPs_linear_body: {FLOPs_attn/FLOPs_linear_body:.2f}")
        
        self.call_count += 1

        TFLOPs = (FLOPs_linear_body + FLOPs_linear_final_layer + FLOPs_attn) / 1e12
        mfu = 3 * TFLOPs / time_this_step / get_gpu_performance()
        return mfu
