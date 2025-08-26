import json
import argparse
from typing import Dict, Any, Tuple
from dataclasses import dataclass
 
 
class Qwen3SparseMoeparamCalculator:
     
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
         
        self.vocab_size = self.config.get('vocab_size', 0)
        self.hidden_size = self.config.get('hidden_size', 0)
 
        self.num_layers = self.config.get('num_hidden_layers', 0)
        self.num_attention_heads = self.config.get('num_attention_heads', 0)
        self.num_key_value_heads = self.config.get('num_key_value_heads', 0)
        self.head_dim = self.config.get('head_dim', 0)
 
        self.num_experts = self.config.get('num_experts', 0)
        self.num_experts_per_tok = self.config.get('num_experts_per_tok', 0)
        self.moe_intermediate_size = self.config.get('moe_intermediate_size', 0)
 
    def _embedding_param(self):
        return self.vocab_size * self.hidden_size
 
    def _attention_param(self):
        q_proj_param = self.hidden_size * self.num_attention_heads * self.head_dim
        k_proj_param = self.hidden_size * self.num_key_value_heads * self.head_dim
        v_proj_param = k_proj_param
        o_proj_param = self.num_attention_heads * self.head_dim * self.hidden_size
        attenton_param = q_proj_param + k_proj_param + v_proj_param + o_proj_param
        return attenton_param
     
    def _gate_param(self):
        gate_param = self.hidden_size * self.num_experts
        return gate_param
     
    def _expert_weight_param(self):
        gate_up_proj_param = 2 * self.hidden_size * self.moe_intermediate_size
        down_proj_param = self.moe_intermediate_size * self.hidden_size
        expert_weight_param = gate_up_proj_param + down_proj_param
        return expert_weight_param * self.num_experts
 
    def _moe_param(self):
        gate_param = self._gate_param()
        moe_param = gate_param + self._expert_weight_param()
        return moe_param
     
    def total_param(self):
        embedding_param = self._embedding_param()
        attention_param = self._attention_param()
        moe_param = self._moe_param()
 
        decoder_param = (attention_param + moe_param) * self.num_layers
 
        total_param = embedding_param + decoder_param
 
        return total_param
     
    def total_weight(self):
        total_weight = self.total_param() * 2
        return total_weight
 
    def quantized_weight(self):
        embedding_param = self._embedding_param() * 2
        # "weight_precision": "int8"
        attention_weight = self._attention_param() * 1
        # "expert_weight_precision": "int4"
        expert_weight = self._expert_weight_param() * 0.5
        moe_weight = expert_weight + self._gate_param() * 2
 
        decoder_weight = (attention_weight + moe_weight) * self.num_layers
        total_weight = embedding_param + decoder_weight
        return total_weight
         
         
def main():
    parser = argparse.ArgumentParser(description='Calculate model memory usage')
    parser.add_argument('config_path', nargs='?', default='/data/models/Qwen3-30B-A3B/config.json')
     
    args = parser.parse_args()
     
    calculator = Qwen3SparseMoeparamCalculator(args.config_path)
     
    total_param = calculator.total_param()
    print(f"total_param: {total_param / 10**9} B")
 
    total_weight = calculator.total_weight()
    print(f"total_weight: {total_weight / 2**30} GB")
     
    quantized_weight = calculator.quantized_weight()
    print(f"quantized_weight: {quantized_weight / 2**30} GB")
     
if __name__ == "__main__":
    main()
