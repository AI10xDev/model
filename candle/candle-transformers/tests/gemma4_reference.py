"""Generate gemma4_reference.json using the unmodified official HF implementation.

See gemma4_reference.md for installation, provenance, and coverage. This script
never loads pretrained weights, imports Candle, or implements decoder arithmetic.
"""

import copy
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import transformers
from transformers import Gemma4ForCausalLM, Gemma4TextConfig


CONFIG = {
    "model_type": "gemma4_text",
    "hidden_size": 8,
    "intermediate_size": 12,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 4,
    "global_head_dim": 8,
    "attention_k_eq_v": True,
    "vocab_size": 16,
    "sliding_window": 3,
    "max_position_embeddings": 12,
    "layer_types": ["sliding_attention", "full_attention"],
    "final_logit_softcapping": 0.7,
    "hidden_size_per_layer_input": 0,
    "num_kv_shared_layers": 0,
    "enable_moe_block": False,
    "use_double_wide_mlp": False,
    "hidden_activation": "gelu_pytorch_tanh",
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "tie_word_embeddings": True,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {
            "rope_type": "proportional",
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
        },
    },
}
TOKENS = [1, 7, 3, 10, 2, 8, 4, 6, 5]
PREFILLS = [1, 2, 4, 7]


def weight(name, shape):
    # Exactly the wrapping-u32 name seed from gemma4_tests.rs::weights.
    seed = 17
    for byte in name.encode("ascii"):
        seed = (seed * 31 + byte) & 0xFFFFFFFF
    values = []
    for i in range(shape.numel()):
        n = (seed + i * 747796405) & 0xFFFFFFFF
        n = ((n ^ (n >> 16)) * 2246822519) & 0xFFFFFFFF
        value = ((n ^ (n >> 13)) % 200) + 1
        values.append(value if "norm" in name else (-value if n & 256 == 0 else value))
    # Perform each operation in float32, including the rounding of 0.6.
    result = torch.tensor(values, dtype=torch.float32)
    result = 0.6 + result / 250.0 if "norm" in name else result / 1000.0
    return result.reshape(shape)


@torch.inference_mode()
def main():
    assert transformers.__version__ == "5.19.0"
    assert torch.__version__ == "2.8.0+cpu"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)  # All persistent state is replaced below, including buffers.
    cases = []
    for name in ["tied", "untied", "half_rope_negative_scalars", "separate_kv_ignores_global_override"]:
        config = copy.deepcopy(CONFIG)
        scalars = [0.8, 1.3]
        if name == "untied":
            config["tie_word_embeddings"] = False
        if name == "separate_kv_ignores_global_override":
            config["attention_k_eq_v"] = False
            assert config["num_global_key_value_heads"] != config["num_key_value_heads"]
        if name == "half_rope_negative_scalars":
            config["rope_parameters"]["full_attention"].update(
                partial_rotary_factor=0.5, rope_theta=100.0
            )
            config["rope_parameters"]["sliding_attention"]["rope_theta"] = 100.0
            scalars = [-0.6, -1.1]
        model = Gemma4ForCausalLM(
            Gemma4TextConfig(**copy.deepcopy(config), attn_implementation="eager")
        ).float().cpu().eval()
        state = {}
        for key, tensor in model.state_dict().items():
            relative = key.removeprefix("model.")
            if relative.endswith("layer_scalar"):
                state[key] = torch.tensor([scalars[int(relative.split(".")[1])]])
            elif key == "lm_head.weight" and config["tie_word_embeddings"]:
                state[key] = weight("embed_tokens.weight", tensor.shape)
            else:
                assert relative.endswith(".weight"), relative
                state[key] = weight(relative, tensor.shape)
        model.load_state_dict(state, strict=True)
        assert model.config._attn_implementation == "eager"
        assert model.model.layers[0].self_attn.v_proj is not None
        global_attention = model.model.layers[1].self_attn
        if config["attention_k_eq_v"]:
            assert global_attention.v_proj is None
            assert global_attention.k_proj.weight.shape == (8, 8)
        else:
            assert global_attention.v_proj is not None
            assert global_attention.k_proj.weight.shape == (16, 8)
            assert global_attention.v_proj.weight.shape == (16, 8)
        if config["tie_word_embeddings"]:
            assert model.lm_head.weight is model.model.embed_tokens.weight

        def forward(tokens, cache=None, use_cache=False):
            return model(
                input_ids=torch.tensor([tokens]),
                past_key_values=cache,
                use_cache=use_cache,
                logits_to_keep=1,
            )

        def row(output):
            assert output.logits.shape == (1, 1, 16)
            assert torch.isfinite(output.logits).all()
            return output.logits[0, 0].tolist()

        fresh = [row(forward(TOKENS[:end])) for end in range(1, len(TOKENS) + 1)]
        cached = []
        for prefill in PREFILLS:
            output = forward(TOKENS[:prefill], use_cache=True)
            logits = [row(output)]
            for offset in range(prefill, len(TOKENS)):
                output = forward([TOKENS[offset]], output.past_key_values, use_cache=True)
                logits.append(row(output))
            torch.testing.assert_close(
                torch.tensor(logits), torch.tensor(fresh[prefill - 1:]), atol=2e-6, rtol=0
            )
            cached.append({"prefill": prefill, "logits": logits})
        cases.append({
            "name": name, "config": config, "layer_scalars": scalars,
            "fresh": fresh, "cached": cached,
        })
    fixture = {
        "provenance": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "source": "https://github.com/huggingface/transformers/tree/v5.19.0/src/transformers/models/gemma4",
            "backend": "CPU float32, eager attention, eval, inference_mode, one thread",
            "weights": "gemma4_tests.rs wrapping-u32 relative-name algorithm; no pretrained model",
        },
        "tokens": TOKENS,
        "cases": cases,
    }
    # A generated fixture is the only file this script writes.
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(f"Generated {destination}: {len(cases)} cases, official HF logits")


if __name__ == "__main__":
    main()
