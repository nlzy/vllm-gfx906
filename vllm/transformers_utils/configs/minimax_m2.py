# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Configuration for MiniMax M2 and M2.1 models
# This config bypasses HuggingFace auto_map to allow loading without trust_remote_code
from transformers import PretrainedConfig


class MiniMaxM2Config(PretrainedConfig):
    model_type = "minimax_m2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=3072,
        intermediate_size=1536,
        mlp_intermediate_size=8192,
        num_hidden_layers=62,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=196608,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        use_qk_norm=True,
        qk_norm_type="per_layer",
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        rotary_dim=64,
        sliding_window=None,
        # MoE parameters
        num_local_experts=256,
        num_experts_per_tok=8,
        use_routing_bias=True,
        scoring_func="sigmoid",
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        shared_intermediate_size=0,
        shared_moe_mode="sigmoid",
        # MTP (Multi-Token Prediction) parameters
        use_mtp=False,
        num_mtp_modules=0,
        mtp_transformer_layers=1,
        # Layer normalization
        layernorm_full_attention_beta=1.0,
        layernorm_linear_attention_beta=1.0,
        layernorm_mlp_beta=1.0,
        # Attention type list
        attn_type_list=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mlp_intermediate_size = mlp_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.rotary_dim = rotary_dim
        self.sliding_window = sliding_window

        # MoE
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_routing_bias = use_routing_bias
        self.scoring_func = scoring_func
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.shared_intermediate_size = shared_intermediate_size
        self.shared_moe_mode = shared_moe_mode

        # MTP
        self.use_mtp = use_mtp
        self.num_mtp_modules = num_mtp_modules
        self.mtp_transformer_layers = mtp_transformer_layers

        # Layer normalization
        self.layernorm_full_attention_beta = layernorm_full_attention_beta
        self.layernorm_linear_attention_beta = layernorm_linear_attention_beta
        self.layernorm_mlp_beta = layernorm_mlp_beta

        # Attention type list (uniform for all layers by default)
        if attn_type_list is None:
            self.attn_type_list = [1] * num_hidden_layers
        else:
            self.attn_type_list = attn_type_list

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
