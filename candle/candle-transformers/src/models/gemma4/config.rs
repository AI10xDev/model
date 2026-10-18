//! Configuration for the Gemma 4 multimodal model.

use candle_nn::Activation;

// ── Text config defaults ────────────────────────────────────────────────────

fn default_attention_bias() -> bool {
    false
}
fn default_head_dim() -> usize {
    256
}
fn default_hidden_activation() -> Activation {
    Activation::GeluPytorchTanh
}
fn default_num_attention_heads() -> usize {
    8
}
fn default_num_key_value_heads() -> usize {
    4
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.
}
fn default_vocab_size() -> usize {
    262144
}
fn default_query_pre_attn_scalar() -> usize {
    256
}
fn default_max_position_embeddings() -> usize {
    131072
}
fn default_tie_word_embeddings() -> bool {
    true
}
fn default_sliding_window_pattern() -> usize {
    6
}
fn default_global_head_dim() -> usize {
    512
}
fn default_use_flash_attn() -> bool {
    false
}

// ── Rope parameters ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4RopeLayerParams {
    pub rope_theta: Option<f64>,
    pub rope_type: Option<String>,
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4RopeParameters {
    pub full_attention: Option<Gemma4RopeLayerParams>,
    pub sliding_attention: Option<Gemma4RopeLayerParams>,
    pub rope_theta: Option<f64>,
    pub rope_type: Option<String>,
    pub partial_rotary_factor: Option<f64>,
}

// ── Gemma4TextConfig ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4TextConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub final_logit_softcapping: Option<f64>,
    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(
        default = "default_sliding_window_pattern",
        alias = "_sliding_window_pattern"
    )]
    pub sliding_window_pattern: usize,
    pub layer_types: Vec<String>,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: usize,
    pub num_global_key_value_heads: Option<usize>,
    pub rope_parameters: Option<Gemma4RopeParameters>,
    pub use_bidirectional_attention: Option<String>,
    #[serde(default = "default_use_flash_attn")]
    pub use_flash_attn: bool,
}

impl Gemma4TextConfig {
    /// Validate the supported dense, text-only subset before allocating weights.
    pub fn validate(&self) -> candle::Result<()> {
        let mut unsupported = Vec::new();
        if !matches!(self.model_type.as_deref(), None | Some("gemma4_text")) {
            unsupported.push("model_type (only gemma4_text is supported; Unified is unsupported)");
        }
        if self.hidden_size_per_layer_input != 0 {
            unsupported.push("hidden_size_per_layer_input (per-layer embeddings/PLE)");
        }
        if self.num_kv_shared_layers != 0 {
            unsupported.push("num_kv_shared_layers (cross-layer KV sharing)");
        }
        if self.enable_moe_block {
            unsupported.push("enable_moe_block (MoE)");
        }
        if self.use_double_wide_mlp {
            unsupported.push("use_double_wide_mlp (double-wide MLP)");
        }
        if !matches!(
            self.use_bidirectional_attention.as_deref(),
            None | Some("vision")
        ) {
            unsupported.push(
                "use_bidirectional_attention (only absent or vision is supported, not all)",
            );
        }
        if !unsupported.is_empty() {
            candle::bail!(
                "unsupported Gemma4 text configuration: {}",
                unsupported.join(", ")
            );
        }

        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("vocab_size", self.vocab_size),
            ("sliding_window", self.sliding_window),
            ("sliding_window_pattern", self.sliding_window_pattern),
            ("max_position_embeddings", self.max_position_embeddings),
            ("query_pre_attn_scalar", self.query_pre_attn_scalar),
        ] {
            if value == 0 {
                candle::bail!("Gemma4 {field} must be positive");
            }
        }
        if self.max_position_embeddings > u32::MAX as usize {
            candle::bail!("Gemma4 max_position_embeddings must fit in u32 for RoPE positions");
        }
        for (field, value) in [
            ("head_dim", self.head_dim),
            ("global_head_dim", self.global_head_dim),
        ] {
            if value == 0 || value % 2 != 0 {
                candle::bail!("Gemma4 {field} must be nonzero and even for RoPE");
            }
        }
        for (field, heads, dim) in [
            ("num_key_value_heads", self.num_key_value_heads, self.head_dim),
            (
                "num_global_key_value_heads",
                self.num_global_key_value_heads
                    .unwrap_or(self.num_key_value_heads),
                self.global_head_dim,
            ),
        ] {
            if heads == 0 || self.num_attention_heads % heads != 0 {
                candle::bail!("Gemma4 {field} must be positive and divide num_attention_heads");
            }
            if self.num_attention_heads.checked_mul(dim).is_none()
                || heads.checked_mul(dim).is_none()
            {
                candle::bail!("Gemma4 attention projection dimensions overflow for {field}");
            }
        }
        if self.layer_types.len() != self.num_hidden_layers {
            candle::bail!("Gemma4 layer_types must have exactly num_hidden_layers entries");
        }
        for (index, layer_type) in self.layer_types.iter().enumerate() {
            if !matches!(layer_type.as_str(), "sliding_attention" | "full_attention") {
                candle::bail!("Gemma4 layer_types[{index}] has unknown value {layer_type:?}");
            }
        }
        for (field, value) in [
            ("rms_norm_eps", Some(self.rms_norm_eps)),
            ("final_logit_softcapping", self.final_logit_softcapping),
            ("rope_theta", Some(self.rope_theta)),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value <= 0.0 {
                    candle::bail!("Gemma4 {field} must be finite and positive");
                }
            }
        }
        if let Some(rp) = &self.rope_parameters {
            // Flat text RoPE parameters are global fallbacks; sliding RoPE is standard.
            for (field, theta, rope_type, factor, sliding) in [
                (
                    "rope_parameters",
                    rp.rope_theta,
                    rp.rope_type.as_deref(),
                    rp.partial_rotary_factor,
                    false,
                ),
                (
                    "rope_parameters.full_attention",
                    rp.full_attention.as_ref().and_then(|p| p.rope_theta),
                    rp.full_attention.as_ref().and_then(|p| p.rope_type.as_deref()),
                    rp.full_attention
                        .as_ref()
                        .and_then(|p| p.partial_rotary_factor),
                    false,
                ),
                (
                    "rope_parameters.sliding_attention",
                    rp.sliding_attention.as_ref().and_then(|p| p.rope_theta),
                    rp.sliding_attention
                        .as_ref()
                        .and_then(|p| p.rope_type.as_deref()),
                    rp.sliding_attention
                        .as_ref()
                        .and_then(|p| p.partial_rotary_factor),
                    true,
                ),
            ] {
                if let Some(theta) = theta {
                    if !theta.is_finite() || theta <= 0.0 {
                        candle::bail!("Gemma4 {field}.rope_theta must be finite and positive");
                    }
                }
                if let Some(rope_type) = rope_type {
                    if rope_type != "default" && (sliding || rope_type != "proportional") {
                        candle::bail!("unsupported Gemma4 {field}.rope_type: {rope_type:?}");
                    }
                }
                if let Some(factor) = factor {
                    if !factor.is_finite() || factor <= 0.0 || factor > 1.0 {
                        candle::bail!("Gemma4 {field}.partial_rotary_factor must be in (0, 1]");
                    }
                    if (sliding || rope_type == Some("default")) && factor != 1.0 {
                        candle::bail!(
                            "Gemma4 {field}.partial_rotary_factor must be 1 for standard RoPE"
                        );
                    }
                }
            }
            let global_type = rp
                .full_attention
                .as_ref()
                .and_then(|p| p.rope_type.as_deref())
                .or(rp.rope_type.as_deref());
            if global_type == Some("default") && self.partial_rotary_factor() != 1.0 {
                candle::bail!(
                    "Gemma4 full_attention.partial_rotary_factor must be 1 for default rope_type"
                );
            }
        }
        let factor = self.partial_rotary_factor();
        let rotary_dim = self.global_head_dim as f64 * factor;
        if !factor.is_finite()
            || factor <= 0.0
            || factor > 1.0
            || rotary_dim < 2.0
            || rotary_dim % 2.0 != 0.0
        {
            candle::bail!(
                "Gemma4 full_attention.partial_rotary_factor must be in (0, 1] and produce a nonzero even rotary dimension with global_head_dim"
            );
        }
        Ok(())
    }

    pub fn effective_sliding_window(&self) -> usize {
        if self.use_bidirectional_attention.as_deref() == Some("all") {
            (self.sliding_window / 2) + 1
        } else {
            self.sliding_window
        }
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| {
                let full = rp.full_attention.as_ref();
                full.and_then(|p| p.partial_rotary_factor)
                    .or(rp.partial_rotary_factor)
                    .or_else(|| {
                        let rope_type = full
                            .and_then(|p| p.rope_type.as_deref())
                            .or(rp.rope_type.as_deref());
                        (rope_type == Some("default")).then_some(1.0)
                    })
            })
            .unwrap_or(0.25)
    }

    pub fn rope_global_base_freq(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| {
                rp.full_attention
                    .as_ref()
                    .and_then(|fa| fa.rope_theta)
                    .or(rp.rope_theta)
            })
            .unwrap_or(self.rope_theta)
    }

    pub fn rope_local_base_freq(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.sliding_attention.as_ref())
            .and_then(|sa| sa.rope_theta)
            .unwrap_or(10000.0)
    }

    pub fn is_sliding(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|s| s == "sliding_attention")
            .unwrap_or(false)
    }
}

// ── Vision config defaults ──────────────────────────────────────────────────

fn default_vision_hidden_size() -> usize {
    768
}
fn default_vision_intermediate_size() -> usize {
    3072
}
fn default_vision_num_hidden_layers() -> usize {
    16
}
fn default_vision_num_attention_heads() -> usize {
    12
}
fn default_vision_num_key_value_heads() -> usize {
    12
}
fn default_vision_head_dim() -> usize {
    64
}
fn default_vision_hidden_activation() -> Activation {
    Activation::GeluPytorchTanh
}
fn default_vision_rms_norm_eps() -> f64 {
    1e-6
}
fn default_vision_patch_size() -> usize {
    16
}
fn default_vision_position_embedding_size() -> usize {
    10240
}
fn default_vision_pooling_kernel_size() -> usize {
    3
}
fn default_vision_default_output_length() -> usize {
    280
}
fn default_vision_standardize() -> bool {
    false
}

// ── Gemma4VisionConfig ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_vision_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_vision_hidden_activation")]
    pub hidden_activation: Activation,
    #[serde(default = "default_vision_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_position_embedding_size")]
    pub position_embedding_size: usize,
    #[serde(default = "default_vision_pooling_kernel_size")]
    pub pooling_kernel_size: usize,
    #[serde(default = "default_vision_default_output_length")]
    pub default_output_length: usize,
    #[serde(default = "default_vision_standardize")]
    pub standardize: bool,
    pub rope_parameters: Option<Gemma4RopeParameters>,
}

impl Gemma4VisionConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| {
                rp.full_attention
                    .as_ref()
                    .and_then(|fa| fa.rope_theta)
                    .or(rp.rope_theta)
            })
            .unwrap_or(100.0)
    }
}

// ── Audio config defaults ───────────────────────────────────────────────────

fn default_audio_input_feat_size() -> usize {
    128
}
fn default_audio_hidden_size() -> usize {
    1024
}
fn default_conf_attention_chunk_size() -> usize {
    12
}
fn default_conf_attention_context_left() -> usize {
    13
}
fn default_conf_attention_context_right() -> usize {
    0
}
fn default_conf_attention_invalid_logits_value() -> f64 {
    -1e9
}
fn default_conf_attention_logit_cap() -> f64 {
    50.0
}
fn default_conf_num_attention_heads() -> usize {
    8
}
fn default_conf_num_hidden_layers() -> usize {
    12
}
fn default_conf_conv_kernel_size() -> usize {
    5
}
fn default_conf_reduction_factor() -> usize {
    1
}
fn default_conf_residual_weight() -> f64 {
    0.5
}
fn default_sscp_conv_channel_size() -> Vec<usize> {
    vec![128, 32]
}
fn default_sscp_conv_kernel_size() -> Vec<Vec<usize>> {
    vec![vec![3, 3], vec![3, 3]]
}
fn default_sscp_conv_stride_size() -> Vec<Vec<usize>> {
    vec![vec![2, 2], vec![2, 2]]
}
fn default_audio_vocab_size() -> usize {
    128
}
fn default_sscp_conv_group_norm_eps() -> f64 {
    1e-6
}
fn default_sscp_conv_eps() -> f64 {
    1e-3
}
fn default_audio_rms_norm_eps() -> f64 {
    1e-6
}
fn default_gradient_clipping() -> f64 {
    1e10
}
fn default_output_proj_dims() -> Option<usize> {
    Some(1536)
}

// ── Gemma4AudioConfig ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4AudioConfig {
    #[serde(default = "default_audio_input_feat_size")]
    pub input_feat_size: usize,
    #[serde(default = "default_audio_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_output_proj_dims")]
    pub output_proj_dims: Option<usize>,
    #[serde(
        default = "default_conf_attention_chunk_size",
        alias = "attention_chunk_size"
    )]
    pub conf_attention_chunk_size: usize,
    #[serde(
        default = "default_conf_attention_context_left",
        alias = "attention_context_left"
    )]
    pub conf_attention_context_left: usize,
    #[serde(
        default = "default_conf_attention_context_right",
        alias = "attention_context_right"
    )]
    pub conf_attention_context_right: usize,
    #[serde(
        default = "default_conf_attention_invalid_logits_value",
        alias = "attention_invalid_logits_value"
    )]
    pub conf_attention_invalid_logits_value: f64,
    #[serde(
        default = "default_conf_attention_logit_cap",
        alias = "attention_logit_cap"
    )]
    pub conf_attention_logit_cap: f64,
    #[serde(
        default = "default_conf_num_attention_heads",
        alias = "num_attention_heads"
    )]
    pub conf_num_attention_heads: usize,
    #[serde(
        default = "default_conf_num_hidden_layers",
        alias = "num_hidden_layers"
    )]
    pub conf_num_hidden_layers: usize,
    #[serde(default = "default_conf_conv_kernel_size", alias = "conv_kernel_size")]
    pub conf_conv_kernel_size: usize,
    #[serde(default = "default_conf_reduction_factor")]
    pub conf_reduction_factor: usize,
    #[serde(default = "default_conf_residual_weight", alias = "residual_weight")]
    pub conf_residual_weight: f64,
    #[serde(
        default = "default_sscp_conv_channel_size",
        alias = "subsampling_conv_channels"
    )]
    pub sscp_conv_channel_size: Vec<usize>,
    #[serde(default = "default_sscp_conv_kernel_size")]
    pub sscp_conv_kernel_size: Vec<Vec<usize>>,
    #[serde(default = "default_sscp_conv_stride_size")]
    pub sscp_conv_stride_size: Vec<Vec<usize>>,
    #[serde(default = "default_audio_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_sscp_conv_group_norm_eps")]
    pub sscp_conv_group_norm_eps: f64,
    #[serde(default = "default_sscp_conv_eps")]
    pub sscp_conv_eps: f64,
    #[serde(default = "default_audio_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_gradient_clipping")]
    pub gradient_clipping: f64,
}

// ── Top-level config defaults ───────────────────────────────────────────────

fn default_image_token_id() -> usize {
    258880
}
fn default_audio_token_id() -> usize {
    258881
}
fn default_video_token_id() -> usize {
    258884
}

// ── Gemma4Config ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
    pub vision_config: Gemma4VisionConfig,
    pub audio_config: Option<Gemma4AudioConfig>,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: usize,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: usize,
    #[serde(default = "default_video_token_id")]
    pub video_token_id: usize,
}

#[cfg(test)]
mod tests {
    use super::Gemma4TextConfig;
    use serde_json::{json, Value};

    fn config(overrides: Value) -> Gemma4TextConfig {
        let mut fixture = json!({
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 16,
            "vocab_size": 32,
            "sliding_window": 8,
            "max_position_embeddings": 32,
            "layer_types": ["sliding_attention", "full_attention"]
        });
        fixture
            .as_object_mut()
            .unwrap()
            .extend(overrides.as_object().unwrap().clone());
        serde_json::from_value(fixture).unwrap()
    }

    #[test]
    fn dense_defaults_and_k_equals_v_are_supported() {
        let cfg = config(json!({}));
        cfg.validate().unwrap();
        assert_eq!(cfg.model_type, None);
        assert!(!cfg.attention_k_eq_v);
        assert_eq!(cfg.hidden_size_per_layer_input, 0);
        assert_eq!(cfg.num_kv_shared_layers, 0);
        assert!(!cfg.enable_moe_block);
        assert!(!cfg.use_double_wide_mlp);

        let cfg = config(json!({
            "model_type": "gemma4_text",
            "attention_k_eq_v": true,
            "num_global_key_value_heads": 1,
            "use_bidirectional_attention": "vision",
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25
                },
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
            }
        }));
        cfg.validate().unwrap();
        assert!(cfg.attention_k_eq_v);
        assert_eq!(cfg.effective_sliding_window(), 8);
    }

    #[test]
    fn real_variant_flags_are_reported_together() {
        let cfg = config(json!({
            "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 18,
            "enable_moe_block": true,
            "use_double_wide_mlp": true
        }));
        let error = cfg.validate().unwrap_err().to_string();
        for field in [
            "hidden_size_per_layer_input",
            "num_kv_shared_layers",
            "enable_moe_block",
            "use_double_wide_mlp",
        ] {
            assert!(error.contains(field), "{error}");
        }
    }

    #[test]
    fn nested_rope_parameters_and_legacy_fallbacks() {
        let cfg = config(json!({"rope_theta": 12345.0}));
        assert_eq!(cfg.rope_global_base_freq(), 12345.0);
        assert_eq!(cfg.rope_local_base_freq(), 10000.0);
        assert_eq!(cfg.partial_rotary_factor(), 0.25);

        let cfg = config(json!({
            "rope_theta": 12345.0,
            "rope_parameters": {
                "rope_theta": 23456.0,
                "full_attention": {"rope_theta": 34567.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_theta": 45678.0}
            }
        }));
        cfg.validate().unwrap();
        assert_eq!(cfg.rope_global_base_freq(), 34567.0);
        assert_eq!(cfg.rope_local_base_freq(), 45678.0);
        assert_eq!(cfg.partial_rotary_factor(), 0.5);

        let cfg = config(json!({
            "rope_parameters": {"rope_theta": 23456.0, "rope_type": "default"}
        }));
        cfg.validate().unwrap();
        assert_eq!(cfg.rope_global_base_freq(), 23456.0);
        assert_eq!(cfg.partial_rotary_factor(), 1.0);
    }

    #[test]
    fn malformed_shapes_and_types_are_rejected() {
        for (overrides, field) in [
            (json!({"model_type": "gemma4_unified_text"}), "model_type"),
            (json!({"model_type": "gemma4"}), "model_type"),
            (json!({"use_bidirectional_attention": "all"}), "use_bidirectional_attention"),
            (json!({"use_bidirectional_attention": "unknown"}), "use_bidirectional_attention"),
            (json!({"hidden_size": 0}), "hidden_size"),
            (json!({"intermediate_size": 0}), "intermediate_size"),
            (json!({"num_hidden_layers": 0}), "num_hidden_layers"),
            (json!({"vocab_size": 0}), "vocab_size"),
            (json!({"num_attention_heads": 0}), "num_attention_heads"),
            (json!({"num_key_value_heads": 0}), "num_key_value_heads"),
            (json!({"num_key_value_heads": 3}), "num_key_value_heads"),
            (json!({"num_global_key_value_heads": 0}), "num_global_key_value_heads"),
            (json!({"num_global_key_value_heads": 3}), "num_global_key_value_heads"),
            (json!({"head_dim": 0}), "head_dim"),
            (json!({"head_dim": 7}), "head_dim"),
            (json!({"global_head_dim": 0}), "global_head_dim"),
            (json!({"global_head_dim": 15}), "global_head_dim"),
            (json!({"global_head_dim": 2}), "partial_rotary_factor"),
            (json!({"layer_types": ["full_attention"]}), "layer_types"),
            (json!({"layer_types": ["full_attention", "full_attention", "full_attention"]}), "layer_types"),
            (json!({"layer_types": ["sliding_attention", "unknown"]}), "layer_types[1]"),
            (json!({"sliding_window": 0}), "sliding_window"),
            (json!({"max_position_embeddings": 0}), "max_position_embeddings"),
            (json!({"query_pre_attn_scalar": 0}), "query_pre_attn_scalar"),
            (json!({"rms_norm_eps": 0.0}), "rms_norm_eps"),
            (json!({"final_logit_softcapping": -1.0}), "final_logit_softcapping"),
            (json!({"rope_theta": 0.0}), "rope_theta"),
        ] {
            let error = config(overrides).validate().unwrap_err().to_string();
            assert!(error.contains(field), "expected {field}: {error}");
        }
        for (layer, params, field) in [
            ("full_attention", json!({"rope_type": "linear"}), "rope_type"),
            ("sliding_attention", json!({"rope_type": "proportional"}), "rope_type"),
            ("full_attention", json!({"rope_theta": 0.0}), "rope_theta"),
            ("sliding_attention", json!({"rope_theta": -1.0}), "rope_theta"),
            ("full_attention", json!({"partial_rotary_factor": 0.0}), "partial_rotary_factor"),
            ("full_attention", json!({"partial_rotary_factor": 1.5}), "partial_rotary_factor"),
            ("full_attention", json!({"partial_rotary_factor": 0.2}), "partial_rotary_factor"),
            ("full_attention", json!({"rope_type": "default", "partial_rotary_factor": 0.25}), "partial_rotary_factor"),
            ("sliding_attention", json!({"partial_rotary_factor": 0.5}), "partial_rotary_factor"),
        ] {
            let error = config(json!({"rope_parameters": {(layer): params}}))
                .validate()
                .unwrap_err()
                .to_string();
            assert!(error.contains(field), "expected {field}: {error}");
        }
    }

    #[test]
    fn nonfinite_values_are_rejected_for_programmatic_configs() {
        let mut cfg = config(json!({}));
        cfg.rms_norm_eps = f64::NAN;
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("rms_norm_eps"));
        cfg.rms_norm_eps = 1e-6;
        cfg.final_logit_softcapping = Some(f64::INFINITY);
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("final_logit_softcapping"));
    }
}
