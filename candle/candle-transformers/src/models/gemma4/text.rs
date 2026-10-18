//! Gemma 4 text decoder.
//!
//! Experimental dense text-only decoder. PLE, shared-KV consumer layers and MoE
//! are rejected by configuration validation rather than approximated as Gemma 3.

use std::sync::Arc;

use candle::quantized::{GgmlDType, QMatMul, QTensor};
use candle::{CustomOp1, DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear_bias, Activation, Linear, VarBuilder};

use super::config::Gemma4TextConfig;

#[derive(Debug, Clone)]
enum DecoderLinear {
    Dense(Linear),
    Quantized {
        weight: QMatMul,
        bias: Option<Tensor>,
        lora: Option<QuantizedLora>,
    },
}

#[derive(Debug, Clone)]
struct QuantizedLora {
    a: Tensor,
    b: Tensor,
    scale: f64,
}

// Retain only the packed weight in the graph, never a dense dequantization.
struct FrozenQMatMul(Arc<QTensor>);

impl CustomOp1 for FrozenQMatMul {
    fn name(&self) -> &'static str {
        "gemma4-frozen-qmatmul"
    }

    fn cpu_fwd(
        &self,
        s: &candle::CpuStorage,
        l: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        self.0.cpu_fwd(s, l)
    }

    fn cuda_fwd(
        &self,
        s: &candle::CudaStorage,
        l: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        self.0.cuda_fwd(s, l)
    }

    fn metal_fwd(
        &self,
        s: &candle::MetalStorage,
        l: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        self.0.metal_fwd(s, l)
    }

    fn bwd(&self, arg: &Tensor, _: &Tensor, grad: &Tensor) -> Result<Option<Tensor>> {
        let (out_dim, _) = self.0.shape().dims2()?;
        // First order only: release this dense weight after dX = dY W, even when
        // CANDLE_GRAD_DO_NOT_DETACH is set by the caller.
        let weight = self.0.dequantize(arg.device())?;
        let dx = grad
            .detach()
            .to_dtype(DType::F32)?
            .contiguous()?
            .reshape(((), out_dim))?
            .matmul(&weight)?
            .reshape(arg.shape())?
            .to_dtype(arg.dtype())?;
        Ok(Some(dx.detach()))
    }
}

impl DecoderLinear {
    fn new(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(linear_bias(in_dim, out_dim, bias, vb)?))
    }

    fn quantize(&mut self, dtype: GgmlDType) -> Result<usize> {
        let Self::Dense(linear) = self else {
            return Ok(0);
        };
        // QTensor::quantize converts to F32 internally. Replace each projection
        // immediately so its dense storage is released before quantizing the next.
        let weight = QTensor::quantize(linear.weight(), dtype)?;
        *self = Self::Quantized {
            // Do not honor CANDLE_DEQUANTIZE_ALL / CANDLE_DEQUANTIZE_ALL_F16.
            weight: QMatMul::QTensor(Arc::new(weight)),
            bias: linear.bias().cloned(),
            lora: None,
        };
        Ok(1)
    }
}

impl Module for DecoderLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(xs),
            Self::Quantized { weight, bias, lora } => {
                let input = xs.to_dtype(DType::F32)?.contiguous()?;
                let ys = match (weight, lora) {
                    (QMatMul::QTensor(weight), Some(_)) => {
                        input.apply_op1(FrozenQMatMul(weight.clone()))?
                    }
                    _ => weight.forward(&input)?,
                }
                .to_dtype(xs.dtype())?;
                let ys = match lora {
                    None => ys,
                    Some(lora) => {
                        // Recast masters on every forward so optimizer updates are visible.
                        let a = Linear::new(lora.a.to_dtype(xs.dtype())?, None);
                        let b = Linear::new(lora.b.to_dtype(xs.dtype())?, None);
                        (ys + (xs.apply(&a)?.apply(&b)? * lora.scale)?)?
                    }
                };
                match bias {
                    None => Ok(ys),
                    Some(bias) => ys.broadcast_add(bias),
                }
            }
        }
    }
}

fn validate_frozen_weight(name: &str, weight: &Tensor) -> Result<()> {
    if weight.track_op() {
        candle::bail!("Gemma4 frozen decoder: {name} is trainable; load frozen weights, not a VarMap or tensors with gradient dependencies");
    }
    if !matches!(weight.dtype(), DType::F32 | DType::BF16) {
        candle::bail!(
            "Gemma4 frozen decoder: {name} must be F32 or BF16, got {:?}",
            weight.dtype()
        );
    }
    Ok(())
}

fn validate_quantized_dtype(dtype: GgmlDType) -> Result<()> {
    if !matches!(dtype, GgmlDType::Q4_0 | GgmlDType::Q8_0) {
        candle::bail!("Gemma4 frozen decoder supports only Q4_0 or Q8_0, got {dtype:?}");
    }
    Ok(())
}

// RMSNorm uses the checkpoint weight directly, unlike Gemma 3's +1 offset.

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", candle_nn::Init::Const(1.))?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .broadcast_mul(&self.weight.to_dtype(internal_dtype)?)?
            .to_dtype(x_dtype)
    }
}

/// Pure RMS normalization without learned weight (used for V norm).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
}

// ── RotaryEmbedding (standard, for sliding layers) ──────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let rope = if train {
            candle_nn::rotary_emb::rope_slow
        } else {
            candle_nn::rotary_emb::rope
        };
        let q_embed = rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── ProportionalRotaryEmbedding (for global/full layers) ────────────────────

#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl ProportionalRotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32));
        }
        // Pad with zeros for non-rotated dimensions -> cos=1, sin=0 -> identity
        inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let rope = if train {
            candle_nn::rotary_emb::rope_slow
        } else {
            candle_nn::rotary_emb::rope
        };
        let q_embed = rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: DecoderLinear,
    up_proj: DecoderLinear,
    down_proj: DecoderLinear,
    act_fn: Activation,
}

impl MLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj =
            DecoderLinear::new(hidden_size, intermediate_size, bias, vb.pp("gate_proj"))?;
        let up_proj = DecoderLinear::new(hidden_size, intermediate_size, bias, vb.pp("up_proj"))?;
        let down_proj =
            DecoderLinear::new(intermediate_size, hidden_size, bias, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ── Flash attention ─────────────────────────────────────────────────────────

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    candle::bail!("compile with '--features flash-attn' to use FlashAttention")
}

// ── KvCache ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

// ── Attention ───────────────────────────────────────────────────────────────

fn attention_query_chunk_len(batch: usize, heads: usize, kv_len: usize) -> usize {
    // At most 256 MiB per FP32 score tensor, below CUDA's u32 launch counts
    // and the fused softmax kernel's signed 32-bit element offsets.
    ((1usize << 26) / batch / heads / kv_len).clamp(1, 1024)
}

fn eager_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    train: bool,
) -> Result<Tensor> {
    let (batch, heads, q_len, dim) = q.dims4()?;
    if !train {
        let chunk_len = attention_query_chunk_len(batch, heads, k.dim(2)?);
        if q_len > chunk_len {
            let mut outputs = Vec::new();
            for start in (0..q_len).step_by(chunk_len) {
                let len = chunk_len.min(q_len - start);
                let q = q.narrow(2, start, len)?.contiguous()?;
                // Slice only queries: every chunk still sees all keys and the
                // original absolute-position causal/sliding mask.
                let mask = mask.map(|m| m.narrow(2, start, len)).transpose()?;
                outputs.push(eager_attention(&q, k, v, mask.as_ref(), false)?);
            }
            return Tensor::cat(&outputs, 2);
        }
    }

    // Fold query-head groups into rows instead of materializing repeated K/V.
    // Restore the query-head layout before applying the original mask.
    let (_, kv_heads, kv_len, _) = k.dims4()?;
    let grouped_rows = (heads / kv_heads) * q_len;
    let q_grouped = q.reshape((batch, kv_heads, grouped_rows, dim))?;
    // Gemma 4 uses unit QK scaling, with softmax evaluated in FP32.
    let attn_weights = q_grouped
        .matmul(&k.transpose(2, 3)?)?
        .reshape((batch, heads, q_len, kv_len))?;
    let attn_weights = match mask {
        None => attn_weights,
        Some(mask) => attn_weights.broadcast_add(mask)?,
    };
    let attn_weights = attn_weights.to_dtype(DType::F32)?;
    let attn_weights = if train {
        candle_nn::ops::softmax(&attn_weights, D::Minus1)?
    } else {
        candle_nn::ops::softmax_last_dim(&attn_weights)?
    }
    .to_dtype(q.dtype())?;
    attn_weights
        .reshape((batch, kv_heads, grouped_rows, kv_len))?
        .matmul(v)?
        .reshape((batch, heads, q_len, dim))
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: DecoderLinear,
    k_proj: DecoderLinear,
    v_proj: Option<DecoderLinear>,
    o_proj: DecoderLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    is_sliding: bool,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    use_flash_attn: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let bias = cfg.attention_bias;
        let is_sliding = cfg.is_sliding(layer_idx);

        let (head_dim, num_kv_heads) = if is_sliding {
            (cfg.head_dim, cfg.num_key_value_heads)
        } else {
            let global_kv = if cfg.attention_k_eq_v {
                cfg.num_global_key_value_heads
                    .unwrap_or(cfg.num_key_value_heads)
            } else {
                cfg.num_key_value_heads
            };
            (cfg.global_head_dim, global_kv)
        };

        let num_kv_groups = num_heads / num_kv_heads;
        let q_proj = DecoderLinear::new(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = DecoderLinear::new(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = if cfg.attention_k_eq_v && !is_sliding {
            None
        } else {
            Some(DecoderLinear::new(
                hidden_sz,
                num_kv_heads * head_dim,
                bias,
                vb.pp("v_proj"),
            )?)
        };
        let o_proj = DecoderLinear::new(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let kv_cache = if is_sliding {
            KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(
                2,
                cfg.effective_sliding_window(),
            ))
        } else {
            KvCache::Normal(candle_nn::kv_cache::KvCache::new(
                2,
                cfg.max_position_embeddings,
            ))
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            is_sliding,
            rotary_emb_global,
            rotary_emb_local,
            kv_cache,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        train: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        // Global K=V shares the raw projection, not the normalized/rotated keys.
        let v = match &self.v_proj {
            Some(proj) => proj.forward(xs)?,
            None => k.clone(),
        };

        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Q/K norms
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;
        // V norm (RMS without learned weight)
        let v = v_norm(&v, self.rms_norm_eps)?;

        // Apply RoPE
        let (q, k) = if self.is_sliding {
            self.rotary_emb_local
                .apply_rotary_emb_qkv(&q, &k, seqlen_offset, train)?
        } else {
            self.rotary_emb_global
                .apply_rotary_emb_qkv(&q, &k, seqlen_offset, train)?
        };

        let (k, v) = if train {
            (k, v)
        } else {
            match &mut self.kv_cache {
                KvCache::Normal(cache) => cache.append(&k, &v)?,
                KvCache::Rotating(cache) if q_len > 1 && seqlen_offset > 0 => {
                    // Preserve the earliest query's history before the ring append
                    // overwrites it. Attention sees chronological history + this chunk.
                    let with_history = |cache: &candle_nn::kv_cache::RotatingCache,
                                        src: &Tensor|
                     -> Result<Tensor> {
                        let history = cache.current_seq_len().min(cache.max_seq_len() - 1);
                        if history == 0 {
                            return Ok(src.clone());
                        }
                        let data = cache.current_data()?.unwrap();
                        let len = data.dim(2)?;
                        let start = (cache.offset() + len - history) % len;
                        let tail = history.min(len - start);
                        let mut parts = vec![data.narrow(2, start, tail)?];
                        if tail < history {
                            parts.push(data.narrow(2, 0, history - tail)?);
                        }
                        parts.push(src.clone());
                        // cat materializes independent storage, unlike narrow/clone.
                        Tensor::cat(&parts, 2)
                    };
                    let keys = with_history(cache.k_cache(), &k)?;
                    let values = with_history(cache.v_cache(), &v)?;
                    cache.append(&k, &v)?;
                    (keys, values)
                }
                KvCache::Rotating(cache) => cache.append(&k, &v)?,
            }
        };

        let mask = if self.is_sliding {
            sliding_attention_mask
        } else {
            attention_mask
        };

        // The basic flash kernel cannot express the local mask or 512-wide heads.
        // Keep cached chunks eager rather than assuming unequal-length causal alignment.
        let use_flash_attn = !train
            && self.use_flash_attn
            && !self.is_sliding
            && self.head_dim <= 256
            && (seqlen_offset == 0 || q_len == 1);
        let (k, v) = if train || use_flash_attn {
            (
                crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?,
                crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?,
            )
        } else {
            (k.contiguous()?, v.contiguous()?)
        };
        let attn_output = if use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            flash_attn(&q, &k, &v, 1.0, mask.is_some())?.transpose(1, 2)?
        } else {
            eager_attention(&q, &k, &v, mask, train)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    layer_scalar: Tensor,
}

impl DecoderLayer {
    fn validate_frozen(&mut self, index: usize, dtype: GgmlDType) -> Result<()> {
        for (name, weight) in [
            ("input_layernorm.weight", &self.input_layernorm.weight),
            (
                "post_attention_layernorm.weight",
                &self.post_attention_layernorm.weight,
            ),
            (
                "pre_feedforward_layernorm.weight",
                &self.pre_feedforward_layernorm.weight,
            ),
            (
                "post_feedforward_layernorm.weight",
                &self.post_feedforward_layernorm.weight,
            ),
            ("self_attn.q_norm.weight", &self.self_attn.q_norm.weight),
            ("self_attn.k_norm.weight", &self.self_attn.k_norm.weight),
            ("layer_scalar", &self.layer_scalar),
        ] {
            validate_frozen_weight(&format!("layers.{index}.{name}"), weight)?;
        }
        let block_size = dtype.block_size();
        for (name, proj) in self.projections_mut() {
            let name = format!("layers.{index}.{name}");
            match proj {
                DecoderLinear::Dense(linear) => {
                    validate_frozen_weight(&format!("{name}.weight"), linear.weight())?;
                    if let Some(bias) = linear.bias() {
                        validate_frozen_weight(&format!("{name}.bias"), bias)?;
                    }
                    let (out_dim, in_dim) = linear.weight().dims2()?;
                    if out_dim == 0
                        || in_dim == 0
                        || !out_dim.is_multiple_of(block_size)
                        || !in_dim.is_multiple_of(block_size)
                    {
                        candle::bail!("Gemma4 frozen decoder: {name}.weight shape [{out_dim}, {in_dim}] must have both dimensions divisible by {dtype:?} block size {block_size}");
                    }
                }
                DecoderLinear::Quantized {
                    weight: QMatMul::QTensor(weight),
                    bias,
                    ..
                } if weight.dtype() == dtype => {
                    if let Some(bias) = bias {
                        validate_frozen_weight(&format!("{name}.bias"), bias)?;
                    }
                }
                DecoderLinear::Quantized { .. } => {
                    candle::bail!(
                        "Gemma4 frozen decoder: {name} is already quantized with a different dtype"
                    );
                }
            }
        }
        Ok(())
    }

    fn projections_mut(&mut self) -> impl Iterator<Item = (&'static str, &mut DecoderLinear)> {
        [
            ("self_attn.q_proj", Some(&mut self.self_attn.q_proj)),
            ("self_attn.k_proj", Some(&mut self.self_attn.k_proj)),
            ("self_attn.v_proj", self.self_attn.v_proj.as_mut()),
            ("self_attn.o_proj", Some(&mut self.self_attn.o_proj)),
            ("mlp.gate_proj", Some(&mut self.mlp.gate_proj)),
            ("mlp.up_proj", Some(&mut self.mlp.up_proj)),
            ("mlp.down_proj", Some(&mut self.mlp.down_proj)),
        ]
        .into_iter()
        .filter_map(|(name, proj)| proj.map(|proj| (name, proj)))
    }

    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb_global,
            rotary_emb_local,
            cfg,
            layer_idx,
            vb.pp("self_attn"),
        )?;
        let mlp = MLP::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_activation,
            false,
            vb.pp("mlp"),
        )?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar: vb.get_with_hints(1, "layer_scalar", candle_nn::Init::Const(1.))?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        train: bool,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            sliding_attention_mask,
            seqlen_offset,
            train,
        )?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        (residual + xs)?.broadcast_mul(&self.layer_scalar)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

// ── Causal mask ─────────────────────────────────────────────────────────────

fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let history = sliding_window.map_or(seqlen_offset, |w| seqlen_offset.min(w - 1));
    let kv_len = history + tgt_len;
    let mask = (0..tgt_len).flat_map(|i| {
        (0..kv_len).map(move |j| {
            let query = history + i;
            if j > query || sliding_window.is_some_and(|w| query - j >= w) {
                f32::NEG_INFINITY
            } else {
                0f32
            }
        })
    });
    // Build BF16 directly: a 256 x 262144 mask otherwise stages 256 MiB of F32
    // on the host alongside its 128 MiB BF16 conversion. Both values are exact.
    let storage = match dtype {
        DType::BF16 => candle::CpuStorage::BF16(
            mask.map(|v| candle::WithDType::from_f64(v as f64))
                .collect(),
        ),
        _ => candle::CpuStorage::F32(mask.collect()),
    };
    Tensor::from_storage(
        candle::Storage::Cpu(storage),
        (tgt_len, kv_len),
        candle::op::BackpropOp::none(),
        false,
    )
    .to_dtype(dtype)?
    .to_device(device)?
    .expand((b_size, 1, tgt_len, kv_len))
}

// ── TextModel ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
    max_position_embeddings: usize,
    cache_seq_len: usize,
    cache_batch_size: Option<usize>,
}

impl TextModel {
    /// Quantize only frozen attention and MLP projections to Q4_0 or Q8_0.
    /// All parameters must be frozen F32/BF16 tensors, not VarMap weights.
    /// Embeddings, norms, scalars, biases and the output head stay dense.
    /// Returns the number of newly quantized projections (zero on a repeated call
    /// with the same dtype). Clears the inference cache before conversion.
    /// Validation precedes conversion; a backend/allocation error can leave some
    /// projections quantized. Dense storage is released one projection at a time,
    /// unless another model clone or the loading backend still owns that storage.
    /// Quantized decoder input gradients are enabled by `add_quantized_lora`.
    pub fn quantize_frozen_decoder(&mut self, dtype: GgmlDType) -> Result<usize> {
        self.validate_frozen_decoder(dtype)?;
        self.clear_kv_cache();
        let mut count = 0;
        for layer in &mut self.layers {
            for (_, proj) in layer.projections_mut() {
                count += proj.quantize(dtype)?;
            }
        }
        Ok(count)
    }

    fn validate_frozen_decoder(&mut self, dtype: GgmlDType) -> Result<()> {
        validate_quantized_dtype(dtype)?;
        for (name, weight) in [
            ("embed_tokens.weight", self.embed_tokens.embeddings()),
            ("norm.weight", &self.norm.weight),
            ("lm_head.weight", self.lm_head.weight()),
        ] {
            validate_frozen_weight(name, weight)?;
        }
        for (index, layer) in self.layers.iter_mut().enumerate() {
            layer.validate_frozen(index, dtype)?;
        }
        Ok(())
    }

    /// Attach LoRA to every already-quantized attention/MLP projection.
    /// The base must be entirely frozen and quantized before calling this method.
    /// F32 masters are named `layers.{index}.{projection}.lora_a/lora_b` under `vb`;
    /// A has shape `[rank, input]`, B `[output, rank]`, and scaling is alpha/rank.
    /// Newly initialized A is N(0, 0.02), B is zero. Existing builder values load
    /// unchanged. Use a separate VarMap for training; frozen adapter loading is allowed.
    /// Returns the number attached. Repeated attachment is rejected. Input gradients
    /// use F32 dY W with a transient dense weight in backward (first order only).
    /// This is a straight-through approximation when packed kernels quantize activations.
    /// A builder error can leave some adapters attached; discard the model on error.
    pub fn add_quantized_lora(&mut self, rank: usize, alpha: f64, vb: VarBuilder) -> Result<usize> {
        if rank == 0 || !alpha.is_finite() || alpha <= 0. {
            candle::bail!("Gemma4 LoRA requires rank > 0 and finite alpha > 0");
        }
        if !self.device.same_device(vb.device()) {
            candle::bail!("Gemma4 LoRA builder must use the model device");
        }
        let mut dtype = None;
        for layer in &mut self.layers {
            for (_, proj) in layer.projections_mut() {
                match proj {
                    DecoderLinear::Quantized { weight: QMatMul::QTensor(weight), lora: None, .. } => {
                        dtype = Some(weight.dtype());
                    }
                    _ => candle::bail!("Gemma4 LoRA requires already quantized projections without existing adapters"),
                }
            }
        }
        self.validate_frozen_decoder(dtype.unwrap_or(GgmlDType::Q4_0))?;
        let vb = vb.set_dtype(DType::F32);
        self.clear_kv_cache();
        let mut count = 0;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            for (name, proj) in layer.projections_mut() {
                if let DecoderLinear::Quantized {
                    weight: QMatMul::QTensor(weight),
                    lora,
                    ..
                } = proj
                {
                    let (out_dim, in_dim) = weight.shape().dims2()?;
                    let vb = vb.pp("layers").pp(index).pp(name);
                    let a = vb.get_with_hints(
                        (rank, in_dim),
                        "lora_a",
                        candle_nn::Init::Randn {
                            mean: 0.,
                            stdev: 0.02,
                        },
                    )?;
                    let b =
                        vb.get_with_hints((out_dim, rank), "lora_b", candle_nn::Init::Const(0.))?;
                    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
                        candle::bail!("Gemma4 LoRA masters must be F32");
                    }
                    *lora = Some(QuantizedLora {
                        a,
                        b,
                        scale: alpha / rank as f64,
                    });
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Load a standalone causal-LM checkpoint with its backbone at `model.*`.
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_backbone(cfg, vb.pp("model"), vb.pp("lm_head"), None)
    }

    /// Load only the text decoder from a conditional-generation checkpoint.
    /// Pass the checkpoint root; no vision/audio weights are allocated.
    pub fn new_for_conditional_generation(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_backbone(cfg, vb.pp("model.language_model"), vb.pp("lm_head"), None)
    }

    /// Load frozen weights, quantizing each decoder layer before loading the next.
    /// `dtype` is Q4_0 or Q8_0; the builder's F32/BF16 dtype is the compute dtype.
    /// The loading backend must not retain dense layers if bounded memory is needed.
    pub fn new_quantized(cfg: &Gemma4TextConfig, vb: VarBuilder, dtype: GgmlDType) -> Result<Self> {
        Self::new_with_backbone(cfg, vb.pp("model"), vb.pp("lm_head"), Some(dtype))
    }

    /// Stream only the frozen text decoder from a conditional-generation checkpoint.
    pub fn new_quantized_for_conditional_generation(
        cfg: &Gemma4TextConfig,
        vb: VarBuilder,
        dtype: GgmlDType,
    ) -> Result<Self> {
        Self::new_with_backbone(
            cfg,
            vb.pp("model.language_model"),
            vb.pp("lm_head"),
            Some(dtype),
        )
    }

    fn new_with_backbone(
        cfg: &Gemma4TextConfig,
        vb_m: VarBuilder,
        vb_head: VarBuilder,
        quantized_dtype: Option<GgmlDType>,
    ) -> Result<Self> {
        cfg.validate()?;
        if let Some(dtype) = quantized_dtype {
            validate_quantized_dtype(dtype)?;
        }
        if cfg.use_flash_attn && !cfg!(feature = "flash-attn") {
            candle::bail!("compile with '--features flash-attn' to use FlashAttention");
        }
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        if quantized_dtype.is_some() {
            validate_frozen_weight("embed_tokens.weight", embed_tokens.embeddings())?;
        }

        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb_m.dtype(),
            cfg.global_head_dim,
            cfg.rope_global_base_freq(),
            cfg.partial_rotary_factor(),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);
        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            vb_m.dtype(),
            cfg.head_dim,
            cfg.rope_local_base_freq(),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let mut layer = DecoderLayer::new(
                rotary_emb_global.clone(),
                rotary_emb_local.clone(),
                cfg,
                layer_idx,
                vb_l.pp(layer_idx),
            )?;
            if let Some(dtype) = quantized_dtype {
                layer.validate_frozen(layer_idx, dtype)?;
                for (_, proj) in layer.projections_mut() {
                    proj.quantize(dtype)?;
                }
            }
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_head)?
        };
        if quantized_dtype.is_some() {
            validate_frozen_weight("norm.weight", &norm.weight)?;
            validate_frozen_weight("lm_head.weight", lm_head.weight())?;
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb_m.device().clone(),
            dtype: vb_m.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
            max_position_embeddings: cfg.max_position_embeddings,
            cache_seq_len: 0,
            cache_batch_size: None,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }
        let mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;
        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;
        Ok((Some(mask), Some(sliding_mask)))
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        let scale =
            Tensor::new((self.hidden_size as f32).sqrt(), &self.device)?.to_dtype(xs.dtype())?;
        xs.broadcast_mul(&scale)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let xs = self.embed_tokens(input_ids)?;
        self.forward_embeds(&xs, seqlen_offset, b_size, seq_len)
    }

    /// Full-sequence logits of shape `[batch, sequence, vocab]`, with softcapping.
    /// Positions start at zero; the inference KV cache is neither read nor changed.
    /// Uses unfused RoPE and softmax, never FlashAttention. Build with a `VarMap`
    /// for trainable weights (including `layer_scalar`); labels/loss are caller-owned.
    /// Clear the inference cache after updating weights before resuming generation.
    /// Decoder gradients require dense projections or `add_quantized_lora`.
    pub fn forward_train(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.forward_train_hidden(input_ids)?;
        let logits = self.project_logits(&hidden)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => (logits / sc)?.tanh()? * sc,
        }
    }

    /// Normalized full-sequence hidden states `[batch, sequence, hidden_size]`.
    /// Positions start at zero and the inference KV cache is not read or changed.
    /// Uses unfused RoPE/softmax. With a quantized frozen decoder these are features
    /// for output-head-only training unless `add_quantized_lora` has been called.
    pub fn forward_train_hidden(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        if batch_size == 0 || seq_len == 0 {
            candle::bail!("Gemma4 expects nonempty input_ids of shape [batch, sequence]");
        }
        if seq_len > self.max_position_embeddings {
            candle::bail!("Gemma4 sequence exceeds max_position_embeddings");
        }
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, 0)?;
        let mut xs = self.embed_tokens(input_ids)?;
        for layer in &mut self.layers {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                0,
                true,
            )?;
        }
        xs.apply(&self.norm)
    }

    /// Apply the dense output head to normalized hidden states, without softcapping.
    /// Add any output-head LoRA logits before applying the configured softcap.
    pub fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        hidden.apply(&self.lm_head)
    }

    pub fn forward_embeds(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        if xs.dims() != [batch_size, seq_len, self.hidden_size] || batch_size == 0 || seq_len == 0 {
            candle::bail!(
                "Gemma4 expects nonempty embeddings of shape [batch, sequence, hidden_size]"
            );
        }
        if seqlen_offset != self.cache_seq_len {
            candle::bail!("Gemma4 seqlen_offset must equal cached length {}; clear_kv_cache before a new prompt", self.cache_seq_len);
        }
        if self.cache_batch_size.is_some_and(|b| b != batch_size) {
            candle::bail!("Gemma4 batch size must remain unchanged while using the KV cache");
        }
        if seq_len > self.max_position_embeddings.saturating_sub(seqlen_offset) {
            candle::bail!("Gemma4 sequence exceeds max_position_embeddings");
        }
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;

        let mut xs = xs.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
                false,
            )?
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
        let logits = match self.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };
        self.cache_seq_len += seq_len;
        self.cache_batch_size = Some(batch_size);
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
        self.cache_seq_len = 0;
        self.cache_batch_size = None;
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;

    fn check_cached_chunk_masks(device: &Device) -> Result<()> {
        for dtype in [DType::F32, DType::BF16, DType::F16, DType::F64] {
            for window in [None, Some(1), Some(3)] {
                for offset in [0usize, 2, 3, 262142] {
                    let mask = prepare_decoder_attention_mask(2, 2, offset, window, dtype, device)?;
                    let first_key = window.map_or(0, |w| offset.saturating_sub(w - 1));
                    let kv_len = offset + 2 - first_key;
                    assert_eq!(mask.dims(), &[2, 1, 2, kv_len]);
                    assert_eq!(mask.dtype(), dtype);
                    assert!(mask.device().same_device(device));
                    let mask = mask.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                    for batch in 0..2 {
                        let rows = mask.get(batch)?.get(0)?.to_vec2::<f32>()?;
                        for (i, row) in rows.iter().enumerate() {
                            for (j, &value) in row.iter().enumerate() {
                                let query = offset + i;
                                let key = first_key + j;
                                let visible =
                                    key <= query && window.is_none_or(|w| query - key < w);
                                assert_eq!(value, if visible { 0. } else { f32::NEG_INFINITY });
                            }
                        }
                    }
                }
            }
        }
        let global = prepare_decoder_attention_mask(1, 2, 3, None, DType::F32, &Device::Cpu)?;
        assert_eq!(
            global.get(0)?.get(0)?.to_vec2::<f32>()?,
            vec![
                vec![0., 0., 0., 0., f32::NEG_INFINITY],
                vec![0., 0., 0., 0., 0.],
            ]
        );
        Ok(())
    }

    #[test]
    fn cached_chunk_masks_use_retained_history_and_absolute_positions() -> Result<()> {
        check_cached_chunk_masks(&Device::Cpu)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cached_chunk_masks_cuda() -> Result<()> {
        check_cached_chunk_masks(&Device::new_cuda(0)?)
    }

    #[test]
    fn long_context_score_chunks_fit_cuda_indices() {
        // The retained 31B prefill would launch a mask add with 2^33 as u32 = 0.
        assert_eq!((32u64 * 16384 * 16384) as u32, 0);
        for batch in [1, 2] {
            for kv_len in [8192, 16384, 262144] {
                let chunk = attention_query_chunk_len(batch, 32, kv_len);
                assert!(chunk > 0 && chunk < kv_len);
                assert!(batch * 32 * chunk * kv_len <= 1 << 26);
            }
        }
    }

    fn chunked_attention_matches_full(device: &Device) -> Result<()> {
        // Multiple batches/heads, a partial final chunk, and nonzero mask offsets.
        let (batch, heads, q_len, kv_len, dim) = (2, 4, 1025, 7, 4);
        let make = |heads, len| -> Result<Tensor> {
            Tensor::from_vec(
                (0..batch * heads * len * dim)
                    .map(|i| ((i * 17 % 101) as f32 - 50.) / 100.)
                    .collect::<Vec<_>>(),
                (batch, heads, len, dim),
                device,
            )
        };
        for dtype in [DType::F32, DType::BF16] {
            let q = make(heads, q_len)?.to_dtype(dtype)?;
            for kv_heads in [1, 2, heads] {
                let k = make(kv_heads, kv_len)?.to_dtype(dtype)?;
                let v = (&k * 0.7)?;
                for window in [None, Some(kv_len), Some(3)] {
                    let mask = window
                        .map(|window| {
                            let values = (0..q_len)
                                .flat_map(|i| {
                                    (0..kv_len).map(move |j| {
                                        let pos = i % kv_len;
                                        if j > pos || pos - j >= window {
                                            f32::NEG_INFINITY
                                        } else {
                                            0.
                                        }
                                    })
                                })
                                .collect::<Vec<_>>();
                            Tensor::from_vec(values, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)
                        })
                        .transpose()?;
                    let repeated_k =
                        crate::utils::repeat_kv(k.clone(), heads / kv_heads)?.contiguous()?;
                    let repeated_v =
                        crate::utils::repeat_kv(v.clone(), heads / kv_heads)?.contiguous()?;
                    let expected =
                        eager_attention(&q, &repeated_k, &repeated_v, mask.as_ref(), true)?;
                    let actual = eager_attention(&q, &k, &v, mask.as_ref(), false)?;
                    assert_eq!(actual.dims(), expected.dims());
                    let error = (actual - expected)?
                        .to_dtype(DType::F32)?
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?;
                    let tolerance = if dtype == DType::F32 { 1e-5 } else { 0.01 };
                    assert!(error < tolerance, "{dtype:?} {window:?}: {error}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn chunked_attention_cpu() -> Result<()> {
        chunked_attention_matches_full(&Device::Cpu)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn chunked_attention_cuda() -> Result<()> {
        chunked_attention_matches_full(&Device::new_cuda(0)?)
    }
}

#[cfg(test)]
mod lora_tests {
    use super::*;
    use candle::Var;

    #[test]
    fn frozen_qmatmul_exact_input_gradient_and_detached_backward() -> Result<()> {
        let dev = &Device::Cpu;
        let w = Tensor::from_vec(
            (0..96 * 64)
                .map(|i| ((i * 17 % 101) as f32 - 50.) / 100.)
                .collect::<Vec<_>>(),
            (96, 64),
            dev,
        )?;
        let w = Arc::new(QTensor::quantize(&w, GgmlDType::Q4_0)?);
        let op = FrozenQMatMul(w.clone());
        for dtype in [DType::F32, DType::BF16] {
            let x = Var::from_tensor(&Tensor::ones((2, 3, 64), dtype, dev)?)?;
            let input = x.to_dtype(DType::F32)?;
            let output = input.apply_op1(FrozenQMatMul(w.clone()))?;
            let packed = QMatMul::QTensor(w.clone()).forward(&input)?;
            assert_eq!(
                (&output - packed)?.abs()?.max_all()?.to_scalar::<f32>()?,
                0.
            );
            let upstream = Var::from_tensor(&Tensor::from_vec(
                (0..2 * 3 * 96)
                    .map(|i| ((i * 13 % 67) as f32 - 33.) / 50.)
                    .collect::<Vec<_>>(),
                (2, 3, 96),
                dev,
            )?)?;
            let expected = upstream
                .detach()
                .reshape((6, 96))?
                .matmul(&w.dequantize(dev)?)?
                .reshape((2, 3, 64))?;
            let direct = op.bwd(&input, &output, &upstream)?.unwrap();
            assert!(!direct.track_op());
            assert_eq!(direct.dtype(), DType::F32);
            assert_eq!(
                (&direct - &expected)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?,
                0.
            );
            let loss = (output.to_dtype(dtype)?.to_dtype(DType::F32)? * upstream.as_tensor())?
                .sum_all()?;
            let grads = loss.backward()?;
            let actual = grads.get(&x).unwrap();
            // The cast around the Q4 op rounds the upstream gradient first.
            let expected = upstream
                .detach()
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?
                .reshape((6, 96))?
                .matmul(&w.dequantize(dev)?)?
                .reshape((2, 3, 64))?
                .to_dtype(dtype)?;
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                (actual - expected)?
                    .to_dtype(DType::F32)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?,
                0.
            );
        }
        Ok(())
    }
}
