use std::collections::HashMap;

use candle::quantized::GgmlDType;
use candle::{DType, Device, Result, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::gemma4::{config::Gemma4TextConfig, text::TextModel};

fn config() -> Gemma4TextConfig {
    serde_json::from_value(serde_json::json!({
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "global_head_dim": 64,
        "vocab_size": 17,
        "sliding_window": 3,
        "max_position_embeddings": 12,
        "layer_types": ["sliding_attention", "full_attention"],
        "attention_bias": true,
        "tie_word_embeddings": false,
        "final_logit_softcapping": 0.5
    }))
    .unwrap()
}

fn weights(cfg: &Gemma4TextConfig, dtype: DType) -> Result<HashMap<String, Tensor>> {
    // Use a VarMap only to discover checkpoint shapes, then load independent,
    // deterministic frozen tensors through the tensor backend.
    let vars = VarMap::new();
    TextModel::new(
        cfg,
        VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
    )?;
    let data = vars.data().lock().unwrap();
    data.iter()
        .map(|(name, var)| {
            let seed = name
                .bytes()
                .fold(17u32, |s, b| s.wrapping_mul(31).wrapping_add(b as u32));
            let data = (0..var.elem_count())
                .map(|i| {
                    if name.contains("norm") || name.ends_with("layer_scalar") {
                        return 1f32;
                    }
                    let n = seed.wrapping_add((i as u32).wrapping_mul(747796405));
                    let n = (n ^ (n >> 16)).wrapping_mul(2246822519);
                    ((n ^ (n >> 13)) % 401) as f32 / 2000. - 0.1
                })
                .collect::<Vec<_>>();
            Ok((
                name.clone(),
                Tensor::from_vec(data, var.shape(), &Device::Cpu)?.to_dtype(dtype)?,
            ))
        })
        .collect()
}

fn load(
    cfg: &Gemma4TextConfig,
    tensors: HashMap<String, Tensor>,
    dtype: DType,
) -> Result<TextModel> {
    TextModel::new(cfg, VarBuilder::from_tensors(tensors, dtype, &Device::Cpu))
}

fn input() -> Result<Tensor> {
    Tensor::new(&[[1u32, 7, 3, 10], [4, 6, 5, 9]], &Device::Cpu)
}

fn assert_close(
    actual: &Tensor,
    expected: &Tensor,
    max_error: f32,
    relative_rms: f32,
) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut error = 0.;
    let mut signal = 0.;
    let mut maximum = 0f32;
    for (a, e) in actual.iter().zip(&expected) {
        assert!(a.is_finite() && e.is_finite());
        maximum = maximum.max((a - e).abs());
        error += (a - e).powi(2);
        signal += e.powi(2);
    }
    let relative = (error / signal.max(1e-20)).sqrt();
    assert!(maximum <= max_error && relative <= relative_rms,
        "max error {maximum} (limit {max_error}), relative RMS error {relative} (limit {relative_rms})");
    Ok(())
}

#[test]
fn frozen_q4_q8_features_and_logits_stay_finite_and_close_to_dense() -> Result<()> {
    let input = input()?;
    for dtype in [DType::F32, DType::BF16] {
        for shared_kv in [false, true] {
            let mut cfg = config();
            cfg.attention_k_eq_v = shared_kv;
            cfg.tie_word_embeddings = shared_kv;
            for quant in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
                let tensors = weights(&cfg, dtype)?;
                let head = tensors[if shared_kv {
                    "model.embed_tokens.weight"
                } else {
                    "lm_head.weight"
                }]
                .clone();
                let mut model = load(&cfg, tensors, dtype)?;
                let dense_hidden = model.forward_train_hidden(&input)?;
                let dense_raw = model.project_logits(&dense_hidden)?;
                let dense_logits = model.forward_train(&input)?;
                assert_close(&dense_logits, &((&dense_raw / 0.5)?.tanh()? * 0.5)?, 0., 0.)?;
                // Quantization must discard inference caches computed with dense weights.
                model.forward(&input.narrow(1, 0, 1)?, 0)?;
                assert_eq!(
                    model.quantize_frozen_decoder(quant)?,
                    if shared_kv { 13 } else { 14 }
                );
                let hidden = model.forward_train_hidden(&input)?;
                assert_eq!(hidden.dims(), &[2, 4, cfg.hidden_size]);
                assert_eq!(hidden.dtype(), dtype);
                assert!(!hidden.track_op());
                let raw = model.project_logits(&hidden)?;
                let expected_raw = hidden
                    .reshape((8, cfg.hidden_size))?
                    .matmul(&head.t()?)?
                    .reshape((2, 4, cfg.vocab_size))?;
                assert_close(&raw, &expected_raw, 0., 0.)?;
                let logits = model.forward_train(&input)?;
                assert_close(&logits, &((&raw / 0.5)?.tanh()? * 0.5)?, 0., 0.)?;
                let (max_error, relative) = if quant == GgmlDType::Q4_0 {
                    (0.4, 0.3)
                } else {
                    (0.1, 0.08)
                };
                assert_close(&raw, &dense_raw, max_error, relative)?;
                assert_close(&logits, &dense_logits, max_error, relative)?;
                let last = model.forward(&input, 0)?;
                assert_close(&last, &logits.narrow(1, 3, 1)?, 0.03, 0.04)?;
                assert_eq!(model.quantize_frozen_decoder(quant)?, 0);
                let other = if quant == GgmlDType::Q4_0 {
                    GgmlDType::Q8_0
                } else {
                    GgmlDType::Q4_0
                };
                assert!(model
                    .quantize_frozen_decoder(other)
                    .unwrap_err()
                    .to_string()
                    .contains("different dtype"));
            }
        }
    }
    Ok(())
}

#[test]
fn rejects_trainable_weights_including_gradient_dependencies_before_conversion() -> Result<()> {
    let cfg = config();
    let vars = VarMap::new();
    let mut model = TextModel::new(
        &cfg,
        VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
    )?;
    assert!(model
        .quantize_frozen_decoder(GgmlDType::Q4_0)
        .unwrap_err()
        .to_string()
        .contains("trainable"));
    for name in [
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.1.self_attn.o_proj.bias",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.layer_scalar",
        "lm_head.weight",
    ] {
        let mut tensors = weights(&cfg, DType::F32)?;
        let tracked = Var::from_tensor(&tensors[name])?.affine(1., 0.)?;
        assert!(!tracked.is_variable() && tracked.track_op());
        tensors.insert(name.to_owned(), tracked);
        let mut model = load(&cfg, tensors, DType::F32)?;
        let before = model.forward_train(&input()?)?;
        let error = model
            .quantize_frozen_decoder(GgmlDType::Q8_0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("trainable") && error.contains(name.trim_start_matches("model.")),
            "{error}"
        );
        assert_close(&model.forward_train(&input()?)?, &before, 0., 0.)?;
    }
    Ok(())
}

#[test]
fn rejects_unsupported_dtypes_and_unaligned_projection_dimensions() -> Result<()> {
    let cfg = config();
    let mut model = load(&cfg, weights(&cfg, DType::F32)?, DType::F32)?;
    for dtype in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::Q4_1,
        GgmlDType::Q8K,
    ] {
        assert!(model
            .quantize_frozen_decoder(dtype)
            .unwrap_err()
            .to_string()
            .contains("only Q4_0 or Q8_0"));
    }
    for (hidden, intermediate, head, name) in [
        (66, 96, 32, "self_attn.q_proj"),
        (64, 95, 32, "mlp.gate_proj"),
        (64, 96, 34, "self_attn.q_proj"),
    ] {
        let mut cfg = config();
        cfg.hidden_size = hidden;
        cfg.intermediate_size = intermediate;
        cfg.head_dim = head;
        for quant in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
            let mut model = load(&cfg, weights(&cfg, DType::F32)?, DType::F32)?;
            let before = model.forward_train(&input()?)?;
            let error = model
                .quantize_frozen_decoder(quant)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(name) && error.contains("block size 32"),
                "{error}"
            );
            assert_close(&model.forward_train(&input()?)?, &before, 0., 0.)?;
        }
    }
    let mut model = load(&cfg, weights(&cfg, DType::F16)?, DType::F16)?;
    assert!(model
        .quantize_frozen_decoder(GgmlDType::Q4_0)
        .unwrap_err()
        .to_string()
        .contains("must be F32 or BF16"));
    Ok(())
}
