use std::collections::HashMap;

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma4::{config::Gemma4TextConfig, text::TextModel};

const PREFIXES: [&str; 2] = ["model", "model.language_model"];
const TOKENS: [u32; 9] = [1, 7, 3, 10, 2, 8, 4, 6, 5];

fn config() -> Gemma4TextConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "gemma4_text",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 4,
        "global_head_dim": 8,
        "attention_k_eq_v": true,
        "vocab_size": 16,
        "sliding_window": 3,
        "max_position_embeddings": 12,
        "layer_types": ["sliding_attention", "full_attention"],
        "final_logit_softcapping": 0.7
    }))
    .unwrap()
}

fn weights(cfg: &Gemma4TextConfig, prefix: &str) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();
    let mut insert = |name: &str, shape: &[usize]| -> Result<()> {
        // Seed by the relative name so both checkpoint layouts have identical weights.
        let seed = name
            .bytes()
            .fold(17u32, |s, b| s.wrapping_mul(31).wrapping_add(u32::from(b)));
        let data: Vec<f32> = (0..shape.iter().product())
            .map(|i: usize| {
                let n = seed.wrapping_add((i as u32).wrapping_mul(747796405));
                let n = (n ^ (n >> 16)).wrapping_mul(2246822519);
                let value = ((n ^ (n >> 13)) % 200 + 1) as f32;
                if name.contains("norm") {
                    0.6 + value / 250.
                } else {
                    let sign = if n & 256 == 0 { -1. } else { 1. };
                    sign * value / 1000.
                }
            })
            .collect();
        let key = if name == "lm_head.weight" {
            name.to_owned()
        } else {
            format!("{prefix}.{name}")
        };
        tensors.insert(key, Tensor::from_vec(data, shape, &Device::Cpu)?);
        Ok(())
    };
    insert("embed_tokens.weight", &[cfg.vocab_size, cfg.hidden_size])?;
    insert("norm.weight", &[cfg.hidden_size])?;
    for layer in 0..cfg.num_hidden_layers {
        let (dim, kv_heads) = if layer == 0 {
            (cfg.head_dim, cfg.num_key_value_heads)
        } else if cfg.attention_k_eq_v {
            (cfg.global_head_dim, cfg.num_global_key_value_heads.unwrap())
        } else {
            (cfg.global_head_dim, cfg.num_key_value_heads)
        };
        for (proj, rows, cols) in [
            ("q_proj", cfg.num_attention_heads * dim, cfg.hidden_size),
            ("k_proj", kv_heads * dim, cfg.hidden_size),
            ("o_proj", cfg.hidden_size, cfg.num_attention_heads * dim),
        ] {
            insert(
                &format!("layers.{layer}.self_attn.{proj}.weight"),
                &[rows, cols],
            )?;
        }
        // Only K=V global attention omits the V projection tensor.
        if layer == 0 || !cfg.attention_k_eq_v {
            insert(
                &format!("layers.{layer}.self_attn.v_proj.weight"),
                &[kv_heads * dim, cfg.hidden_size],
            )?;
        }
        for norm in ["q_norm", "k_norm"] {
            insert(&format!("layers.{layer}.self_attn.{norm}.weight"), &[dim])?;
        }
        for (proj, rows, cols) in [
            ("gate_proj", cfg.intermediate_size, cfg.hidden_size),
            ("up_proj", cfg.intermediate_size, cfg.hidden_size),
            ("down_proj", cfg.hidden_size, cfg.intermediate_size),
        ] {
            insert(&format!("layers.{layer}.mlp.{proj}.weight"), &[rows, cols])?;
        }
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            insert(&format!("layers.{layer}.{norm}.weight"), &[cfg.hidden_size])?;
        }
    }
    if !cfg.tie_word_embeddings {
        insert("lm_head.weight", &[cfg.vocab_size, cfg.hidden_size])?;
    }
    for (layer, scalar) in [0.8f32, 1.3].into_iter().enumerate() {
        tensors.insert(
            format!("{prefix}.layers.{layer}.layer_scalar"),
            Tensor::new(&[scalar], &Device::Cpu)?,
        );
    }
    Ok(tensors)
}

fn load(
    cfg: &Gemma4TextConfig,
    prefix: &str,
    tensors: HashMap<String, Tensor>,
) -> Result<TextModel> {
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    match prefix {
        "model" => TextModel::new(cfg, vb),
        "model.language_model" => TextModel::new_for_conditional_generation(cfg, vb),
        _ => unreachable!(),
    }
}

fn ids(tokens: &[u32]) -> Result<Tensor> {
    Tensor::from_slice(tokens, (1, tokens.len()), &Device::Cpu)
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= tolerance,
            "logit {i}: actual={a}, expected={e}, tolerance={tolerance}"
        );
    }
    Ok(())
}

#[test]
fn checkpoint_prefixes_and_root_untied_head_match() -> Result<()> {
    for tied in [true, false] {
        let mut cfg = config();
        cfg.tie_word_embeddings = tied;
        let mut outputs = Vec::new();
        for prefix in PREFIXES {
            let tensors = weights(&cfg, prefix)?;
            assert!(!tensors.contains_key(&format!("{prefix}.layers.1.self_attn.v_proj.weight")));
            assert!(tensors
                .keys()
                .all(|key| { key == "lm_head.weight" || key.starts_with(&format!("{prefix}.")) }));
            assert!(!tensors.keys().any(|key| key.contains(".model.")));
            assert_eq!(tensors.contains_key("lm_head.weight"), !tied);
            let mut model = load(&cfg, prefix, tensors)?;
            let logits = model.forward(&ids(&TOKENS)?, 0)?;
            assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
            let values = logits.flatten_all()?.to_vec1::<f32>()?;
            assert!(values.iter().any(|x| x.abs() > 1e-3));
            assert!(values.iter().all(|x| {
                x.is_finite() && x.abs() <= cfg.final_logit_softcapping.unwrap() as f32
            }));
            outputs.push(logits);
        }
        assert_close(&outputs[0], &outputs[1], 0.)?;
    }
    Ok(())
}

#[test]
fn cached_decoding_matches_fresh_prefills_across_local_window() -> Result<()> {
    let cfg = config();
    for prefix in PREFIXES {
        let tensors = weights(&cfg, prefix)?;
        let mut fresh = load(&cfg, prefix, tensors.clone())?;
        let mut expected = Vec::new();
        for end in 1..=TOKENS.len() {
            fresh.clear_kv_cache();
            expected.push(fresh.forward(&ids(&TOKENS[..end])?, 0)?);
        }
        let mut cached = load(&cfg, prefix, tensors)?;
        // Exercise prefill below and above the window, and multiple cache wraps.
        for prefill in [1, 2, 4, 7] {
            for _ in 0..2 {
                cached.clear_kv_cache();
                assert_close(
                    &cached.forward(&ids(&TOKENS[..prefill])?, 0)?,
                    &expected[prefill - 1],
                    2e-5,
                )?;
                for offset in prefill..TOKENS.len() {
                    assert_close(
                        &cached.forward(&ids(&TOKENS[offset..offset + 1])?, offset)?,
                        &expected[offset],
                        2e-5,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn chunked_prefills_match_training_for_batched_inputs_and_window_one() -> Result<()> {
    for window in [1, 3] {
        let mut cfg = config();
        cfg.sliding_window = window;
        let mut model = load(&cfg, "model", weights(&cfg, "model")?)?;
        let input = Tensor::new(&[TOKENS, [2, 3, 4, 5, 6, 7, 8, 9, 1]], &Device::Cpu)?;
        let expected = model.forward_train(&input)?;
        for chunk in 1..=5 {
            model.clear_kv_cache();
            for offset in (0..TOKENS.len()).step_by(chunk) {
                let len = chunk.min(TOKENS.len() - offset);
                assert_close(
                    &model.forward(&input.narrow(1, offset, len)?, offset)?,
                    &expected.narrow(1, offset + len - 1, 1)?,
                    2e-5,
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn official_transformers_5_19_0_golden_prefills_and_cached_tokens() -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Cached {
        prefill: usize,
        logits: Vec<Vec<f32>>,
    }
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        config: Gemma4TextConfig,
        layer_scalars: Vec<f32>,
        fresh: Vec<Vec<f32>>,
        cached: Vec<Cached>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        tokens: Vec<u32>,
        cases: Vec<Case>,
    }

    // Real HF CPU/F32 outputs, not Candle self-comparisons. Regeneration and
    // provenance are documented in gemma4_reference.md; Python is not run here.
    let fixture: Fixture = serde_json::from_str(include_str!("gemma4_reference.json")).unwrap();
    assert_eq!(fixture.tokens, TOKENS);
    assert_eq!(
        fixture
            .cases
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        [
            "tied",
            "untied",
            "half_rope_negative_scalars",
            "separate_kv_ignores_global_override"
        ]
    );
    for case in fixture.cases {
        let cfg = &case.config;
        if case.name == "tied" {
            let mut defaults = cfg.clone();
            // HF records explicit RoPE defaults, while config() tests fallback defaults.
            defaults.rope_parameters = None;
            assert_eq!(defaults, config());
        }
        if case.name == "separate_kv_ignores_global_override" {
            assert!(!cfg.attention_k_eq_v);
            assert_eq!(cfg.num_key_value_heads, 2);
            assert_eq!(cfg.num_global_key_value_heads, Some(1));
        }
        assert_eq!(case.fresh.len(), TOKENS.len());
        assert_eq!(case.layer_scalars.len(), cfg.num_hidden_layers);
        assert_eq!(
            case.cached.iter().map(|c| c.prefill).collect::<Vec<_>>(),
            [1, 2, 4, 7]
        );
        for prefix in PREFIXES {
            let mut tensors = weights(cfg, prefix)?;
            if !cfg.attention_k_eq_v {
                for proj in ["k_proj", "v_proj"] {
                    assert_eq!(
                        tensors[&format!("{prefix}.layers.1.self_attn.{proj}.weight")].dims(),
                        &[16, 8]
                    );
                }
            }
            for (layer, scalar) in case.layer_scalars.iter().enumerate() {
                tensors.insert(
                    format!("{prefix}.layers.{layer}.layer_scalar"),
                    Tensor::new(&[*scalar], &Device::Cpu)?,
                );
            }
            let mut model = load(cfg, prefix, tensors)?;
            let check =
                |actual: Tensor, expected: &[f32], prefill: usize, end: usize| -> Result<()> {
                    assert_eq!(actual.dims(), &[1, 1, cfg.vocab_size]);
                    assert_eq!(expected.len(), cfg.vocab_size);
                    for (i, (actual, expected)) in actual
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .zip(expected)
                        .enumerate()
                    {
                        assert!(
                            actual.is_finite()
                                && expected.is_finite()
                                && (actual - expected).abs() <= 2e-5,
                            "HF golden {}/{prefix}, prefill={prefill}, end={end}, logit={i}: \
                         Candle={actual}, HF={expected}, atol=2e-5",
                            case.name
                        );
                    }
                    Ok(())
                };
            for end in 1..=TOKENS.len() {
                model.clear_kv_cache();
                check(
                    model.forward(&ids(&TOKENS[..end])?, 0)?,
                    &case.fresh[end - 1],
                    end,
                    end,
                )?;
            }
            for cached in &case.cached {
                let prefill = cached.prefill;
                assert_eq!(cached.logits.len(), TOKENS.len() - prefill + 1);
                model.clear_kv_cache();
                check(
                    model.forward(&ids(&TOKENS[..prefill])?, 0)?,
                    &cached.logits[0],
                    prefill,
                    prefill,
                )?;
                for offset in prefill..TOKENS.len() {
                    check(
                        model.forward(&ids(&TOKENS[offset..offset + 1])?, offset)?,
                        &cached.logits[offset - prefill + 1],
                        prefill,
                        offset + 1,
                    )?;
                }
            }
            // Includes a full ring followed by a two-token chunk (the first
            // query still needs keys evicted by appending the second token).
            for chunks in [
                vec![1; 9],
                vec![2, 2, 2, 2, 1],
                vec![3, 2, 3, 1],
                vec![4, 4, 1],
                vec![1, 5, 2, 1],
            ] {
                model.clear_kv_cache();
                let mut offset = 0;
                for len in chunks {
                    let end = offset + len;
                    check(
                        model.forward(&ids(&TOKENS[offset..end])?, offset)?,
                        &case.fresh[end - 1],
                        len,
                        end,
                    )?;
                    offset = end;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn prevalidation_errors_preserve_cache_and_reset_allows_reuse() -> Result<()> {
    let cfg = config();
    for prefix in PREFIXES {
        let tensors = weights(&cfg, prefix)?;
        let mut clean = load(&cfg, prefix, tensors.clone())?;
        clean.forward(&ids(&TOKENS[..2])?, 0)?;
        let expected = clean.forward(&ids(&TOKENS[2..3])?, 2)?;
        for case in [
            "overflow",
            "offset",
            "batch",
            "empty",
            "empty_batch",
            "shape",
        ] {
            let mut model = load(&cfg, prefix, tensors.clone())?;
            model.forward(&ids(&TOKENS[..2])?, 0)?;
            let (result, message) = match case {
                "overflow" => (
                    model.forward(&ids(&vec![1; cfg.max_position_embeddings - 1])?, 2),
                    "max_position_embeddings",
                ),
                "offset" => (
                    model.forward(&ids(&TOKENS[2..3])?, 1),
                    "seqlen_offset must equal cached length 2",
                ),
                "batch" => (
                    model.forward(&Tensor::new(&[[3u32], [4u32]], &Device::Cpu)?, 2),
                    "batch size must remain unchanged",
                ),
                "empty" => (model.forward(&ids(&[])?, 2), "nonempty embeddings"),
                "empty_batch" => (
                    model.forward_embeds(
                        &Tensor::zeros((0, 1, cfg.hidden_size), DType::F32, &Device::Cpu)?,
                        2,
                        0,
                        1,
                    ),
                    "nonempty embeddings",
                ),
                "shape" => (
                    model.forward_embeds(
                        &Tensor::zeros((1, 1, cfg.hidden_size + 1), DType::F32, &Device::Cpu)?,
                        2,
                        1,
                        1,
                    ),
                    "shape [batch, sequence, hidden_size]",
                ),
                _ => unreachable!(),
            };
            let error = result.unwrap_err().to_string();
            assert!(error.contains(message), "{prefix}/{case}: {error}");
            assert_close(&model.forward(&ids(&TOKENS[2..3])?, 2)?, &expected, 0.)?;
        }

        let mut model = load(&cfg, prefix, tensors.clone())?;
        let too_long = ids(&vec![1; cfg.max_position_embeddings + 1])?;
        let error = model.forward(&too_long, 0).unwrap_err().to_string();
        assert!(error.contains("max_position_embeddings"), "{error}");
        let full = ids(&vec![1; cfg.max_position_embeddings])?;
        let mut fresh = load(&cfg, prefix, tensors)?;
        assert_close(&model.forward(&full, 0)?, &fresh.forward(&full, 0)?, 0.)?;
        for _ in 0..2 {
            let error = model
                .forward(&ids(&[2])?, cfg.max_position_embeddings)
                .unwrap_err()
                .to_string();
            assert!(error.contains("max_position_embeddings"), "{error}");
        }
        model.clear_kv_cache();
        fresh.clear_kv_cache();
        let new_batch = Tensor::new(&[[2u32, 3], [4u32, 5]], &Device::Cpu)?;
        assert_close(
            &model.forward(&new_batch, 0)?,
            &fresh.forward(&new_batch, 0)?,
            0.,
        )?;
    }
    Ok(())
}

#[test]
fn last_layer_scalar_and_final_softcap_affect_logits() -> Result<()> {
    for prefix in PREFIXES {
        let mut cfg = config();
        let tensors = weights(&cfg, prefix)?;
        let input = ids(&TOKENS)?;
        let capped = load(&cfg, prefix, tensors.clone())?.forward(&input, 0)?;
        let cap = cfg.final_logit_softcapping.take().unwrap();
        let uncapped = load(&cfg, prefix, tensors.clone())?.forward(&input, 0)?;
        let expected = ((&uncapped / cap)?.tanh()? * cap)?;
        assert_close(&capped, &expected, 1e-6)?;
        assert!((&capped - &uncapped)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .any(|x| *x > 1e-3));
        cfg.final_logit_softcapping = Some(cap);
        let mut zero_scalar = tensors;
        zero_scalar.insert(
            format!("{prefix}.layers.1.layer_scalar"),
            Tensor::new(&[0f32], &Device::Cpu)?,
        );
        let zero = load(&cfg, prefix, zero_scalar)?.forward(&input, 0)?;
        assert_close(&zero, &Tensor::zeros_like(&capped)?, 0.)?;
    }
    Ok(())
}

#[test]
fn missing_tensors_report_checkpoint_paths() -> Result<()> {
    let mut cfg = config();
    cfg.tie_word_embeddings = false;
    for prefix in PREFIXES {
        for missing in [
            format!("{prefix}.embed_tokens.weight"),
            format!("{prefix}.layers.0.self_attn.v_proj.weight"),
            format!("{prefix}.layers.1.self_attn.k_proj.weight"),
            format!("{prefix}.layers.1.layer_scalar"),
            "lm_head.weight".to_owned(),
        ] {
            let mut tensors = weights(&cfg, prefix)?;
            assert!(tensors.remove(&missing).is_some());
            let error = load(&cfg, prefix, tensors).unwrap_err().to_string();
            assert!(error.contains(&missing), "expected {missing}: {error}");
        }
    }
    Ok(())
}
