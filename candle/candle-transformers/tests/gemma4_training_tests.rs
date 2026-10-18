use candle::{DType, Device, Result, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap, SGD};
use candle_transformers::models::gemma4::{config::Gemma4TextConfig, text::TextModel};

fn config() -> Gemma4TextConfig {
    serde_json::from_value(serde_json::json!({
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 4,
        "global_head_dim": 8,
        "vocab_size": 16,
        "sliding_window": 3,
        "max_position_embeddings": 12,
        "layer_types": ["sliding_attention", "full_attention"],
        "final_logit_softcapping": 2.0
    }))
    .unwrap()
}

fn model(cfg: &Gemma4TextConfig, conditional: bool) -> Result<(TextModel, VarMap)> {
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
    let model = if conditional {
        TextModel::new_for_conditional_generation(cfg, vb)?
    } else {
        TextModel::new(cfg, vb)?
    };
    // Deterministic, nondegenerate projections without changing scratch norm/scalar defaults.
    for (name, var) in vars.data().lock().unwrap().iter() {
        if name.contains("norm") || name.ends_with("layer_scalar") {
            assert!(var
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| *v == 1.));
            continue;
        }
        let seed = name
            .bytes()
            .fold(17u32, |s, b| s.wrapping_mul(31).wrapping_add(u32::from(b)));
        let data: Vec<_> = (0..var.elem_count())
            .map(|i| {
                let n = seed.wrapping_add((i as u32).wrapping_mul(747796405));
                let n = (n ^ (n >> 16)).wrapping_mul(2246822519);
                ((n ^ (n >> 13)) % 401) as f32 / 1000. - 0.2
            })
            .collect();
        var.set(&Tensor::from_vec(data, var.shape(), &Device::Cpu)?)?;
    }
    Ok((model, vars))
}

fn ids() -> Result<Tensor> {
    Tensor::new(
        &[[1u32, 7, 3, 10, 2, 8], [4, 6, 5, 9, 11, 12]],
        &Device::Cpu,
    )
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    for (a, e) in actual
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .zip(expected.flatten_all()?.to_vec1::<f32>()?)
    {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= tolerance,
            "actual={a}, expected={e}, tolerance={tolerance}"
        );
    }
    Ok(())
}

fn loss(model: &mut TextModel, input: &Tensor) -> Result<Tensor> {
    let (_, seq_len) = input.dims2()?;
    let logits = model.forward_train(&input.narrow(1, 0, seq_len - 1)?)?;
    let vocab = logits.dim(2)?;
    candle_nn::loss::cross_entropy(
        &logits.reshape(((), vocab))?,
        &input.narrow(1, 1, seq_len - 1)?.flatten_all()?,
    )
}

#[test]
fn all_training_positions_match_inference_prefixes_and_cached_tokens() -> Result<()> {
    let input = ids()?;
    for shared_kv in [false, true] {
        for tied in [false, true] {
            let mut cfg = config();
            cfg.attention_k_eq_v = shared_kv;
            cfg.tie_word_embeddings = tied;
            cfg.attention_bias = !tied;
            if !tied {
                cfg.final_logit_softcapping = None;
            }
            let (mut model, _) = model(&cfg, !tied)?;
            let all = model.forward_train(&input)?;
            assert_eq!(all.dims(), &[2, 6, cfg.vocab_size]);
            for pos in 0..6 {
                model.clear_kv_cache();
                let last = model.forward(&input.narrow(1, 0, pos + 1)?, 0)?;
                assert_close(&all.narrow(1, pos, 1)?, &last, 2e-5)?;
            }
            model.clear_kv_cache();
            for pos in 0..6 {
                let last = model.forward(&input.narrow(1, pos, 1)?, pos)?;
                assert_close(&all.narrow(1, pos, 1)?, &last, 2e-5)?;
            }
        }
    }
    Ok(())
}

#[test]
fn training_is_causal_and_neither_reads_nor_changes_inference_cache() -> Result<()> {
    let (mut model, _) = model(&config(), false)?;
    let input = ids()?;
    let expected = model.forward_train(&input)?;
    let changed = Tensor::cat(
        &[
            &input.narrow(1, 0, 3)?,
            &Tensor::new(&[[0u32, 0, 0], [1, 1, 1]], &Device::Cpu)?,
        ],
        1,
    )?;
    assert_close(
        &model.forward_train(&changed)?.narrow(1, 0, 3)?,
        &expected.narrow(1, 0, 3)?,
        0.,
    )?;
    model.forward(&input.narrow(1, 0, 4)?, 0)?;
    // Cache has already wrapped locally. Training may use a different batch size.
    assert_close(&model.forward_train(&input)?, &expected, 0.)?;
    assert_close(
        &model.forward_train(&input.narrow(0, 0, 1)?)?,
        &expected.narrow(0, 0, 1)?,
        2e-5,
    )?;
    assert_close(
        &model.forward(&input.narrow(1, 4, 1)?, 4)?,
        &expected.narrow(1, 4, 1)?,
        2e-5,
    )?;
    Ok(())
}

#[test]
fn training_validates_inputs_without_affecting_cache() -> Result<()> {
    let cfg = config();
    let (mut model, _) = model(&cfg, false)?;
    let input = ids()?;
    let expected = model.forward_train(&input)?.narrow(1, 1, 1)?;
    model.forward(&input.narrow(1, 0, 1)?, 0)?;
    for shape in [(0, 1), (1, 0), (1, cfg.max_position_embeddings + 1)] {
        assert!(model
            .forward_train(&Tensor::zeros(shape, DType::U32, &Device::Cpu)?)
            .is_err());
    }
    assert!(model.forward_train(&input.flatten_all()?).is_err());
    assert_close(&model.forward(&input.narrow(1, 1, 1)?, 1)?, &expected, 2e-5)?;
    Ok(())
}

#[test]
fn qkv_and_mlp_gradients_are_finite_nonzero_and_update_parameters() -> Result<()> {
    let input = ids()?;
    for shared_kv in [false, true] {
        let mut cfg = config();
        cfg.attention_k_eq_v = shared_kv;
        cfg.tie_word_embeddings = !shared_kv;
        let (mut model, vars) = model(&cfg, false)?;
        let mut optimizer = SGD::new(vars.all_vars(), 0.001)?;
        for _ in 0..2 {
            let before = loss(&mut model, &input)?;
            let grads = before.backward()?;
            let mut checked = Vec::new();
            for (name, var) in vars.data().lock().unwrap().iter() {
                let grad = grads
                    .get(var)
                    .unwrap_or_else(|| panic!("missing gradient: {name}"));
                let values = grad.flatten_all()?.to_vec1::<f32>()?;
                assert!(
                    values.iter().all(|v| v.is_finite()),
                    "nonfinite gradient: {name}"
                );
                // The last scalar is almost scale-invariant under final RMSNorm.
                if !name.ends_with("layer_scalar") {
                    assert!(
                        values.iter().any(|v| v.abs() > 1e-9),
                        "zero gradient: {name}"
                    );
                    checked.push((
                        name.clone(),
                        var.clone(),
                        var.flatten_all()?.to_vec1::<f32>()?,
                    ));
                }
            }
            for layer in 0..2 {
                for proj in ["q_proj", "k_proj", "v_proj"] {
                    let name = format!("model.layers.{layer}.self_attn.{proj}.weight");
                    assert_eq!(
                        checked.iter().any(|(n, _, _)| n == &name),
                        !(shared_kv && layer == 1 && proj == "v_proj")
                    );
                }
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    assert!(checked
                        .iter()
                        .any(|(n, _, _)| n == &format!("model.layers.{layer}.mlp.{proj}.weight")));
                }
            }
            optimizer.step(&grads)?;
            for (name, var, old) in checked {
                let new = var.flatten_all()?.to_vec1::<f32>()?;
                assert!(
                    new.iter().all(|v| v.is_finite()),
                    "nonfinite update: {name}"
                );
                assert!(
                    new.iter().zip(old).any(|(a, b)| *a != b),
                    "no update: {name}"
                );
            }
            let after = loss(&mut model, &input)?.to_scalar::<f32>()?;
            assert!(after < before.to_scalar::<f32>()?, "loss did not decrease");
        }
    }
    Ok(())
}

#[test]
fn attention_gradients_match_finite_differences_through_rope_and_softmax() -> Result<()> {
    let (mut model, vars) = model(&config(), false)?;
    let input = ids()?;
    let grads = loss(&mut model, &input)?.backward()?;
    for layer in 0..2 {
        for proj in ["q_proj", "k_proj", "v_proj"] {
            let name = format!("model.layers.{layer}.self_attn.{proj}.weight");
            let var = vars.data().lock().unwrap()[&name].clone();
            let values = grads.get(&var).unwrap().flatten_all()?.to_vec1::<f32>()?;
            let (index, analytical) = values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .unwrap();
            let original = var.flatten_all()?.to_vec1::<f32>()?;
            let mut perturbed = original.clone();
            let eps = 1e-3;
            perturbed[index] = original[index] + eps;
            var.set(&Tensor::from_vec(
                perturbed.clone(),
                var.shape(),
                &Device::Cpu,
            )?)?;
            let plus = loss(&mut model, &input)?.to_scalar::<f32>()?;
            perturbed[index] = original[index] - eps;
            var.set(&Tensor::from_vec(perturbed, var.shape(), &Device::Cpu)?)?;
            let minus = loss(&mut model, &input)?.to_scalar::<f32>()?;
            var.set(&Tensor::from_vec(original, var.shape(), &Device::Cpu)?)?;
            let numerical = (plus - minus) / (2. * eps);
            assert!(
                (analytical - numerical).abs() < 2e-4 + 0.02 * analytical.abs(),
                "{name}: analytical={analytical}, numerical={numerical}"
            );
        }
    }
    Ok(())
}

#[test]
fn trained_checkpoint_roundtrip_preserves_logits_and_trainability() -> Result<()> {
    let input = ids()?;
    for conditional in [false, true] {
        let mut cfg = config();
        cfg.tie_word_embeddings = !conditional;
        let (mut trained, vars) = model(&cfg, conditional)?;
        SGD::new(vars.all_vars(), 0.001)?.backward_step(&loss(&mut trained, &input)?)?;
        // Non-default values ensure loading never replaces checkpoint values with init hints.
        for (name, var) in vars.data().lock().unwrap().iter() {
            if name.contains("norm") || name.ends_with("layer_scalar") {
                var.set(&var.affine(0.7, 0.)?)?;
            }
        }
        let expected = trained.forward_train(&input)?;
        let path = std::env::temp_dir().join(format!(
            "gemma4-training-{}-{conditional}.safetensors",
            std::process::id()
        ));
        let result = (|| -> Result<()> {
            vars.save(&path)?;
            let vb = VarBuilder::from_buffered_safetensors(
                std::fs::read(&path)?,
                DType::F32,
                &Device::Cpu,
            )?;
            let mut loaded = if conditional {
                TextModel::new_for_conditional_generation(&cfg, vb)?
            } else {
                TextModel::new(&cfg, vb)?
            };
            assert_close(&loaded.forward_train(&input)?, &expected, 0.)?;
            assert_close(
                &loaded.forward(&input, 0)?,
                &expected.narrow(1, 5, 1)?,
                2e-5,
            )?;
            let (mut resumed, mut resumed_vars) = model(&cfg, conditional)?;
            resumed_vars.load(&path)?;
            assert_close(&resumed.forward_train(&input)?, &expected, 0.)?;
            let before = loss(&mut resumed, &input)?;
            SGD::new(resumed_vars.all_vars(), 0.001)?.backward_step(&before)?;
            assert!(loss(&mut resumed, &input)?.to_scalar::<f32>()? < before.to_scalar::<f32>()?);
            Ok(())
        })();
        let _ = std::fs::remove_file(path);
        result?;
    }
    Ok(())
}
