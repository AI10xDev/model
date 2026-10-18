use std::collections::HashMap;

use candle::quantized::{GgmlDType, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap, SGD};
use candle_transformers::models::gemma4::{config::Gemma4TextConfig, text::TextModel};

fn config(shared_kv: bool) -> Gemma4TextConfig {
    serde_json::from_value(serde_json::json!({
        "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 32, "global_head_dim": 64, "vocab_size": 17,
        "sliding_window": 3, "max_position_embeddings": 12,
        "layer_types": ["sliding_attention", "full_attention"],
        "attention_bias": true, "attention_k_eq_v": shared_kv,
        "tie_word_embeddings": shared_kv, "final_logit_softcapping": 4.0
    }))
    .unwrap()
}

fn values(name: &str, len: usize, scale: f32) -> Vec<f32> {
    let seed = name
        .bytes()
        .fold(17u32, |s, b| s.wrapping_mul(31).wrapping_add(b as u32));
    (0..len)
        .map(|i| {
            let n = seed.wrapping_add((i as u32).wrapping_mul(747796405));
            let n = (n ^ (n >> 16)).wrapping_mul(2246822519);
            (((n ^ (n >> 13)) % 401) as f32 / 200. - 1.) * scale
        })
        .collect()
}

fn weights(cfg: &Gemma4TextConfig, dtype: DType, dev: &Device) -> Result<HashMap<String, Tensor>> {
    let vars = VarMap::new();
    TextModel::new(
        cfg,
        VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
    )?;
    let data = vars.data().lock().unwrap();
    data.iter()
        .map(|(name, var)| {
            let data = if name.contains("norm") || name.ends_with("layer_scalar") {
                vec![1f32; var.elem_count()]
            } else {
                values(name, var.elem_count(), 0.1)
            };
            Ok((
                name.clone(),
                Tensor::from_vec(data, var.shape(), dev)?.to_dtype(dtype)?,
            ))
        })
        .collect()
}

fn input(dev: &Device) -> Result<Tensor> {
    Tensor::new(&[[1u32, 7, 3, 10], [4, 6, 5, 9]], dev)
}

fn loss(model: &mut TextModel, ids: &Tensor) -> Result<Tensor> {
    let logits = model.forward_train(ids)?.to_dtype(DType::F32)?;
    let target = Tensor::from_vec(
        values("target", logits.elem_count(), 1.),
        logits.shape(),
        ids.device(),
    )?;
    (logits * target)?.sum_all()
}

fn max_abs(t: &Tensor) -> Result<f32> {
    t.to_dtype(DType::F32)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar()
}

// Smooth reference for the requested dX = dY dequantize(W). GGML CPU kernels
// also quantize activations, so infinitesimal differences of the packed forward
// itself need not equal this derivative. Fold LoRA into dequantized weights only
// in this test reference, never in the production training path.
fn reference(
    cfg: &Gemma4TextConfig,
    base: &HashMap<String, Tensor>,
    vars: &VarMap,
) -> Result<TextModel> {
    let mut tensors = base.clone();
    let vars = vars.data().lock().unwrap();
    for (name, a) in vars.iter().filter(|(name, _)| name.ends_with("lora_a")) {
        let prefix = name.trim_end_matches(".lora_a");
        let b = &vars[&format!("{prefix}.lora_b")];
        let key = format!("model.{prefix}.weight");
        let weight = QTensor::quantize(&base[&key], GgmlDType::Q4_0)?.dequantize(&Device::Cpu)?;
        tensors.insert(key, (weight + (b.matmul(a)? * 2.)?)?);
    }
    TextModel::new(
        cfg,
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
    )
}

#[test]
fn q4_gradients_cross_two_layers_and_match_numerical_reference() -> Result<()> {
    let cfg = config(false);
    let mut base = weights(&cfg, DType::F32, &Device::Cpu)?;
    // Keep the unscaled attention logits away from saturation so activation
    // quantization noise does not dominate the smooth-reference comparison.
    for (name, weight) in &mut base {
        if name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight") {
            *weight = weight.affine(0.25, 0.)?;
        }
    }
    let mut model = TextModel::new_quantized(
        &cfg,
        VarBuilder::from_tensors(base.clone(), DType::F32, &Device::Cpu),
        GgmlDType::Q4_0,
    )?;
    let vars = VarMap::new();
    assert_eq!(
        model.add_quantized_lora(
            4,
            8.,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu)
        )?,
        14
    );
    for (name, var) in vars.data().lock().unwrap().iter() {
        var.set(&Tensor::from_vec(
            values(name, var.elem_count(), 0.02),
            var.shape(),
            &Device::Cpu,
        )?)?;
    }
    let ids = input(&Device::Cpu)?;
    let grads = loss(&mut model, &ids)?.backward()?;
    let expected = loss(&mut reference(&cfg, &base, &vars)?, &ids)?.backward()?;
    for (name, var) in vars.data().lock().unwrap().iter() {
        let actual = grads.get(var).unwrap();
        let expected = expected.get(var).unwrap();
        let error = max_abs(&(actual - expected)?)?;
        let magnitude = max_abs(expected)?;
        assert!(magnitude > 1e-6, "{name}: missing signal");
        assert!(
            error < 0.06 * magnitude + 1e-4,
            "{name}: error {error}, magnitude {magnitude}"
        );
    }
    // Check individual coordinates in the earliest adapter, for both masters.
    for suffix in ["lora_a", "lora_b"] {
        let name = format!("layers.0.self_attn.q_proj.{suffix}");
        let var = vars.data().lock().unwrap()[&name].clone();
        let analytic = expected
            .get(&var)
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        let index = (0..analytic.len())
            .max_by(|&i, &j| analytic[i].abs().total_cmp(&analytic[j].abs()))
            .unwrap();
        let original = var.flatten_all()?.to_vec1::<f32>()?;
        let eps = 0.001f32;
        let eval = |delta: f32| -> Result<f32> {
            let mut data = original.clone();
            data[index] += delta;
            var.set(&Tensor::from_vec(data, var.shape(), &Device::Cpu)?)?;
            loss(&mut reference(&cfg, &base, &vars)?, &ids)?.to_scalar()
        };
        let numerical = (eval(eps)? - eval(-eps)?) / (2. * eps);
        var.set(&Tensor::from_vec(original, var.shape(), &Device::Cpu)?)?;
        assert!(
            (numerical - analytic[index]).abs() < 0.02 * analytic[index].abs() + 0.003,
            "{name}: numerical {numerical}, analytic {}",
            analytic[index]
        );
        let quantized = grads.get(&var).unwrap().flatten_all()?.to_vec1::<f32>()?[index];
        assert!(
            (quantized - numerical).abs() < 0.06 * numerical.abs() + 0.003,
            "{name}: numerical {numerical}, quantized-path gradient {quantized}"
        );
    }
    Ok(())
}

fn training_smoke(dtype: DType, dev: &Device, shared_kv: bool) -> Result<()> {
    let cfg = config(shared_kv);
    let base = weights(&cfg, dtype, dev)?;
    let snapshots = base
        .iter()
        .map(|(n, t)| Ok((n.clone(), t.copy()?)))
        .collect::<Result<HashMap<_, _>>>()?;
    let mut model = TextModel::new_quantized(
        &cfg,
        VarBuilder::from_tensors(base.clone(), dtype, dev),
        GgmlDType::Q4_0,
    )?;
    let ids = input(dev)?;
    let before = model.forward_train(&ids)?;
    assert!(!before.track_op());
    let vars = VarMap::new();
    // A BF16 builder must still allocate F32 master variables.
    let count = model.add_quantized_lora(4, 8., VarBuilder::from_varmap(&vars, dtype, dev))?;
    assert_eq!(count, if shared_kv { 13 } else { 14 });
    assert_eq!(vars.all_vars().len(), count * 2);
    assert_eq!(max_abs(&(model.forward_train(&ids)? - &before)?)?, 0.);
    let mut a_sq = 0.;
    let mut a_count = 0;
    for (name, var) in vars.data().lock().unwrap().iter() {
        assert_eq!(var.dtype(), DType::F32);
        assert!(name.starts_with("layers."));
        if name.ends_with("lora_b") {
            assert_eq!(max_abs(var)?, 0.);
            assert_eq!(var.dim(1)?, 4);
        } else {
            assert!(name.ends_with("lora_a"));
            assert_eq!(var.dim(0)?, 4);
            a_sq += var.sqr()?.sum_all()?.to_scalar::<f32>()?;
            a_count += var.elem_count();
            // Make optimizer coverage deterministic after checking initialization.
            var.set(&Tensor::from_vec(
                values(name, var.elem_count(), 0.02),
                var.shape(),
                dev,
            )?)?;
        }
    }
    assert!((0.015..0.025).contains(&(a_sq / a_count as f32).sqrt()));
    let mut optimizer = SGD::new(vars.all_vars(), 0.01)?;
    for step in 0..2 {
        let old = vars
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(n, v)| Ok((n.clone(), v.copy()?)))
            .collect::<Result<HashMap<_, _>>>()?;
        let grads = loss(&mut model, &ids)?.backward()?;
        for (name, var) in vars.data().lock().unwrap().iter() {
            let grad = grads.get(var).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(grad.dtype(), DType::F32);
            let size = max_abs(grad)?;
            assert!(size.is_finite());
            if step == 0 && name.ends_with("lora_a") {
                assert_eq!(size, 0.);
            } else {
                assert!(size > 0., "step {step}: zero gradient for {name}");
            }
        }
        optimizer.step(&grads)?;
        for (name, var) in vars.data().lock().unwrap().iter() {
            let change = max_abs(&(var.as_tensor() - &old[name])?)?;
            if step == 0 && name.ends_with("lora_a") {
                assert_eq!(change, 0.);
            } else {
                assert!(change > 0., "step {step}: no update to {name}");
            }
        }
    }
    assert!(max_abs(&(model.forward_train(&ids)? - &before)?)? > 0.);
    assert_eq!(model.quantize_frozen_decoder(GgmlDType::Q4_0)?, 0);
    for (name, tensor) in &base {
        assert!(!tensor.track_op());
        assert_eq!(
            max_abs(&(tensor - &snapshots[name])?)?,
            0.,
            "base changed: {name}"
        );
    }
    // Removing the adapter contribution restores the original quantized model,
    // including its dense embedding/head/norms and packed projection weights.
    for (name, var) in vars.data().lock().unwrap().iter() {
        if name.ends_with("lora_b") {
            var.set(&var.zeros_like()?)?;
        }
    }
    assert_eq!(max_abs(&(model.forward_train(&ids)? - before)?)?, 0.);
    Ok(())
}

#[test]
fn q4_f32_updates_every_adapter_and_freezes_base() -> Result<()> {
    for shared in [false, true] {
        training_smoke(DType::F32, &Device::Cpu, shared)?;
    }
    Ok(())
}

#[test]
fn q4_bf16_cpu_master_casts_are_fresh() -> Result<()> {
    training_smoke(DType::BF16, &Device::Cpu, false)
}

#[cfg(feature = "cuda")]
#[test]
fn q4_bf16_cuda_smoke() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    training_smoke(DType::BF16, &dev, true)
}

#[test]
fn streamed_constructors_match_conversion_and_reject_trainable_base() -> Result<()> {
    for shared in [false, true] {
        let cfg = config(shared);
        let base = weights(&cfg, DType::F32, &Device::Cpu)?;
        for quant in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
            let vb = VarBuilder::from_tensors(base.clone(), DType::F32, &Device::Cpu);
            let mut converted = TextModel::new(&cfg, vb.clone())?;
            converted.quantize_frozen_decoder(quant)?;
            let mut streamed = TextModel::new_quantized(&cfg, vb, quant)?;
            let conditional = base
                .iter()
                .map(|(n, t)| (n.replacen("model.", "model.language_model.", 1), t.clone()))
                .collect();
            let mut conditional = TextModel::new_quantized_for_conditional_generation(
                &cfg,
                VarBuilder::from_tensors(conditional, DType::F32, &Device::Cpu),
                quant,
            )?;
            let ids = input(&Device::Cpu)?;
            let expected = converted.forward_train(&ids)?;
            assert_eq!(max_abs(&(streamed.forward_train(&ids)? - &expected)?)?, 0.);
            assert_eq!(
                max_abs(&(conditional.forward_train(&ids)? - expected)?)?,
                0.
            );
            let vars = VarMap::new();
            let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
            assert!(TextModel::new_quantized(&cfg, vb.clone(), quant)
                .unwrap_err()
                .to_string()
                .contains("trainable"));
            assert!(
                TextModel::new_quantized_for_conditional_generation(&cfg, vb, quant)
                    .unwrap_err()
                    .to_string()
                    .contains("trainable")
            );
        }
    }
    Ok(())
}

#[test]
fn rejects_invalid_or_duplicate_adapters_without_allocating_variables() -> Result<()> {
    let cfg = config(false);
    let base = weights(&cfg, DType::F32, &Device::Cpu)?;
    let mut model = TextModel::new(
        &cfg,
        VarBuilder::from_tensors(base, DType::F32, &Device::Cpu),
    )?;
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
    assert!(model.add_quantized_lora(4, 8., vb.clone()).is_err());
    model.quantize_frozen_decoder(GgmlDType::Q4_0)?;
    for (rank, alpha) in [
        (0, 8.),
        (4, f64::NAN),
        (4, f64::INFINITY),
        (4, 0.),
        (4, -1.),
    ] {
        assert!(model.add_quantized_lora(rank, alpha, vb.clone()).is_err());
    }
    assert!(vars.all_vars().is_empty());
    model.add_quantized_lora(4, 8., vb)?;
    let duplicate = VarMap::new();
    assert!(model
        .add_quantized_lora(
            4,
            8.,
            VarBuilder::from_varmap(&duplicate, DType::F32, &Device::Cpu)
        )
        .is_err());
    assert!(duplicate.all_vars().is_empty());
    Ok(())
}
