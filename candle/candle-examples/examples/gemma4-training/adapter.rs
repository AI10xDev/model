use candle::{DType, Result, Tensor};
use candle_nn::{Init, VarBuilder};
use candle_transformers::models::gemma4::text::TextModel;

/// Output-head-only LoRA. The quantized backbone is frozen, not differentiable.
pub struct HeadAdapter {
    a: Tensor,
    b: Tensor,
    scale: f64,
    softcap: Option<f64>,
}

impl HeadAdapter {
    pub fn new(
        hidden: usize,
        vocab: usize,
        rank: usize,
        alpha: f64,
        softcap: Option<f64>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            a: vb.get_with_hints(
                (rank, hidden),
                "lora_a",
                Init::Randn {
                    mean: 0.,
                    stdev: 0.02,
                },
            )?,
            b: vb.get_with_hints((vocab, rank), "lora_b", Init::Const(0.))?,
            scale: alpha / rank as f64,
            softcap,
        })
    }

    fn forward(&self, base: &mut TextModel, input: &Tensor) -> Result<Tensor> {
        let (batch, sequence) = input.dims2()?;
        let hidden = base.forward_train_hidden(input)?.detach();
        let logits = base.project_logits(&hidden)?.to_dtype(DType::F32)?;
        // Recast on every forward: gradients reach the current F32 master variables.
        let a = self.a.to_dtype(hidden.dtype())?;
        let b = self.b.to_dtype(hidden.dtype())?;
        let delta = hidden.flatten_to(1)?.matmul(&a.t()?)?.matmul(&b.t()?)?;
        let delta = (delta.to_dtype(DType::F32)? * self.scale)?.reshape((batch, sequence, ()))?;
        let logits = (logits + delta)?;
        match self.softcap {
            None => Ok(logits),
            Some(sc) => (logits / sc)?.tanh()? * sc,
        }
    }
}

pub struct TrainingModel {
    pub base: TextModel,
    pub adapter: Option<HeadAdapter>,
}

impl TrainingModel {
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor> {
        match &self.adapter {
            None => self.base.forward_train(input),
            Some(adapter) => adapter.forward(&mut self.base, input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::GgmlDType;
    use candle::Device;
    use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
    use candle_transformers::models::gemma4::config::Gemma4TextConfig;
    use std::collections::HashMap;

    fn fixture(
        device: &Device,
        dtype: DType,
        quant: GgmlDType,
    ) -> Result<(TrainingModel, VarMap, HashMap<String, Tensor>)> {
        let config: Gemma4TextConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "global_head_dim": 32,
            "vocab_size": 17,
            "sliding_window": 2,
            "max_position_embeddings": 4,
            "layer_types": ["sliding_attention", "full_attention"],
            "tie_word_embeddings": true,
            "final_logit_softcapping": 0.5
        }))
        .unwrap();
        let source = VarMap::new();
        TextModel::new(
            &config,
            VarBuilder::from_varmap(&source, DType::F32, &Device::Cpu),
        )?;
        let mut frozen = HashMap::new();
        for (name, var) in source.data().lock().unwrap().iter() {
            // Avoid dependence on the unseedable CPU initializer and hash-map order.
            let seed = name
                .bytes()
                .fold(17u32, |s, b| s.wrapping_mul(31).wrapping_add(b as u32));
            let values = (0..var.elem_count())
                .map(|i| {
                    if var.rank() != 2 {
                        return 1f32;
                    }
                    let n = seed.wrapping_add((i as u32).wrapping_mul(747796405));
                    let n = (n ^ (n >> 16)).wrapping_mul(2246822519);
                    ((n ^ (n >> 13)) % 401) as f32 / 4000. - 0.05
                })
                .collect::<Vec<_>>();
            var.set(&Tensor::from_vec(values, var.shape(), &Device::Cpu)?)?;
            let tensor = var.detach().to_dtype(dtype)?.to_device(device)?;
            assert!(!tensor.track_op());
            frozen.insert(name.clone(), tensor);
        }
        let mut base = TextModel::new(
            &config,
            VarBuilder::from_tensors(frozen.clone(), dtype, device),
        )?;
        assert_eq!(base.quantize_frozen_decoder(quant)?, 14);
        let vars = VarMap::new();
        let adapter = HeadAdapter::new(
            64,
            17,
            2,
            3.,
            config.final_logit_softcapping,
            VarBuilder::from_varmap(&vars, DType::F32, device),
        )?;
        let a = vars.data().lock().unwrap()["lora_a"].clone();
        let values = (0..a.elem_count())
            .map(|i| ((i * 37 % 101) as f32 - 50.) / 1000.)
            .collect::<Vec<_>>();
        a.set(&Tensor::from_vec(values, a.shape(), device)?)?;
        Ok((
            TrainingModel {
                base,
                adapter: Some(adapter),
            },
            vars,
            frozen,
        ))
    }

    fn values(tensor: &Tensor) -> Result<Vec<f32>> {
        tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        for (a, e) in values(actual)?.into_iter().zip(values(expected)?) {
            assert!(a.is_finite() && e.is_finite());
            assert!((a - e).abs() <= tolerance, "actual={a}, expected={e}");
        }
        Ok(())
    }

    fn trains_only_adapter(device: &Device) -> Result<()> {
        let input = Tensor::new(&[[1u32, 7, 3], [4, 6, 5]], device)?;
        let targets = Tensor::new(&[7u32, 3, 10, 6, 5, 9], device)?;
        for dtype in [DType::BF16, DType::F32] {
            for quant in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
                let (mut model, vars, frozen) = fixture(device, dtype, quant)?;
                let snapshots = frozen
                    .iter()
                    .map(|(name, tensor)| Ok((name.clone(), values(tensor)?)))
                    .collect::<Result<HashMap<_, _>>>()?;
                let a = vars.data().lock().unwrap()["lora_a"].clone();
                let b = vars.data().lock().unwrap()["lora_b"].clone();
                assert_eq!(vars.all_vars().len(), 2);
                assert!(values(&a)?.iter().any(|&v| v != 0.));
                assert!(values(&b)?.iter().all(|&v| v == 0.));
                let hidden = model.base.forward_train_hidden(&input)?;
                assert_eq!(hidden.dtype(), dtype);
                assert!(!hidden.track_op());
                let raw = model.base.project_logits(&hidden)?.to_dtype(DType::F32)?;
                let initial = model.forward_train(&input)?;
                assert_close(&initial, &((&raw / 0.5)?.tanh()? * 0.5)?, 1e-6)?;
                let mut previous = values(&initial)?;
                let mut optimizer = AdamW::new(
                    vars.all_vars(),
                    ParamsAdamW {
                        lr: 0.02,
                        // Isolate gradient-driven updates, especially the initially zero A gradient.
                        weight_decay: 0.,
                        ..Default::default()
                    },
                )?;
                for step in 0..2 {
                    let before_a = values(&a)?;
                    let before_b = values(&b)?;
                    let logits = model.forward_train(&input)?;
                    assert_eq!(logits.dtype(), DType::F32);
                    assert_eq!(logits.dims(), &[2, 3, 17]);
                    let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &targets)?;
                    assert_eq!(loss.dtype(), DType::F32);
                    assert!(loss.to_scalar::<f32>()?.is_finite());
                    let grads = loss.backward()?;
                    for var in [&a, &b] {
                        assert_eq!(var.dtype(), DType::F32);
                        let grad = grads.get(var).expect("missing adapter gradient");
                        assert_eq!(grad.dtype(), DType::F32);
                        assert!(values(grad)?.iter().all(|v| v.is_finite()));
                    }
                    let a_grad = values(grads.get(&a).unwrap())?;
                    assert_eq!(a_grad.iter().all(|&v| v == 0.), step == 0);
                    assert!(values(grads.get(&b).unwrap())?.iter().any(|&v| v != 0.));
                    for tensor in frozen.values() {
                        assert!(grads.get(tensor).is_none());
                    }
                    optimizer.step(&grads)?;
                    assert_eq!(values(&a)? == before_a, step == 0);
                    assert_ne!(values(&b)?, before_b);
                    for var in [&a, &b] {
                        assert_eq!(var.dtype(), DType::F32);
                        assert!(values(var)?.iter().all(|v| v.is_finite()));
                    }

                    // Reconstruct from current masters, not casts retained by a prior forward.
                    let delta = (hidden
                        .flatten_to(1)?
                        .matmul(&a.to_dtype(dtype)?.t()?)?
                        .matmul(&b.to_dtype(dtype)?.t()?)?
                        .to_dtype(DType::F32)?
                        * 1.5)?
                        .reshape((2, 3, 17))?;
                    let uncapped = (&raw + &delta)?;
                    let expected = ((&uncapped / 0.5)?.tanh()? * 0.5)?;
                    let updated = model.forward_train(&input)?;
                    assert_eq!(updated.dtype(), DType::F32);
                    assert_close(&updated, &expected, 1e-6)?;
                    assert_ne!(values(&updated)?, previous);
                    previous = values(&updated)?;
                    let wrong_order = (((&raw / 0.5)?.tanh()? * 0.5)? + &delta)?;
                    assert!(values(&updated)?
                        .iter()
                        .zip(values(&wrong_order)?)
                        .any(|(a, b)| (a - b).abs() > 1e-4));
                    model.adapter.as_mut().unwrap().softcap = None;
                    assert_close(&model.forward_train(&input)?, &uncapped, 1e-6)?;
                    model.adapter.as_mut().unwrap().softcap = Some(0.5);
                }
                assert_close(&model.base.forward_train_hidden(&input)?, &hidden, 0.)?;
                assert_close(
                    &model.base.project_logits(&hidden)?.to_dtype(DType::F32)?,
                    &raw,
                    0.,
                )?;
                for (name, tensor) in &frozen {
                    assert_eq!(values(tensor)?, snapshots[name], "base changed: {name}");
                }
            }
        }
        Ok(())
    }

    fn numerical_gradients(device: &Device) -> Result<()> {
        let input = Tensor::new(&[[1u32, 7, 3]], device)?;
        for quant in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
            let (mut model, vars, _) = fixture(device, DType::F32, quant)?;
            let b = vars.data().lock().unwrap()["lora_b"].clone();
            b.set(&Tensor::from_vec(
                (0..b.elem_count())
                    .map(|i| (i as f32 - 16.) / 100.)
                    .collect::<Vec<_>>(),
                b.shape(),
                device,
            )?)?;
            let loss = model.forward_train(&input)?.sqr()?.mean_all()?;
            let grads = loss.backward()?;
            for name in ["lora_a", "lora_b"] {
                let var = vars.data().lock().unwrap()[name].clone();
                let gradient = values(grads.get(&var).expect("missing gradient"))?;
                assert!(gradient.iter().all(|v| v.is_finite()));
                // Probe a nontrivial coordinate of each matrix with only four extra forwards.
                let (index, &analytic) = gradient
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                    .unwrap();
                assert!(analytic.abs() > 1e-5, "vacuous gradient check for {name}");
                let original = values(&var)?;
                let epsilon = 0.002f32;
                let mut losses = Vec::new();
                for sign in [-1., 1.] {
                    let mut perturbed = original.clone();
                    perturbed[index] += sign * epsilon;
                    var.set(&Tensor::from_vec(perturbed, var.shape(), device)?)?;
                    losses.push(
                        model
                            .forward_train(&input)?
                            .sqr()?
                            .mean_all()?
                            .to_scalar::<f32>()?,
                    );
                }
                var.set(&Tensor::from_vec(original, var.shape(), device)?)?;
                let numerical = (losses[1] - losses[0]) / (2. * epsilon);
                assert!(
                    (analytic - numerical).abs() < 2e-5 + analytic.abs() * 0.02,
                    "{quant:?} {name}[{index}]: analytic={analytic}, numerical={numerical}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn quantized_adapter_training_cpu() -> Result<()> {
        trains_only_adapter(&Device::Cpu)
    }

    #[test]
    fn adapter_numerical_gradients_cpu() -> Result<()> {
        numerical_gradients(&Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn quantized_adapter_training_cuda() -> Result<()> {
        trains_only_adapter(&Device::new_cuda(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn adapter_numerical_gradients_cuda() -> Result<()> {
        numerical_gradients(&Device::new_cuda(0)?)
    }
}
