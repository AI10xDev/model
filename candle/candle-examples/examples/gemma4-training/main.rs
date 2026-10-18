#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod adapter;
mod dataset;

use adapter::{HeadAdapter, TrainingModel};
use anyhow::{ensure, Context, Result};
use candle::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_transformers::models::gemma4::{config::Gemma4TextConfig, text::TextModel};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

#[derive(Parser)]
#[command(about = "Gemma4 full F32 training or frozen Q4/Q8 decoder LoRA training")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Preprocess(dataset::PrepareArgs),
    Train(TrainArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum Mode {
    Full,
    QuantizedHead,
    QuantizedLora,
}

impl Mode {
    fn is_quantized(self) -> bool {
        self != Self::Full
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::QuantizedHead => "quantized-head",
            Self::QuantizedLora => "quantized-lora",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum Precision {
    F32,
    Bf16,
}

impl Precision {
    fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 => DType::BF16,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Quantization {
    Q4_0,
    Q8_0,
}

impl Quantization {
    fn dtype(self) -> candle::quantized::GgmlDType {
        match self {
            Self::Q4_0 => candle::quantized::GgmlDType::Q4_0,
            Self::Q8_0 => candle::quantized::GgmlDType::Q8_0,
        }
    }
}

#[derive(Args)]
struct TrainArgs {
    #[arg(long, value_enum, default_value = "full")]
    mode: Mode,
    /// Compute dtype; adapter master weights, AdamW moments and loss remain F32.
    #[arg(long, value_enum, default_value = "f32")]
    precision: Precision,
    /// Frozen attention/MLP weights only; not QAT or full-parameter training.
    #[arg(long, value_enum, default_value = "q4-0")]
    quantization: Quantization,
    #[arg(long, default_value_t = 8)]
    lora_rank: usize,
    #[arg(long, default_value_t = 8.0)]
    lora_alpha: f64,
    /// Warm-start adapters; still requires the original --checkpoint and mode.
    #[arg(long, requires = "checkpoint")]
    adapter_checkpoint: Option<PathBuf>,
    #[arg(long)]
    config: PathBuf,
    /// Directory produced by the preprocess subcommand.
    #[arg(long)]
    dataset: PathBuf,
    /// Fresh run directory; existing files are never overwritten.
    #[arg(long)]
    output: PathBuf,
    /// Initialize random weights instead of loading a checkpoint.
    #[arg(
        long,
        conflicts_with = "checkpoint",
        required_unless_present = "checkpoint"
    )]
    from_scratch: bool,
    /// Local checkpoint directory. Reload is a warm start, not an optimizer resume.
    #[arg(
        long,
        conflicts_with = "from_scratch",
        required_unless_present = "from_scratch"
    )]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value_t = 100)]
    steps: usize,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    /// Microbatches per optimizer update; graphs are released between backwards.
    #[arg(long, default_value_t = 1)]
    gradient_accumulation_steps: usize,
    #[arg(long, default_value_t = 0.0001)]
    learning_rate: f64,
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,
    #[arg(long, default_value_t = 1.0)]
    max_grad_norm: f64,
    #[arg(long, default_value_t = 10)]
    eval_every: usize,
    #[arg(long, default_value_t = 10)]
    eval_batches: usize,
    #[arg(long, default_value_t = 100)]
    save_every: usize,
    /// Seeds data order only; Candle CPU random initialization is not seedable.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Safety cap, not a memory guarantee. F32 AdamW requires >=16 bytes/parameter.
    #[arg(long, default_value_t = 100_000_000)]
    max_parameters: usize,
    /// CPU is the default. Mutually exclusive with --cuda.
    #[arg(long, conflicts_with = "cuda")]
    cpu: bool,
    /// Explicitly require CUDA device 0; never silently fall back to CPU.
    #[arg(long, conflicts_with = "cpu")]
    cuda: bool,
}

#[derive(Deserialize)]
struct DatasetMetadata {
    format_version: usize,
    token_format: String,
    sequence_length: usize,
    window_length: usize,
    stride: usize,
    vocab_size: usize,
    train: SplitMetadata,
    valid: SplitMetadata,
}

#[derive(Deserialize)]
struct SplitMetadata {
    stored_tokens: usize,
    windows: usize,
}

fn initializer_range(raw: &Value) -> Result<f32> {
    let text = raw.get("text_config").unwrap_or(raw);
    let std = match text.get("initializer_range") {
        None => 0.02,
        Some(value) => value
            .as_f64()
            .context("initializer_range must be numeric, finite and positive")?,
    } as f32;
    ensure!(
        std.is_finite() && std > 0.0,
        "initializer_range must be finite and positive in F32"
    );
    Ok(std)
}

fn parse_config(raw: &Value) -> Result<(Gemma4TextConfig, bool)> {
    initializer_range(raw)?;
    let text = raw.get("text_config").unwrap_or(raw);
    let dropout = match text.get("attention_dropout") {
        None => 0.0,
        Some(value) => value.as_f64().context(
            "attention_dropout must be numeric and zero; attention dropout is unsupported",
        )?,
    };
    ensure!(
        dropout.is_finite() && dropout == 0.0,
        "attention_dropout must be finite and zero; nonzero attention dropout is unsupported"
    );
    let conditional = raw.get("text_config").is_some();
    let expected = if conditional { "gemma4" } else { "gemma4_text" };
    if let Some(kind) = raw.get("model_type") {
        ensure!(
            kind.as_str() == Some(expected),
            "expected {expected} config"
        );
    }
    let mut config: Gemma4TextConfig =
        serde_json::from_value(raw.get("text_config").unwrap_or(raw).clone())?;
    config.use_flash_attn = false;
    config.validate()?;
    Ok((config, conditional))
}

fn initialize_weights(vars: &VarMap, std: f32) -> Result<()> {
    for (name, var) in vars.data().lock().unwrap().iter() {
        if var.rank() == 2 {
            var.set(&Tensor::randn(0f32, std, var.shape(), var.device())?)?;
        } else if name.ends_with(".bias") {
            var.set(&var.zeros_like()?)?;
        }
        // Gemma4 norm weights and layer_scalar retain their initial ones.
    }
    Ok(())
}

fn load_split(
    path: &Path,
    meta: &SplitMetadata,
    sequence_length: usize,
    vocab_size: usize,
) -> Result<Vec<Vec<u32>>> {
    ensure!(
        meta.windows > 0,
        "{}: metadata has no windows",
        path.display()
    );
    let tokens = meta
        .windows
        .checked_mul(sequence_length)
        .and_then(|n| n.checked_add(1))
        .context("dataset metadata token count overflow")?;
    ensure!(
        meta.stored_tokens == tokens,
        "{}: metadata stored_tokens must equal windows * sequence_length + 1",
        path.display()
    );
    let bytes = tokens
        .checked_mul(4)
        .context("dataset metadata byte count overflow")?;
    ensure!(
        fs::metadata(path)?.len() == u64::try_from(bytes)?,
        "{}: byte length differs from dataset metadata (expected {bytes} bytes)",
        path.display()
    );
    let data = dataset::load(path, sequence_length, vocab_size)?;
    ensure!(
        data.len() == meta.windows,
        "{}: loaded window count differs from dataset metadata",
        path.display()
    );
    Ok(data)
}

// Count the supported dense model before allocating weights or optimizer state.
fn parameter_count(c: &Gemma4TextConfig) -> f64 {
    let h = c.hidden_size as f64;
    let mut n = c.vocab_size as f64 * h + h;
    if !c.tie_word_embeddings {
        n += c.vocab_size as f64 * h;
    }
    for i in 0..c.num_hidden_layers {
        let local = c.is_sliding(i);
        let shared = c.attention_k_eq_v && !local;
        let d = if local { c.head_dim } else { c.global_head_dim } as f64;
        let kv = if shared {
            c.num_global_key_value_heads
                .unwrap_or(c.num_key_value_heads)
        } else {
            c.num_key_value_heads
        } as f64;
        let q = c.num_attention_heads as f64 * d;
        let k = kv * d * if shared { 1.0 } else { 2.0 };
        n += h * (2.0 * q + k) + 2.0 * d;
        if c.attention_bias {
            n += q + k + h;
        }
        n += 3.0 * h * c.intermediate_size as f64 + 4.0 * h + 1.0;
    }
    n
}

fn checkpoint_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let index = directory.join("model.safetensors.index.json");
    if !index.try_exists()? {
        return Ok(vec![directory.join("model.safetensors")]);
    }
    let index: Value = serde_json::from_slice(&fs::read(index)?)?;
    let map = index["weight_map"]
        .as_object()
        .context("missing weight_map")?;
    let mut files = BTreeSet::new();
    for value in map.values() {
        let name = value.as_str().context("shard filename must be a string")?;
        let mut components = Path::new(name).components();
        ensure!(
            matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
            "shard filename must be a plain local filename: {name}"
        );
        files.insert(directory.join(name));
    }
    ensure!(!files.is_empty(), "checkpoint index has no shards");
    Ok(files.into_iter().collect())
}

fn mmap_checkpoint(directory: &Path) -> Result<candle::safetensors::MmapedSafetensors> {
    let files = checkpoint_files(directory)?;
    // Checkpoints must not be modified while mapped.
    let weights = unsafe { candle::safetensors::MmapedSafetensors::multi(&files)? };
    let mut seen = BTreeSet::new();
    for (name, _) in weights.tensors() {
        ensure!(
            seen.insert(name.clone()),
            "duplicate checkpoint tensor {name}"
        );
    }
    Ok(weights)
}

fn load_weights(vars: &VarMap, directory: &Path) -> Result<()> {
    let weights = mmap_checkpoint(directory)?;
    for (name, var) in vars.data().lock().unwrap().iter() {
        let value = weights.load(name, var.device())?.to_dtype(var.dtype())?;
        var.set(&value).with_context(|| format!("loading {name}"))?;
    }
    Ok(())
}

fn batch_loss(model: &mut TrainingModel, windows: &[&Vec<u32>], device: &Device) -> Result<Tensor> {
    let length = windows[0].len() - 1;
    let input: Vec<u32> = windows
        .iter()
        .flat_map(|w| w[..length].iter().copied())
        .collect();
    let target: Vec<u32> = windows
        .iter()
        .flat_map(|w| w[1..].iter().copied())
        .collect();
    let input = Tensor::from_vec(input, (windows.len(), length), device)?;
    let target = Tensor::from_vec(target, windows.len() * length, device)?;
    let logits = model
        .forward_train(&input)?
        .flatten_to(1)?
        .to_dtype(DType::F32)?;
    Ok(candle_nn::loss::cross_entropy(&logits, &target)?)
}

fn evaluate(
    model: &mut TrainingModel,
    data: &[Vec<u32>],
    args: &TrainArgs,
    device: &Device,
) -> Result<f64> {
    let mut sum = 0.0;
    let mut count = 0;
    // Candle tracks this graph, but each batch is dropped without backward.
    for chunk in data.chunks(args.batch_size).take(args.eval_batches) {
        let loss = batch_loss(model, &chunk.iter().collect::<Vec<_>>(), device)?
            .to_scalar::<f32>()? as f64;
        ensure!(loss.is_finite(), "non-finite validation loss");
        sum += loss * chunk.len() as f64;
        count += chunk.len();
    }
    ensure!(count > 0, "no validation batches");
    Ok(sum / count as f64)
}

fn clip_gradients(
    grads: &mut candle::backprop::GradStore,
    vars: &[Var],
    max_norm: f64,
) -> Result<f64> {
    let mut squared = 0.0;
    for var in vars {
        let grad = grads
            .get(var)
            .context("missing trainable parameter gradient")?;
        squared += grad
            .to_dtype(DType::F32)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()? as f64;
    }
    let norm = squared.sqrt();
    ensure!(
        norm.is_finite(),
        "non-finite gradients; no optimizer update performed"
    );
    if norm > max_norm {
        for var in vars {
            let clipped = (grads.get(var).unwrap() * (max_norm / norm))?;
            grads.insert(var, clipped);
        }
    }
    Ok(norm)
}

fn save_checkpoint(vars: &VarMap, args: &TrainArgs, step: usize, state: &Value) -> Result<()> {
    let path = args.output.join(format!("checkpoint-{step}"));
    let pending = args.output.join(format!(".checkpoint-{step}.partial"));
    ensure!(
        !path.try_exists()?,
        "checkpoint already exists: {}",
        path.display()
    );
    fs::create_dir(&pending)?;
    let filename = if args.mode.is_quantized() {
        "adapter.safetensors"
    } else {
        "model.safetensors"
    };
    vars.save(pending.join(filename))?;
    for name in [
        "config.json",
        "source-config.json",
        "tokenizer.json",
        "dataset-metadata.json",
    ] {
        fs::copy(args.output.join(name), pending.join(name))?;
    }
    fs::write(
        pending.join("training-state.json"),
        serde_json::to_vec_pretty(state)?,
    )?;
    fs::rename(pending, path)?;
    Ok(())
}

fn train(args: TrainArgs) -> Result<()> {
    ensure!(
        args.mode.is_quantized() || args.precision == Precision::F32,
        "full training supports F32 only; use --mode quantized-lora or quantized-head for BF16/F32 mixed precision"
    );
    ensure!(
        args.mode.is_quantized() || args.adapter_checkpoint.is_none(),
        "--adapter-checkpoint requires --mode quantized-lora or quantized-head"
    );
    if args.mode.is_quantized() {
        ensure!(
            args.checkpoint.is_some(),
            "quantized training requires a frozen pretrained --checkpoint, not --from-scratch"
        );
        ensure!(
            args.lora_rank > 0 && args.lora_alpha.is_finite() && args.lora_alpha > 0.,
            "lora-rank and lora-alpha must be positive and finite"
        );
    }
    for (name, value) in [
        ("steps", args.steps),
        ("batch-size", args.batch_size),
        (
            "gradient-accumulation-steps",
            args.gradient_accumulation_steps,
        ),
        ("eval-every", args.eval_every),
        ("eval-batches", args.eval_batches),
        ("save-every", args.save_every),
        ("max-parameters", args.max_parameters),
    ] {
        ensure!(value > 0, "{name} must be positive");
    }
    ensure!(
        args.learning_rate.is_finite() && args.learning_rate > 0.0,
        "learning-rate must be finite and positive"
    );
    ensure!(
        args.weight_decay.is_finite() && args.weight_decay >= 0.0,
        "weight-decay must be finite and nonnegative"
    );
    ensure!(
        args.max_grad_norm.is_finite() && args.max_grad_norm > 0.0,
        "max-grad-norm must be finite and positive"
    );
    ensure!(
        !args.output.try_exists()?,
        "output directory already exists"
    );
    let source_config = fs::read(&args.config)?;
    let mut raw: Value = serde_json::from_slice(&source_config)?;
    let (mut config, conditional) = parse_config(&raw)?;
    if args.mode == Mode::QuantizedHead {
        ensure!(
            args.lora_rank <= config.hidden_size.min(config.vocab_size),
            "lora-rank must not exceed hidden size or vocabulary size"
        );
    }
    let std = initializer_range(&raw)?;
    let parameters = parameter_count(&config);
    ensure!(parameters <= args.max_parameters as f64,
        "model has approximately {parameters:.0} parameters, above --max-parameters {}; full F32 AdamW needs at least {:.1} GiB for weights/gradients/moments alone. Quantized modes load and quantize one decoder layer at a time",
        args.max_parameters, parameters * 16.0 / 1073741824.0);
    let metadata_bytes = fs::read(args.dataset.join("metadata.json"))?;
    let meta: DatasetMetadata = serde_json::from_slice(&metadata_bytes)?;
    ensure!(
        meta.format_version == 1 && meta.token_format == "u32-le",
        "unsupported dataset format"
    );
    ensure!(
        meta.sequence_length > 0 && meta.sequence_length <= config.max_position_embeddings,
        "dataset sequence length must be positive and within model context"
    );
    ensure!(
        meta.window_length
            == meta
                .sequence_length
                .checked_add(1)
                .context("window length overflow")?
            && meta.stride == meta.sequence_length,
        "inconsistent dataset window/stride metadata"
    );
    ensure!(
        meta.vocab_size == config.vocab_size,
        "dataset/model vocabulary size mismatch"
    );
    let tokenizer_bytes = fs::read(args.dataset.join("tokenizer.json"))?;
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).map_err(anyhow::Error::msg)?;
    ensure!(
        tokenizer
            .get_vocab(true)
            .values()
            .all(|&id| (id as usize) < config.vocab_size),
        "tokenizer IDs exceed model vocabulary"
    );
    if let Some(checkpoint) = &args.checkpoint {
        ensure!(fs::read(checkpoint.join("tokenizer.json"))? == tokenizer_bytes,
            "checkpoint tokenizer differs from preprocessing tokenizer; reprocess using the checkpoint tokenizer");
        let previous: Value = serde_json::from_slice(&fs::read(checkpoint.join("config.json"))?)?;
        let (mut previous, previous_conditional) = parse_config(&previous)?;
        previous.max_position_embeddings = config.max_position_embeddings;
        ensure!(
            previous == config && previous_conditional == conditional,
            "checkpoint architecture/config differs from --config (only context length may differ)"
        );
    }
    let train_data = load_split(
        &args.dataset.join("train.bin"),
        &meta.train,
        meta.sequence_length,
        config.vocab_size,
    )?;
    let valid_data = load_split(
        &args.dataset.join("valid.bin"),
        &meta.valid,
        meta.sequence_length,
        config.vocab_size,
    )?;
    // Bound precomputed RoPE tables to the training context without altering frequencies.
    config.max_position_embeddings = meta.sequence_length;
    let text = if conditional {
        &mut raw["text_config"]
    } else {
        &mut raw
    };
    text["max_position_embeddings"] = json!(meta.sequence_length);
    text["use_flash_attn"] = json!(false);
    let device = if args.cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let mut vars = VarMap::new();
    let dtype = args.precision.dtype();
    let vb = if args.mode.is_quantized() {
        let weights = mmap_checkpoint(args.checkpoint.as_ref().unwrap())?;
        VarBuilder::from_backend(Box::new(weights), dtype, device.clone())
    } else {
        VarBuilder::from_varmap(&vars, DType::F32, &device)
    };
    if args.mode.is_quantized() {
        println!(
            "loading frozen decoder layer-by-layer with {:?} weights",
            args.quantization.dtype()
        );
    }
    let mut model = match (args.mode.is_quantized(), conditional) {
        (true, true) => TextModel::new_quantized_for_conditional_generation(
            &config,
            vb,
            args.quantization.dtype(),
        )?,
        (true, false) => TextModel::new_quantized(&config, vb, args.quantization.dtype())?,
        (false, true) => TextModel::new_for_conditional_generation(&config, vb)?,
        (false, false) => TextModel::new(&config, vb)?,
    };
    let mut quantized_projections = 0;
    let adapter = if args.mode.is_quantized() {
        quantized_projections = (0..config.num_hidden_layers)
            .map(|i| {
                if config.attention_k_eq_v && !config.is_sliding(i) {
                    6
                } else {
                    7
                }
            })
            .sum();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
        let adapter = if args.mode == Mode::QuantizedLora {
            let targets = model.add_quantized_lora(args.lora_rank, args.lora_alpha, vb)?;
            ensure!(
                targets == quantized_projections,
                "adapter target count mismatch"
            );
            None
        } else {
            Some(HeadAdapter::new(
                config.hidden_size,
                config.vocab_size,
                args.lora_rank,
                args.lora_alpha,
                config.final_logit_softcapping,
                vb,
            )?)
        };
        if let Some(path) = &args.adapter_checkpoint {
            let state: Value =
                serde_json::from_slice(&fs::read(path.join("training-state.json"))?)?;
            ensure!(
                state["mode"] == args.mode.as_str()
                    && state["lora_rank"] == args.lora_rank
                    && state["lora_alpha"] == args.lora_alpha
                    && state["quantization"] == format!("{:?}", args.quantization.dtype())
                    && state["dtype"] == dtype.as_str(),
                "adapter training policy mismatch"
            );
            let source = state["source_checkpoint"]
                .as_str()
                .context("adapter missing base checkpoint")?;
            ensure!(
                fs::canonicalize(source)? == fs::canonicalize(args.checkpoint.as_ref().unwrap())?,
                "adapter requires its original base checkpoint"
            );
            let previous: Value = serde_json::from_slice(&fs::read(path.join("config.json"))?)?;
            let (mut previous, previous_conditional) = parse_config(&previous)?;
            previous.max_position_embeddings = config.max_position_embeddings;
            ensure!(
                previous == config && previous_conditional == conditional,
                "adapter architecture mismatch"
            );
            ensure!(
                fs::read(path.join("tokenizer.json"))? == tokenizer_bytes,
                "adapter tokenizer mismatch"
            );
            vars.load(path.join("adapter.safetensors"))?;
        }
        adapter
    } else {
        if let Some(checkpoint) = &args.checkpoint {
            load_weights(&vars, checkpoint)?;
        } else {
            initialize_weights(&vars, std)?;
        }
        None
    };
    let mut model = TrainingModel {
        base: model,
        adapter,
    };
    // Upstream layer_scalar is a buffer: preserve it in checkpoints but do not optimize it.
    let mut named_vars: Vec<_> = vars
        .data()
        .lock()
        .unwrap()
        .iter()
        .filter(|(name, _)| !name.ends_with(".layer_scalar"))
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect();
    named_vars.sort_by(|a, b| a.0.cmp(&b.0));
    let trainable: Vec<_> = named_vars.into_iter().map(|(_, var)| var).collect();
    let actual_parameters: usize = if args.mode.is_quantized() {
        parameters as usize
    } else {
        vars.all_vars().iter().map(|v| v.elem_count()).sum()
    };
    ensure!(
        actual_parameters <= args.max_parameters,
        "actual parameter count exceeds safety cap"
    );
    let mut optimizer = AdamW::new(
        trainable.clone(),
        ParamsAdamW {
            lr: args.learning_rate,
            weight_decay: args.weight_decay,
            ..Default::default()
        },
    )?;
    fs::create_dir(&args.output)?;
    fs::write(args.output.join("source-config.json"), source_config)?;
    fs::write(
        args.output.join("config.json"),
        serde_json::to_vec_pretty(&raw)?,
    )?;
    fs::write(args.output.join("tokenizer.json"), tokenizer_bytes)?;
    fs::write(args.output.join("dataset-metadata.json"), metadata_bytes)?;
    let mut state = json!({
        "format_version": 1, "objective": "all-token causal language modeling",
        "mode": args.mode.as_str(),
        "dtype": dtype.as_str(), "master_dtype": "f32", "optimizer_dtype": "f32", "loss_dtype": "f32",
        "optimizer": "AdamW", "optimizer_state_saved": false,
        "quantization": args.mode.is_quantized().then(|| format!("{:?}", args.quantization.dtype())),
        "quantized_projections": quantized_projections,
        "lora_rank": args.mode.is_quantized().then_some(args.lora_rank),
        "lora_alpha": args.mode.is_quantized().then_some(args.lora_alpha),
        "adapter_target": match args.mode {
            Mode::Full => None,
            Mode::QuantizedHead => Some("lm_head only, additive before softcap; tied base embeddings unchanged"),
            Mode::QuantizedLora => Some("all decoder attention q/k/v/o and MLP gate/up/down projections; absent shared-KV v_proj excluded"),
        },
        "trainable_parameters": trainable.iter().map(|v| v.elem_count()).sum::<usize>(),
        "adapter_checkpoint": args.adapter_checkpoint,
        "adapter_initialization": args.mode.is_quantized().then_some("A: normal(0, 0.02); B: zero; RNG not seeded by --seed"),
        "quantized_matmul_input_dtype": args.mode.is_quantized().then_some("f32 for Candle CUDA fallback compatibility; output cast to compute dtype"),
        "reload": "warm start only; optimizer and data-order restart",
        "text_only": true, "frozen_parameters": if args.mode.is_quantized() { "entire base model" } else { "*.layer_scalar" },
        "parameters": actual_parameters, "device": format!("{device:?}"),
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "batch_size": args.batch_size,
        "seed": args.seed, "seed_scope": "data order only, not scratch initialization",
        "initializer_range": args.checkpoint.is_none().then_some(std),
        "initialization_policy": if args.checkpoint.is_some() { "checkpoint warm start" }
            else { "matrices: normal(0, initializer_range); biases: zeros; norms and layer_scalar: ones; CPU RNG unseeded" },
        "sequence_length": meta.sequence_length, "train_windows": train_data.len(),
        "valid_windows": valid_data.len(), "eval_batches": args.eval_batches,
        "source_checkpoint": args.checkpoint.as_ref().map(fs::canonicalize).transpose()?, "steps_requested": args.steps,
        "step": 0, "tokens_seen": 0
    });
    state["gradient_accumulation_steps"] = json!(args.gradient_accumulation_steps);
    state["quantized_loading"] = json!(args.mode.is_quantized().then_some(
        "one dense decoder layer at a time, immediately quantized; no dense base copy retained"
    ));
    state["quantized_backward"] = json!((args.mode == Mode::QuantizedLora).then_some("first-order input gradients; one transient F32 dequantized projection; frozen base weights"));
    fs::write(
        args.output.join("run-config.json"),
        serde_json::to_vec_pretty(&state)?,
    )?;
    let mut metrics = File::create(args.output.join("metrics.jsonl"))?;
    let initial_loss = evaluate(&mut model, &valid_data, &args, &device)?;
    println!("parameters={actual_parameters} initial_validation_loss={initial_loss:.6}");
    writeln!(
        metrics,
        "{}",
        json!({"step": 0, "validation_loss": initial_loss})
    )?;
    let mut order: Vec<_> = (0..train_data.len()).collect();
    let mut rng = StdRng::seed_from_u64(args.seed);
    order.shuffle(&mut rng);
    let mut cursor = 0;
    let mut tokens_seen = 0u64;
    for step in 1..=args.steps {
        let mut accumulated = candle::backprop::GradStore::default();
        let mut loss_sum = 0.0f64;
        let mut windows_seen = 0usize;
        for _ in 0..args.gradient_accumulation_steps {
            if cursor == order.len() {
                order.shuffle(&mut rng);
                cursor = 0;
            }
            let end = cursor.saturating_add(args.batch_size).min(order.len());
            let batch: Vec<_> = order[cursor..end].iter().map(|&i| &train_data[i]).collect();
            let loss = batch_loss(&mut model, &batch, &device)?;
            let loss_value = loss.to_scalar::<f32>()?;
            ensure!(loss_value.is_finite(), "non-finite loss at step {step}");
            let grads = loss.backward()?;
            // Weight by actual window count, including short batches at epoch ends.
            for var in &trainable {
                let grad = (grads
                    .get(var)
                    .context("missing trainable parameter gradient")?
                    * batch.len() as f64)?;
                let sum = match accumulated.get(var) {
                    None => grad,
                    Some(previous) => (previous + grad)?,
                };
                accumulated.insert(var, sum.detach());
            }
            loss_sum += loss_value as f64 * batch.len() as f64;
            windows_seen += batch.len();
            cursor = end;
            // Only adapter gradients survive; release this microbatch's graph.
        }
        for var in &trainable {
            let mean = (accumulated.get(var).unwrap() / windows_seen as f64)?;
            accumulated.insert(var, mean);
        }
        let norm = clip_gradients(&mut accumulated, &trainable, args.max_grad_norm)?;
        optimizer.step(&accumulated)?;
        drop(accumulated);
        tokens_seen += windows_seen as u64 * meta.sequence_length as u64;
        let loss_value = loss_sum / windows_seen as f64;
        let mut metric = json!({"step": step, "train_loss": loss_value,
            "gradient_norm": norm, "tokens_seen": tokens_seen});
        if step % args.eval_every == 0 || step == args.steps {
            let valid = evaluate(&mut model, &valid_data, &args, &device)?;
            metric["validation_loss"] = json!(valid);
            state["validation_loss"] = json!(valid);
            state["validation_step"] = json!(step);
        }
        println!("{metric}");
        writeln!(metrics, "{metric}")?;
        metrics.flush()?;
        state["step"] = json!(step);
        state["tokens_seen"] = json!(tokens_seen);
        state["train_loss"] = json!(loss_value);
        if step % args.save_every == 0 || step == args.steps {
            save_checkpoint(&vars, &args, step, &state)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Preprocess(args) => dataset::prepare(args),
        Command::Train(args) => train(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const CONFIG: &str = include_str!("fixtures/config.json");
    const TOKENIZER: &str = include_str!("fixtures/tokenizer.json");
    const DOCUMENTS: &str = include_str!("fixtures/train.jsonl");

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Result<Self> {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "gemma4-training-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }

        fn args(&self, output: &str) -> TrainArgs {
            TrainArgs {
                mode: Mode::Full,
                precision: Precision::F32,
                quantization: Quantization::Q4_0,
                lora_rank: 8,
                lora_alpha: 8.,
                adapter_checkpoint: None,
                config: self.0.join("config.json"),
                dataset: self.0.join("prepared"),
                output: self.0.join(output),
                from_scratch: true,
                checkpoint: None,
                steps: 3,
                batch_size: 1,
                gradient_accumulation_steps: 1,
                learning_rate: 0.001,
                weight_decay: 0.1,
                max_grad_norm: 1.0,
                eval_every: 2,
                eval_batches: 2,
                save_every: 1,
                seed: 42,
                max_parameters: 100_000,
                cpu: true,
                cuda: false,
            }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn read_json(path: impl AsRef<Path>) -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    fn metrics(path: &Path) -> Result<Vec<Value>> {
        fs::read_to_string(path.join("metrics.jsonl"))?
            .lines()
            .map(|line| Ok(serde_json::from_str(line)?))
            .collect()
    }

    #[test]
    fn preprocess_train_and_warm_start_cpu() -> Result<()> {
        let dir = TempDir::new()?;
        fs::write(dir.0.join("config.json"), CONFIG)?;
        fs::write(dir.0.join("tokenizer.json"), TOKENIZER)?;
        fs::write(dir.0.join("input.jsonl"), DOCUMENTS)?;
        dataset::prepare(dataset::PrepareArgs {
            input: dir.0.join("input.jsonl"),
            tokenizer: dir.0.join("tokenizer.json"),
            output: dir.0.join("prepared"),
            sequence_length: 16,
            vocab_size: 16,
            validation_fraction: 0.75,
            seed: 42,
            bos_token_id: Some(1),
            eos_token_id: 2,
        })?;
        let metadata = read_json(dir.0.join("prepared/metadata.json"))?;
        let windows = dataset::load(&dir.0.join("prepared/train.bin"), 16, 16)?;
        assert_eq!(metadata["train"]["windows"], windows.len());
        assert!(windows.len() > 2);
        let mut args = dir.args("scratch");
        // Force a short final batch, then an epoch rollover in just three steps.
        let batch_size = windows.len() - 1;
        args.batch_size = batch_size;
        train(args)?;
        let output = dir.0.join("scratch");
        let records = metrics(&output)?;
        assert_eq!(records.len(), 4);
        assert!(records[0]["validation_loss"].as_f64().unwrap().is_finite());
        assert!(records[1].get("validation_loss").is_none());
        let expected_tokens = [
            batch_size * 16,
            windows.len() * 16,
            (windows.len() + batch_size) * 16,
        ];
        let (config, _) = parse_config(&serde_json::from_str(CONFIG)?)?;
        for step in 1..=3 {
            let metric = &records[step];
            assert_eq!(metric["step"], step);
            assert_eq!(metric["tokens_seen"], expected_tokens[step - 1]);
            for key in ["train_loss", "gradient_norm"] {
                assert!(metric[key].as_f64().unwrap().is_finite(), "{metric}");
            }
            let checkpoint = output.join(format!("checkpoint-{step}"));
            assert!(!output.join(format!(".checkpoint-{step}.partial")).exists());
            let state = read_json(checkpoint.join("training-state.json"))?;
            assert_eq!(state["step"], step);
            assert_eq!(state["tokens_seen"], expected_tokens[step - 1]);
            assert_eq!(state["parameters"], parameter_count(&config) as usize);
            assert_eq!(state["dtype"], "f32");
            assert_eq!(state["optimizer"], "AdamW");
            assert_eq!(state["optimizer_state_saved"], false);
            assert_eq!(state["frozen_parameters"], "*.layer_scalar");
            assert_eq!(state["source_checkpoint"], Value::Null);
            assert_eq!(state["initializer_range"], json!(0.02f32));
            assert!(state["initialization_policy"]
                .as_str()
                .unwrap()
                .contains("CPU RNG unseeded"));
            assert_eq!(state["train_windows"], windows.len());
            assert_eq!(state["valid_windows"], metadata["valid"]["windows"]);
            if step >= 2 {
                assert_eq!(state["validation_step"], step);
                assert!(state["validation_loss"].as_f64().unwrap().is_finite());
            }
            for name in [
                "config.json",
                "source-config.json",
                "tokenizer.json",
                "dataset-metadata.json",
            ] {
                assert_eq!(
                    fs::read(checkpoint.join(name))?,
                    fs::read(output.join(name))?
                );
            }
        }
        assert_eq!(
            fs::read_to_string(output.join("source-config.json"))?,
            CONFIG
        );
        assert_eq!(
            fs::read_to_string(output.join("tokenizer.json"))?,
            TOKENIZER
        );
        assert_eq!(read_json(output.join("dataset-metadata.json"))?, metadata);
        let mut expected_config: Value = serde_json::from_str(CONFIG)?;
        expected_config["max_position_embeddings"] = json!(16);
        expected_config["use_flash_attn"] = json!(false);
        assert_eq!(read_json(output.join("config.json"))?, expected_config);

        let checkpoint = output.join("checkpoint-3");
        // Initialization settings are irrelevant when every weight is loaded.
        let mut warm_config: Value = serde_json::from_str(CONFIG)?;
        warm_config["initializer_range"] = json!(0.3);
        fs::write(dir.0.join("config.json"), serde_json::to_vec(&warm_config)?)?;
        let mut args = dir.args("warm");
        args.from_scratch = false;
        args.checkpoint = Some(checkpoint.clone());
        args.steps = 1;
        args.batch_size = batch_size;
        train(args)?;
        let warm = dir.0.join("warm");
        let warm_metrics = metrics(&warm)?;
        assert_eq!(warm_metrics.len(), 2);
        assert!(
            (warm_metrics[0]["validation_loss"].as_f64().unwrap()
                - records[3]["validation_loss"].as_f64().unwrap())
            .abs()
                < 1e-6
        );
        let state = read_json(warm.join("checkpoint-1/training-state.json"))?;
        assert_eq!(state["source_checkpoint"], json!(checkpoint));
        assert_eq!(state["initializer_range"], Value::Null);
        assert_eq!(state["initialization_policy"], "checkpoint warm start");
        assert_eq!(state["step"], 1);
        assert_eq!(state["tokens_seen"], batch_size * 16);
        assert_eq!(state["steps_requested"], 1);
        assert_eq!(warm_metrics[1]["tokens_seen"], batch_size * 16);

        for (before, after) in [
            (output.join("checkpoint-1"), output.join("checkpoint-3")),
            (checkpoint, warm.join("checkpoint-1")),
        ] {
            let before = candle::safetensors::load(before.join("model.safetensors"), &Device::Cpu)?;
            let after = candle::safetensors::load(after.join("model.safetensors"), &Device::Cpu)?;
            assert_eq!(before.len(), after.len());
            let mut frozen = 0;
            let mut changed = BTreeSet::new();
            for (name, tensor) in &after {
                assert_eq!(tensor.dtype(), DType::F32);
                let values = tensor.flatten_all()?.to_vec1::<f32>()?;
                assert!(values.iter().all(|v| v.is_finite()), "{name}");
                let previous = before[name].flatten_all()?.to_vec1::<f32>()?;
                if name.ends_with(".layer_scalar") {
                    frozen += 1;
                    assert_eq!(values, previous);
                    assert!(values.iter().all(|&v| v == 1.0));
                } else if values != previous {
                    changed.insert(name.as_str());
                }
            }
            assert_eq!(frozen, config.num_hidden_layers);
            for suffix in [
                "embed_tokens.weight",
                "self_attn.q_proj.weight",
                "mlp.down_proj.weight",
            ] {
                assert!(
                    changed.iter().any(|name| name.ends_with(suffix)),
                    "unchanged {suffix}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn sharded_weights_load_exact_values_convert_dtype_and_reject_missing_data() -> Result<()> {
        let dir = TempDir::new()?;
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
        let a = vb.get((2, 2), "a")?;
        let b = vb.get(2, "b")?;
        let first = HashMap::from([("a", Tensor::new(&[[1f32, -2.], [3.5, 0.]], &Device::Cpu)?)]);
        let second = HashMap::from([(
            "b",
            Tensor::new(&[0.25f32, -8.], &Device::Cpu)?.to_dtype(DType::BF16)?,
        )]);
        candle::safetensors::save(&first, dir.0.join("first.safetensors"))?;
        candle::safetensors::save(&second, dir.0.join("second.safetensors"))?;
        fs::write(
            dir.0.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "weight_map": {"a": "first.safetensors", "b": "second.safetensors"}
            }))?,
        )?;
        assert_eq!(checkpoint_files(&dir.0)?.len(), 2);
        load_weights(&vars, &dir.0)?;
        assert_eq!(a.to_vec2::<f32>()?, vec![vec![1., -2.], vec![3.5, 0.]]);
        assert_eq!(b.dtype(), DType::F32);
        assert_eq!(b.to_vec1::<f32>()?, vec![0.25, -8.]);
        candle::safetensors::save(&first, dir.0.join("second.safetensors"))?;
        let error = mmap_checkpoint(&dir.0).err().expect("duplicate must fail");
        assert!(error.to_string().contains("duplicate checkpoint tensor a"));
        candle::safetensors::save(&second, dir.0.join("second.safetensors"))?;
        fs::remove_file(dir.0.join("second.safetensors"))?;
        assert!(load_weights(&vars, &dir.0).is_err());
        candle::safetensors::save(
            &HashMap::from([("other", second["b"].clone())]),
            dir.0.join("second.safetensors"),
        )?;
        assert!(load_weights(&vars, &dir.0).is_err());
        Ok(())
    }

    #[test]
    fn quantized_head_checkpoint_roundtrip_and_policy_validation() -> Result<()> {
        let dir = TempDir::new()?;
        let mut raw: Value = serde_json::from_str(CONFIG)?;
        for (key, value) in [
            ("hidden_size", 32),
            ("intermediate_size", 64),
            ("head_dim", 32),
            ("global_head_dim", 32),
        ] {
            raw[key] = json!(value);
        }
        fs::write(dir.0.join("config.json"), serde_json::to_vec(&raw)?)?;
        fs::write(dir.0.join("tokenizer.json"), TOKENIZER)?;
        fs::write(dir.0.join("input.jsonl"), DOCUMENTS)?;
        dataset::prepare(dataset::PrepareArgs {
            input: dir.0.join("input.jsonl"),
            tokenizer: dir.0.join("tokenizer.json"),
            output: dir.0.join("prepared"),
            sequence_length: 8,
            vocab_size: 16,
            validation_fraction: 0.25,
            seed: 42,
            bos_token_id: Some(1),
            eos_token_id: 2,
        })?;
        let base = dir.0.join("base");
        fs::create_dir(&base)?;
        fs::copy(dir.0.join("config.json"), base.join("config.json"))?;
        fs::copy(dir.0.join("tokenizer.json"), base.join("tokenizer.json"))?;
        let vars = VarMap::new();
        let (config, _) = parse_config(&raw)?;
        TextModel::new(
            &config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )?;
        initialize_weights(&vars, 0.02)?;
        vars.save(base.join("model.safetensors"))?;
        let before = fs::read(base.join("model.safetensors"))?;
        for (name, quant, mode) in [
            ("q4", Quantization::Q4_0, Mode::QuantizedHead),
            ("q8", Quantization::Q8_0, Mode::QuantizedHead),
            ("lora-q4", Quantization::Q4_0, Mode::QuantizedLora),
        ] {
            let make_args = |output: &str| {
                let mut args = dir.args(output);
                args.mode = mode;
                args.precision = Precision::Bf16;
                args.quantization = quant;
                args.from_scratch = false;
                args.checkpoint = Some(base.clone());
                args.steps = 2;
                args
            };
            train(make_args(name))?;
            let path = dir.0.join(name).join("checkpoint-2");
            assert!(!path.join("model.safetensors").exists());
            let state = read_json(path.join("training-state.json"))?;
            assert_eq!(state["dtype"], "bf16");
            assert_eq!(state["master_dtype"], "f32");
            assert_eq!(state["optimizer_dtype"], "f32");
            assert_eq!(state["frozen_parameters"], "entire base model");
            assert_eq!(state["mode"], mode.as_str());
            assert_eq!(
                state["trainable_parameters"],
                if mode == Mode::QuantizedHead {
                    8 * (32 + 16)
                } else {
                    9216
                }
            );
            assert_eq!(state["quantized_projections"], 13);
            let saved = candle::safetensors::load(path.join("adapter.safetensors"), &Device::Cpu)?;
            assert_eq!(
                saved.len(),
                if mode == Mode::QuantizedHead { 2 } else { 26 }
            );
            for tensor in saved.values() {
                assert_eq!(tensor.dtype(), DType::F32);
                assert!(tensor
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|v| v.is_finite()));
            }
            let warm_name = format!("{name}-warm");
            let mut args = make_args(&warm_name);
            args.adapter_checkpoint = Some(path.clone());
            train(args)?;
            let initial = metrics(&dir.0.join(&warm_name))?[0]["validation_loss"]
                .as_f64()
                .unwrap();
            assert!((initial - state["validation_loss"].as_f64().unwrap()).abs() < 1e-6);
            let mut args = make_args(&format!("{name}-rejected"));
            args.adapter_checkpoint = Some(path.clone());
            args.lora_alpha = 4.;
            assert!(train(args)
                .unwrap_err()
                .to_string()
                .contains("policy mismatch"));
            let mut args = make_args(&format!("{name}-wrong-mode"));
            args.adapter_checkpoint = Some(path);
            args.mode = if mode == Mode::QuantizedHead {
                Mode::QuantizedLora
            } else {
                Mode::QuantizedHead
            };
            assert!(train(args)
                .unwrap_err()
                .to_string()
                .contains("policy mismatch"));
        }
        assert_eq!(fs::read(base.join("model.safetensors"))?, before);
        let mut args = dir.args("rejected");
        args.mode = Mode::QuantizedHead;
        assert!(train(args)
            .unwrap_err()
            .to_string()
            .contains("frozen pretrained"));
        let mut args = dir.args("rejected");
        args.precision = Precision::Bf16;
        assert!(train(args)
            .unwrap_err()
            .to_string()
            .contains("full training supports F32"));
        Ok(())
    }

    #[test]
    fn accumulation_matches_large_batch_including_short_epoch_tail() -> Result<()> {
        let dir = TempDir::new()?;
        fs::write(dir.0.join("config.json"), CONFIG)?;
        fs::write(dir.0.join("tokenizer.json"), TOKENIZER)?;
        fs::write(dir.0.join("input.jsonl"), DOCUMENTS)?;
        dataset::prepare(dataset::PrepareArgs {
            input: dir.0.join("input.jsonl"),
            tokenizer: dir.0.join("tokenizer.json"),
            output: dir.0.join("prepared"),
            sequence_length: 8,
            vocab_size: 16,
            validation_fraction: 0.75,
            seed: 42,
            bos_token_id: Some(1),
            eos_token_id: 2,
        })?;
        let mut args = dir.args("base");
        args.steps = 1;
        train(args)?;
        let windows = read_json(dir.0.join("prepared/metadata.json"))?["train"]["windows"]
            .as_u64()
            .unwrap() as usize;
        assert!(windows > 2);
        for (output, batch, accumulation) in [("large", windows, 1), ("micro", windows - 1, 2)] {
            let mut args = dir.args(output);
            args.from_scratch = false;
            args.checkpoint = Some(dir.0.join("base/checkpoint-1"));
            args.steps = 1;
            args.batch_size = batch;
            args.gradient_accumulation_steps = accumulation;
            train(args)?;
            let state = read_json(dir.0.join(output).join("checkpoint-1/training-state.json"))?;
            assert_eq!(state["tokens_seen"], windows * 8);
            assert_eq!(state["gradient_accumulation_steps"], accumulation);
        }
        let load = |name: &str| {
            candle::safetensors::load(
                dir.0.join(name).join("checkpoint-1/model.safetensors"),
                &Device::Cpu,
            )
        };
        let large = load("large")?;
        let micro = load("micro")?;
        for (name, tensor) in large {
            let error = (tensor - &micro[&name])?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(error < 2e-5, "{name}: accumulation error {error}");
        }
        let mut args = dir.args("invalid");
        args.gradient_accumulation_steps = 0;
        assert!(train(args)
            .unwrap_err()
            .to_string()
            .contains("gradient-accumulation-steps"));
        assert!(!dir.0.join("invalid").exists());
        Ok(())
    }

    #[test]
    fn checkpoint_index_rejects_nonlocal_and_empty_shards() -> Result<()> {
        let dir = TempDir::new()?;
        for name in [
            "../outside.safetensors",
            "/tmp/outside.safetensors",
            "nested/model.safetensors",
            "",
        ] {
            fs::write(
                dir.0.join("model.safetensors.index.json"),
                serde_json::to_vec(&json!({"weight_map": {"a": name}}))?,
            )?;
            assert!(checkpoint_files(&dir.0)
                .unwrap_err()
                .to_string()
                .contains("plain local filename"));
        }
        for raw in [
            json!({}),
            json!({"weight_map": {}}),
            json!({"weight_map": {"a": 1}}),
        ] {
            fs::write(
                dir.0.join("model.safetensors.index.json"),
                serde_json::to_vec(&raw)?,
            )?;
            assert!(checkpoint_files(&dir.0).is_err());
        }
        Ok(())
    }

    #[test]
    fn parameter_estimate_matches_allocated_tied_bias_and_shared_kv_variants() -> Result<()> {
        for tied in [false, true] {
            for bias in [false, true] {
                for shared in [false, true] {
                    for global_heads in [None, Some(1), Some(2)] {
                        let (mut config, _) = parse_config(&serde_json::from_str(CONFIG)?)?;
                        config.tie_word_embeddings = tied;
                        config.attention_bias = bias;
                        config.attention_k_eq_v = shared;
                        config.num_global_key_value_heads = global_heads;
                        config.global_head_dim = 16;
                        let vars = VarMap::new();
                        let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
                        let _model = if tied {
                            TextModel::new(&config, vb)?
                        } else {
                            TextModel::new_for_conditional_generation(&config, vb)?
                        };
                        let actual: usize = vars.all_vars().iter().map(|v| v.elem_count()).sum();
                        assert_eq!(parameter_count(&config), actual as f64,
                            "tied={tied}, bias={bias}, shared={shared}, global_heads={global_heads:?}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn scratch_initialization_uses_config_std_and_preserves_norms_and_scalars() -> Result<()> {
        let mut raw: Value = serde_json::from_str(CONFIG)?;
        raw["initializer_range"] = json!(0.07);
        raw["hidden_size"] = json!(128);
        raw["intermediate_size"] = json!(256);
        raw["vocab_size"] = json!(1024);
        raw["head_dim"] = json!(32);
        raw["global_head_dim"] = json!(32);
        raw["attention_bias"] = json!(true);
        raw["tie_word_embeddings"] = json!(false);
        let raw = json!({"model_type": "gemma4", "text_config": raw});
        let (config, _) = parse_config(&raw)?;
        let std = initializer_range(&raw)?;
        assert_eq!(std, 0.07f32);
        let vars = VarMap::new();
        let _model = TextModel::new_for_conditional_generation(
            &config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )?;
        let mut constants = HashMap::new();
        for (name, var) in vars.data().lock().unwrap().iter() {
            if var.rank() != 2 && !name.ends_with(".bias") {
                let values = var.flatten_all()?.to_vec1::<f32>()?;
                assert!(values.iter().all(|&v| v == 1.0), "{name}");
                constants.insert(name.clone(), values);
            }
        }
        initialize_weights(&vars, std)?;
        let mut matrices = 0;
        let mut biases = 0;
        let mut scalars = 0;
        for (name, var) in vars.data().lock().unwrap().iter() {
            let values = var.flatten_all()?.to_vec1::<f32>()?;
            if var.rank() == 2 {
                matrices += 1;
                assert!(values.len() >= 4096, "{name}");
                let mean = values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64;
                let measured_std = (values
                    .iter()
                    .map(|&v| (v as f64 - mean).powi(2))
                    .sum::<f64>()
                    / values.len() as f64)
                    .sqrt();
                assert!(mean.abs() < std as f64 * 0.1, "{name}: mean={mean}");
                assert!(
                    (measured_std / std as f64 - 1.0).abs() < 0.1,
                    "{name}: std={measured_std}, expected={std}"
                );
            } else if name.ends_with(".bias") {
                biases += 1;
                assert!(values.iter().all(|&v| v == 0.0), "{name}");
            } else {
                assert_eq!(values, constants[name], "{name}");
                scalars += usize::from(name.ends_with(".layer_scalar"));
            }
        }
        assert!(matrices > 2 && biases > 0);
        assert_eq!(scalars, config.num_hidden_layers);
        assert!(constants.len() > scalars);
        Ok(())
    }

    #[test]
    fn validates_initializer_and_dropout_for_flat_and_nested_configs_before_allocation(
    ) -> Result<()> {
        let dir = TempDir::new()?;
        let fixture: Value = serde_json::from_str(CONFIG)?;
        assert_eq!(initializer_range(&fixture)?, 0.02f32);
        for nested in [false, true] {
            for (key, invalid) in [
                (
                    "initializer_range",
                    vec![
                        json!(0),
                        json!(-0.1),
                        json!(1e100),
                        json!(1e-100),
                        json!(null),
                        json!("NaN"),
                        json!("Infinity"),
                        json!(true),
                        json!([]),
                    ],
                ),
                (
                    "attention_dropout",
                    vec![
                        json!(0.1),
                        json!(-0.1),
                        json!(1),
                        json!(1e-100),
                        json!(null),
                        json!("NaN"),
                        json!("Infinity"),
                        json!(false),
                        json!({}),
                    ],
                ),
            ] {
                for value in invalid {
                    let mut raw = fixture.clone();
                    raw[key] = value;
                    if nested {
                        raw = json!({"model_type": "gemma4", "text_config": raw});
                    }
                    let error = parse_config(&raw).unwrap_err().to_string();
                    assert!(error.contains(key), "{error}");
                    fs::write(dir.0.join("config.json"), serde_json::to_vec(&raw)?)?;
                    let error = train(dir.args("rejected")).unwrap_err().to_string();
                    assert!(error.contains(key), "{error}");
                    assert!(!dir.0.join("rejected").exists());
                }
            }
            let mut raw = fixture.clone();
            raw["attention_dropout"] = json!(0);
            raw["initializer_range"] = json!(1);
            if nested {
                raw = json!({"model_type": "gemma4", "text_config": raw,
                    "initializer_range": "ignored", "attention_dropout": 0.5});
            }
            parse_config(&raw)?;
            assert_eq!(initializer_range(&raw)?, 1f32);
        }
        assert!(!dir.0.join("prepared").exists());
        Ok(())
    }

    #[test]
    fn rejects_whole_window_truncation_and_inconsistent_metadata_without_output() -> Result<()> {
        let dir = TempDir::new()?;
        fs::write(dir.0.join("config.json"), CONFIG)?;
        let dataset = dir.0.join("prepared");
        fs::create_dir(&dataset)?;
        fs::write(dataset.join("tokenizer.json"), TOKENIZER)?;
        let sequence_length = 16;
        let stored_tokens = 3 * sequence_length + 1;
        let metadata = json!({
            "format_version": 1, "token_format": "u32-le", "sequence_length": sequence_length,
            "window_length": sequence_length + 1, "stride": sequence_length, "vocab_size": 16,
            "train": {"stored_tokens": stored_tokens, "windows": 3},
            "valid": {"stored_tokens": stored_tokens, "windows": 3}
        });
        let metadata_bytes = serde_json::to_vec(&metadata)?;
        fs::write(dataset.join("metadata.json"), &metadata_bytes)?;
        for split in ["train", "valid"] {
            fs::write(
                dataset.join(format!("{split}.bin")),
                vec![0u8; 4 * stored_tokens],
            )?;
        }
        for split in ["train", "valid"] {
            let path = dataset.join(format!("{split}.bin"));
            let original = fs::read(&path)?;
            fs::write(&path, &original[..original.len() - 4 * sequence_length])?;
            // This is still a structurally valid stream, but no longer the recorded split.
            assert_eq!(dataset::load(&path, sequence_length, 16)?.len(), 2);
            assert_eq!(fs::read(dataset.join("metadata.json"))?, metadata_bytes);
            let error = train(dir.args("rejected")).unwrap_err().to_string();
            assert!(
                error.contains(&format!("{split}.bin")) && error.contains("byte length"),
                "{error}"
            );
            assert!(!dir.0.join("rejected").exists());
            fs::write(&path, original)?;
            for (tokens, windows, expected) in [
                (stored_tokens - 1, 3, "stored_tokens"),
                (stored_tokens, 2, "stored_tokens"),
                (1, 0, "no windows"),
                (usize::MAX, usize::MAX, "token count overflow"),
                (
                    usize::MAX / 4 + 2,
                    (usize::MAX / 4 + 1) / sequence_length,
                    "byte count overflow",
                ),
            ] {
                let mut invalid = metadata.clone();
                invalid[split] = json!({"stored_tokens": tokens, "windows": windows});
                fs::write(dataset.join("metadata.json"), serde_json::to_vec(&invalid)?)?;
                let error = train(dir.args("rejected")).unwrap_err().to_string();
                assert!(error.contains(expected), "{error}");
                assert!(!dir.0.join("rejected").exists());
            }
            fs::write(dataset.join("metadata.json"), &metadata_bytes)?;
        }
        Ok(())
    }

    #[test]
    fn rejects_config_and_parameter_cap_before_dataset_or_output_allocation() -> Result<()> {
        let dir = TempDir::new()?;
        let fixture: Value = serde_json::from_str(CONFIG)?;
        for (key, value) in [
            ("model_type", json!("gemma3_text")),
            ("enable_moe_block", json!(true)),
            ("hidden_size_per_layer_input", json!(8)),
            ("num_kv_shared_layers", json!(1)),
            ("use_double_wide_mlp", json!(true)),
            ("hidden_size", json!(0)),
            ("head_dim", json!(3)),
            ("layer_types", json!(["unknown", "full_attention"])),
        ] {
            let mut raw = fixture.clone();
            raw[key] = value;
            assert!(parse_config(&raw).is_err(), "{key}");
            fs::write(dir.0.join("config.json"), serde_json::to_vec(&raw)?)?;
            let error = train(dir.args("rejected")).unwrap_err().to_string();
            assert!(
                error.contains(key) || error.contains("expected gemma4_text"),
                "{error}"
            );
            assert!(!dir.0.join("rejected").exists());
        }
        fs::write(dir.0.join("config.json"), CONFIG)?;
        let mut args = dir.args("capped");
        args.max_parameters = parameter_count(&parse_config(&fixture)?.0) as usize - 1;
        let error = train(args).unwrap_err().to_string();
        assert!(error.contains("above --max-parameters"), "{error}");
        assert!(!dir.0.join("capped").exists());
        assert!(!dir.0.join("prepared").exists());
        let mut nested = json!({"model_type": "gemma4", "text_config": fixture});
        nested["text_config"]["use_flash_attn"] = json!(true);
        let (config, conditional) = parse_config(&nested)?;
        assert!(conditional);
        assert!(!config.use_flash_attn);
        nested["model_type"] = json!("gemma4_unified");
        assert!(parse_config(&nested).is_err());
        Ok(())
    }

    #[test]
    fn cli_requires_explicit_initialization_and_rejects_conflicting_devices() {
        let base = [
            "gemma4-training",
            "train",
            "--config",
            "config.json",
            "--dataset",
            "data",
            "--output",
            "run",
        ];
        for (extra, expected) in [
            (vec![], clap::error::ErrorKind::MissingRequiredArgument),
            (
                vec!["--from-scratch", "--checkpoint", "checkpoint"],
                clap::error::ErrorKind::ArgumentConflict,
            ),
            (
                vec!["--from-scratch", "--cpu", "--cuda"],
                clap::error::ErrorKind::ArgumentConflict,
            ),
        ] {
            let error = Cli::try_parse_from(base.iter().copied().chain(extra))
                .err()
                .expect("CLI should reject arguments");
            assert_eq!(error.kind(), expected);
        }
        for extra in [
            vec!["--from-scratch"],
            vec!["--checkpoint", "checkpoint", "--cpu"],
        ] {
            assert!(Cli::try_parse_from(base.iter().copied().chain(extra)).is_ok());
        }
    }

    #[test]
    fn gradient_clipping_uses_global_norm_and_rejects_nonfinite_or_missing_gradients() -> Result<()>
    {
        let a = Var::new(&[3f32], &Device::Cpu)?;
        let b = Var::new(&[4f32], &Device::Cpu)?;
        let vars = [a.clone(), b.clone()];
        let loss = ((a.sqr()?.sum_all()? + b.sqr()?.sum_all()?)? * 0.5)?;
        let mut grads = loss.backward()?;
        assert_eq!(clip_gradients(&mut grads, &vars, 10.)?, 5.);
        assert_eq!(grads.get(&a).unwrap().to_vec1::<f32>()?, vec![3.]);
        assert_eq!(clip_gradients(&mut grads, &vars, 1.)?, 5.);
        let norm = (grads
            .get(&a)
            .unwrap()
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?
            + grads
                .get(&b)
                .unwrap()
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?)
        .sqrt();
        assert!((norm - 1.).abs() < 1e-6);
        for value in [f32::INFINITY, f32::NAN] {
            grads.insert(&a, Tensor::new(&[value], &Device::Cpu)?);
            assert!(clip_gradients(&mut grads, &vars, 1.)
                .unwrap_err()
                .to_string()
                .contains("non-finite gradients"));
        }
        let missing = Var::new(&[1f32], &Device::Cpu)?;
        assert!(clip_gradients(&mut grads, &[missing], 1.)
            .unwrap_err()
            .to_string()
            .contains("missing trainable parameter gradient"));
        Ok(())
    }
}
