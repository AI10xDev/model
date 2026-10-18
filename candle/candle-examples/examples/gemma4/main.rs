#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Error as E, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use candle_transformers::models::gemma4::{config::Gemma4TextConfig, text::TextModel};

use candle::{DType, Device, Tensor};
use candle_examples::hub::Api;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use tokenizers::Tokenizer;

fn parse_text_config(bytes: &[u8], use_flash_attn: bool) -> Result<(Gemma4TextConfig, bool)> {
    let raw: serde_json::Value = serde_json::from_slice(bytes)?;
    let conditional_generation = raw.get("text_config").is_some();
    let expected_type = if conditional_generation {
        "gemma4"
    } else {
        "gemma4_text"
    };
    if let Some(model_type) = raw.get("model_type") {
        anyhow::ensure!(
            model_type.as_str() == Some(expected_type),
            "expected {expected_type} config, got model_type={model_type}"
        );
    }
    let text_config = if conditional_generation {
        let text_config = &raw["text_config"];
        if let Some(model_type) = text_config.get("model_type") {
            anyhow::ensure!(
                model_type.as_str() == Some("gemma4_text"),
                "expected gemma4_text text_config, got model_type={model_type}"
            );
        }
        text_config.clone()
    } else {
        raw
    };
    let mut config: Gemma4TextConfig = serde_json::from_value(text_config)?;
    config.use_flash_attn = use_flash_attn;
    config.validate().context(
        "unsupported Gemma4 text config: this example supports only the partial dense text path; \
         E2B/E4B PLE/KV sharing, MoE, and unified models are not supported",
    )?;
    Ok((config, conditional_generation))
}

fn format_prompt(prompt: &str, raw_prompt: bool) -> String {
    if raw_prompt {
        return prompt.to_owned();
    }
    // Single user turn, add_generation_prompt=true, enable_thinking=false:
    // https://huggingface.co/google/gemma-4-31B-it/blob/main/chat_template.jinja
    format!(
        "<bos><|turn>user\n{}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        prompt.trim()
    )
}

fn prompt_tokens(tokenizer: &Tokenizer, prompt: &str, raw_prompt: bool) -> Result<Vec<u32>> {
    // The template includes BOS; raw prompts must supply their own special tokens.
    let encoding = tokenizer
        .encode(format_prompt(prompt, raw_prompt), false)
        .map_err(E::msg)?;
    anyhow::ensure!(!encoding.get_ids().is_empty(), "prompt contains no tokens");
    Ok(encoding.get_ids().to_vec())
}

fn stop_tokens(tokenizer: &Tokenizer) -> Result<[u32; 3]> {
    let token_id = |token| {
        tokenizer
            .token_to_id(token)
            .with_context(|| format!("cannot find the Gemma4 {token} stop token"))
    };
    Ok([
        token_id("<eos>")?,
        token_id("<turn|>")?,
        token_id("<|tool_response>")?,
    ])
}

fn context_budget(
    declared: usize,
    override_cap: Option<usize>,
    prompt: usize,
    sample: usize,
) -> Result<usize> {
    let cap = override_cap.unwrap_or(declared);
    anyhow::ensure!(
        cap > 0 && cap <= declared,
        "--max-context must be positive and <= base declared context {declared}"
    );
    let total = prompt
        .checked_add(sample)
        .context("prompt + requested sample token budget overflow")?;
    anyhow::ensure!(prompt > 0 && total <= cap,
        "prompt ({prompt}) + requested sample ({sample}) = {total} exceeds runtime context {cap} (base declared {declared})");
    Ok(cap)
}

fn synthetic_prompt(seed: &[u32], length: usize, bos: Option<u32>) -> Result<Vec<u32>> {
    anyhow::ensure!(
        !seed.is_empty() && length > 0,
        "synthetic prompt requires nonempty seed tokens and positive length"
    );
    let preserve_bos = bos == seed.first().copied();
    let mut tokens = Vec::with_capacity(length);
    if preserve_bos {
        tokens.push(seed[0]);
    }
    // Keep the initial BOS without repeating it when there is other seed content.
    let repeat = if preserve_bos && seed.len() > 1 {
        &seed[1..]
    } else {
        seed
    };
    tokens.extend(repeat.iter().copied().cycle().take(length - tokens.len()));
    Ok(tokens)
}

fn checkpoint_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let index = directory.join("model.safetensors.index.json");
    if !index.try_exists()? {
        return Ok(vec![directory.join("model.safetensors")]);
    }
    let index: Value = serde_json::from_slice(&std::fs::read(index)?)?;
    let map = index["weight_map"]
        .as_object()
        .context("missing weight_map")?;
    let mut files = BTreeSet::new();
    for value in map.values() {
        let name = value.as_str().context("shard filename must be a string")?;
        let mut components = Path::new(name).components();
        anyhow::ensure!(
            matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
            "shard filename must be a plain local filename: {name}"
        );
        files.insert(directory.join(name));
    }
    anyhow::ensure!(!files.is_empty(), "checkpoint index has no shards");
    Ok(files.into_iter().collect())
}

fn validate_adapter_metadata(
    state: &Value,
    base: &(Gemma4TextConfig, bool),
    source: &(Gemma4TextConfig, bool),
    trained: &(Gemma4TextConfig, bool),
    base_tokenizer: &[u8],
    adapter_tokenizer: &[u8],
) -> Result<(usize, f64)> {
    anyhow::ensure!(state["format_version"] == 1 && state["mode"] == "quantized-lora"
        && state["quantization"] == "Q4_0" && state["dtype"] == "bf16"
        && state["master_dtype"] == "f32", "adapter requires saved mode quantized-lora, quantization q4-0 (Q4_0), precision bf16 and F32 masters (format_version 1)");
    anyhow::ensure!(
        state["source_checkpoint"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "adapter missing source_checkpoint"
    );
    let rank = usize::try_from(
        state["lora_rank"]
            .as_u64()
            .context("adapter missing positive lora_rank")?,
    )?;
    let alpha = state["lora_alpha"]
        .as_f64()
        .context("adapter missing lora_alpha")?;
    anyhow::ensure!(
        rank > 0 && alpha.is_finite() && alpha > 0.,
        "adapter rank and alpha must be positive and finite"
    );
    for (name, (config, conditional)) in [("source-config", source), ("training config", trained)] {
        let mut config = config.clone();
        config.max_position_embeddings = base.0.max_position_embeddings;
        anyhow::ensure!(
            config == base.0 && *conditional == base.1,
            "adapter {name} architecture mismatch (only context length may differ)"
        );
    }
    anyhow::ensure!(
        trained.0.max_position_embeddings <= source.0.max_position_embeddings
            && state["sequence_length"] == trained.0.max_position_embeddings,
        "adapter training context metadata mismatch"
    );
    anyhow::ensure!(
        base_tokenizer == adapter_tokenizer,
        "adapter tokenizer mismatch"
    );
    Ok((rank, alpha))
}

fn adapter_shapes(config: &Gemma4TextConfig, rank: usize) -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    let h = config.hidden_size;
    for i in 0..config.num_hidden_layers {
        let local = config.is_sliding(i);
        let shared = config.attention_k_eq_v && !local;
        let d = if local {
            config.head_dim
        } else {
            config.global_head_dim
        };
        let kv = if shared {
            config
                .num_global_key_value_heads
                .unwrap_or(config.num_key_value_heads)
        } else {
            config.num_key_value_heads
        } * d;
        let q = config.num_attention_heads * d;
        for (name, input, output) in [
            ("self_attn.q_proj", h, q),
            ("self_attn.k_proj", h, kv),
            ("self_attn.v_proj", h, kv),
            ("self_attn.o_proj", q, h),
            ("mlp.gate_proj", h, config.intermediate_size),
            ("mlp.up_proj", h, config.intermediate_size),
            ("mlp.down_proj", config.intermediate_size, h),
        ] {
            if shared && name == "self_attn.v_proj" {
                continue;
            }
            shapes.insert(format!("layers.{i}.{name}.lora_a"), vec![rank, input]);
            shapes.insert(format!("layers.{i}.{name}.lora_b"), vec![output, rank]);
        }
    }
    shapes
}

fn validate_adapter_tensors(
    tensors: Vec<(String, safetensors::tensor::TensorView<'_>)>,
    expected: &BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    for (name, tensor) in tensors {
        anyhow::ensure!(seen.insert(name.clone()), "duplicate adapter tensor {name}");
        let shape = expected
            .get(&name)
            .with_context(|| format!("unexpected adapter tensor {name}"))?;
        anyhow::ensure!(
            tensor.shape() == shape && tensor.dtype() == safetensors::Dtype::F32,
            "adapter tensor {name} must be F32 with shape {shape:?}"
        );
        anyhow::ensure!(
            tensor
                .data()
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "non-finite adapter tensor {name}"
        );
    }
    anyhow::ensure!(
        seen.len() == expected.len(),
        "missing adapter tensors: expected {}, got {}",
        expected.len(),
        seen.len()
    );
    Ok(())
}

struct TextGeneration {
    model: TextModel,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: TextModel,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = {
            let temperature = temp.unwrap_or(0.);
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        };

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(
        &mut self,
        mut tokens: Vec<u32>,
        sample_len: usize,
        probe: bool,
        ignore_eos: bool,
    ) -> Result<Value> {
        use std::io::Write;
        self.tokenizer.clear();
        self.model.clear_kv_cache();
        let stop_tokens = stop_tokens(self.tokenizer.tokenizer())?;
        let prompt_len = tokens.len();
        if probe {
            println!("synthetic capacity probe: {prompt_len} prompt tokens, {sample_len} requested samples (prompt not printed)");
        } else {
            for &t in tokens.iter() {
                if let Some(t) = self.tokenizer.next_token(t)? {
                    print!("{t}")
                }
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let mut forwarded_tokens = 0usize;
        let mut prefill_seconds = 0.;
        let mut cached_decode_seconds = 0.;
        let mut stopped_on_eos = false;
        self.device.synchronize()?;
        let start_gen = Instant::now();
        for index in 0..sample_len {
            self.device.synchronize()?;
            let start_forward = Instant::now();
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            // Bound prefill masks and decoder activations, preserving absolute RoPE
            // positions and sampling only after the entire prompt has been forwarded.
            let mut logits = None;
            for (chunk_index, chunk) in ctxt.chunks(256).enumerate() {
                let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
                logits = Some(self.model.forward(&input, start_pos + chunk_index * 256)?);
            }
            let logits = logits.context("prompt contains no tokens")?;
            self.device.synchronize()?;
            let seconds = start_forward.elapsed().as_secs_f64();
            if index == 0 {
                prefill_seconds = seconds;
            } else {
                cached_decode_seconds += seconds;
            }
            forwarded_tokens += context_size;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            anyhow::ensure!(
                logits.to_vec1::<f32>()?.iter().all(|v| v.is_finite()),
                "non-finite logits at forward position {start_pos}; capacity is inconclusive"
            );
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if !ignore_eos && stop_tokens.contains(&next_token) {
                stopped_on_eos = true;
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        self.device.synchronize()?;
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        let generated_ids = &tokens[prompt_len..];
        let output_text = self
            .tokenizer
            .tokenizer()
            .decode(generated_ids, true)
            .map_err(E::msg)?;
        Ok(json!({
            "prompt_tokens": prompt_len, "requested_generation_tokens": sample_len,
            "generated_tokens": generated_tokens, "generated_token_ids": generated_ids,
            "forwarded_tokens": forwarded_tokens,
            "max_forwarded_position_zero_based": forwarded_tokens.checked_sub(1),
            "forwarded_position_convention": "For G > 0: forwarded_tokens = P + G - 1; maximum zero-based position = P + G - 2. The final sampled token is not forwarded. For G = 0: no forward, maximum position null.",
            "prefill_tested": generated_tokens > 0, "cached_decode_steps": generated_tokens.saturating_sub(1),
            "cached_decode_tested": generated_tokens > 1,
            "stopped_on_eos": stopped_on_eos, "output_text": output_text,
            "timings_seconds": {"prefill_forward": prefill_seconds, "cached_decode_forward": cached_decode_seconds,
                "generation_wall": dt.as_secs_f64()},
            "timing_convention": "Device synchronized around each forward; forward timings include input upload, exclude sampling/output. Generation wall includes sampling and streamed output."
        }))
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    prompt: String,

    /// Use --prompt verbatim, without chat formatting or automatic BOS/special tokens.
    #[arg(long)]
    raw_prompt: bool,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    /// Load trained all-layer Q4_0/BF16 LoRA from a gemma4-training checkpoint.
    #[arg(long, value_name = "DIR")]
    adapter_checkpoint: Option<PathBuf>,

    /// Reduce the base config context allocation; never extends its declared limit.
    #[arg(long, value_name = "N")]
    max_context: Option<usize>,

    /// Capacity probe: repeat normal prompt token IDs to exactly N tokens.
    #[arg(long, value_name = "N")]
    probe_prompt_tokens: Option<usize>,

    /// Ignore all stop tokens, only for a synthetic capacity probe.
    #[arg(long, requires = "probe_prompt_tokens")]
    ignore_eos: bool,

    /// Write structured JSON after successful generation (probe requires cached decode).
    #[arg(long, value_name = "PATH")]
    report_file: Option<PathBuf>,

    /// Explicit checkpoint ID. Only experimental dense text (e.g. 31B) is supported.
    #[arg(long)]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Request multimodal execution (currently rejected as unvalidated).
    #[arg(long)]
    multimodal: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Use the slower dmmv cuda kernel.
    #[arg(long)]
    force_dmmv: bool,
}

impl Args {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.multimodal,
            "--multimodal is unvalidated/not supported by this example: image/audio inputs are not exposed by this CLI and multimodal support is incomplete; use text-only generation");
        anyhow::ensure!(
            self.max_context != Some(0),
            "--max-context must be positive"
        );
        anyhow::ensure!(
            self.probe_prompt_tokens != Some(0),
            "--probe-prompt-tokens must be positive"
        );
        anyhow::ensure!(
            !self.ignore_eos || self.probe_prompt_tokens.is_some(),
            "--ignore-eos requires --probe-prompt-tokens"
        );
        anyhow::ensure!(
            self.probe_prompt_tokens.is_none() || self.sample_len >= 2,
            "--probe-prompt-tokens requires --sample-len >= 2 to test prefill and cached decode"
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    args.validate()?;
    #[cfg(feature = "cuda")]
    candle::quantized::cuda::set_force_dmmv(args.force_dmmv);

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = Instant::now();
    let adapter_state: Option<Value> = args
        .adapter_checkpoint
        .as_ref()
        .map(|path| -> Result<Value> {
            Ok(serde_json::from_slice(&std::fs::read(
                path.join("training-state.json"),
            )?)?)
        })
        .transpose()?;
    let source_checkpoint = adapter_state
        .as_ref()
        .map(|state| -> Result<PathBuf> {
            let source = state["source_checkpoint"]
                .as_str()
                .context("adapter missing source_checkpoint")?;
            anyhow::ensure!(
                Path::new(source).is_absolute(),
                "adapter source_checkpoint must be absolute, as saved by training"
            );
            std::fs::canonicalize(source).with_context(|| {
                format!("adapter requires its original local base checkpoint: {source}")
            })
        })
        .transpose()?;
    let api = Api::new()?;
    let repo = api.model(&args.model_id).with_revision(&args.revision);
    let config_filename = match &args.config_file {
        Some(file) => PathBuf::from(file),
        None if source_checkpoint.is_some() => {
            source_checkpoint.as_ref().unwrap().join("config.json")
        }
        None => repo.get("config.json")?,
    };
    let (mut config, conditional_generation) =
        parse_text_config(&std::fs::read(&config_filename)?, args.use_flash_attn)
            .with_context(|| format!("invalid config {}", config_filename.display()))?;

    let tokenizer_filename = match &args.tokenizer_file {
        Some(file) => PathBuf::from(file),
        None if source_checkpoint.is_some() => {
            source_checkpoint.as_ref().unwrap().join("tokenizer.json")
        }
        None => repo.get("tokenizer.json")?,
    };
    let tokenizer_bytes = std::fs::read(&tokenizer_filename)?;
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).map_err(E::msg)?;
    anyhow::ensure!(
        tokenizer
            .get_vocab(true)
            .values()
            .all(|&id| (id as usize) < config.vocab_size),
        "tokenizer IDs exceed model vocabulary"
    );
    stop_tokens(&tokenizer)?;
    let seed_tokens = prompt_tokens(&tokenizer, &args.prompt, args.raw_prompt)?;
    let declared_context = config.max_position_embeddings;
    let runtime_context = context_budget(
        declared_context,
        args.max_context,
        args.probe_prompt_tokens.unwrap_or(seed_tokens.len()),
        args.sample_len,
    )?;
    // Validate before fetching/mapping weights, constructing the device, or allocating RoPE/cache.
    let tokens = match args.probe_prompt_tokens {
        Some(length) => synthetic_prompt(&seed_tokens, length, tokenizer.token_to_id("<bos>"))?,
        None => seed_tokens.clone(),
    };
    let adapter_policy = if let (Some(path), Some(state), Some(source)) =
        (&args.adapter_checkpoint, &adapter_state, &source_checkpoint)
    {
        let read_config = |path: PathBuf| -> Result<(Gemma4TextConfig, bool)> {
            parse_text_config(&std::fs::read(path)?, false)
        };
        let base = read_config(source.join("config.json"))?;
        let mut selected = config.clone();
        selected.use_flash_attn = false;
        anyhow::ensure!((selected, conditional_generation) == base,
            "selected config differs from adapter original base config; reduce context with --max-context only");
        anyhow::ensure!(
            std::fs::read(source.join("tokenizer.json"))? == tokenizer_bytes,
            "selected tokenizer differs from adapter original base tokenizer"
        );
        Some(validate_adapter_metadata(
            state,
            &base,
            &read_config(path.join("source-config.json"))?,
            &read_config(path.join("config.json"))?,
            &tokenizer_bytes,
            &std::fs::read(path.join("tokenizer.json"))?,
        )?)
    } else {
        None
    };
    let filenames = match &args.weight_files {
        Some(files) => files.split(',').map(PathBuf::from).collect::<Vec<_>>(),
        None if source_checkpoint.is_some() => {
            checkpoint_files(source_checkpoint.as_ref().unwrap())?
        }
        None => {
            match candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json") {
                Ok(files) => files,
                Err(_) => vec![repo.get("model.safetensors")?],
            }
        }
    };
    if let Some(source) = &source_checkpoint {
        let canonical_files = |files: &[PathBuf]| -> Result<BTreeSet<PathBuf>> {
            files
                .iter()
                .map(|p| Ok(std::fs::canonicalize(p)?))
                .collect()
        };
        let selected = canonical_files(&filenames)?;
        anyhow::ensure!(
            selected.len() == filenames.len()
                && selected == canonical_files(&checkpoint_files(source)?)?,
            "adapter requires exactly the weight files of its original source_checkpoint"
        );
    }
    config.max_position_embeddings = runtime_context;
    let adapter_weights = args
        .adapter_checkpoint
        .as_ref()
        .map(|path| -> Result<_> {
            // Saved files must not be modified while mapped. No VarMap: inference stays frozen.
            let weights = unsafe {
                candle::safetensors::MmapedSafetensors::new(path.join("adapter.safetensors"))?
            };
            let shapes = adapter_shapes(&config, adapter_policy.unwrap().0);
            validate_adapter_tensors(weights.tensors(), &shapes)?;
            let state = adapter_state.as_ref().unwrap();
            anyhow::ensure!(
                state["quantized_projections"] == shapes.len() / 2
                    && state["trainable_parameters"]
                        == shapes
                            .values()
                            .map(|s| s.iter().product::<usize>())
                            .sum::<usize>(),
                "adapter tensor counts differ from training metadata"
            );
            Ok(weights)
        })
        .transpose()?;
    println!("retrieved the files in {:?}", start.elapsed());
    let preparation_seconds = start.elapsed().as_secs_f64();

    let device = candle_examples::device(args.cpu)?;
    device.synchronize()?;
    let start = Instant::now();
    let dtype = if adapter_policy.is_some() || device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let weights = unsafe { candle::safetensors::MmapedSafetensors::multi(&filenames)? };
    let mut seen = BTreeSet::new();
    for (name, _) in weights.tensors() {
        anyhow::ensure!(
            seen.insert(name.clone()),
            "duplicate checkpoint tensor {name}"
        );
    }
    // The mmap backend loads each requested tensor, not a retained dense GPU base copy.
    let vb = VarBuilder::from_backend(Box::new(weights), dtype, device.clone());

    let mut model = match (adapter_policy.is_some(), conditional_generation) {
        (true, true) => TextModel::new_quantized_for_conditional_generation(
            &config,
            vb,
            candle::quantized::GgmlDType::Q4_0,
        )?,
        (true, false) => TextModel::new_quantized(&config, vb, candle::quantized::GgmlDType::Q4_0)?,
        (false, true) => TextModel::new_for_conditional_generation(&config, vb)?,
        (false, false) => TextModel::new(&config, vb)?,
    };
    if let (Some((rank, alpha)), Some(weights)) = (adapter_policy, adapter_weights) {
        let count = model.add_quantized_lora(
            rank,
            alpha,
            VarBuilder::from_backend(Box::new(weights), DType::F32, device.clone()),
        )?;
        anyhow::ensure!(
            count == adapter_shapes(&config, rank).len() / 2,
            "adapter target count mismatch"
        );
    }

    device.synchronize()?;
    let model_load_seconds = start.elapsed().as_secs_f64();
    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    let probe = args.probe_prompt_tokens.is_some();
    let mut report = pipeline.run(tokens, args.sample_len, probe, args.ignore_eos)?;
    anyhow::ensure!(
        !probe || report["cached_decode_tested"] == true,
        "probe stopped before cached decode; retry with --ignore-eos (no success report written)"
    );
    report["format_version"] = json!(1);
    report["status"] = json!("success");
    report["mode"] = json!(if probe { "capacity-probe" } else { "normal" });
    report["declared_context"] = json!(declared_context);
    report["runtime_context"] = json!(runtime_context);
    report["ignore_eos"] = json!(args.ignore_eos);
    report["model_id"] = json!(args.model_id);
    report["revision"] = json!(args.revision);
    report["config_file"] = json!(config_filename);
    report["tokenizer_file"] = json!(tokenizer_filename);
    report["weight_files"] = json!(filenames);
    report["adapter_checkpoint"] = json!(args.adapter_checkpoint);
    report["adapter_metadata"] = json!(adapter_state);
    report["device"] = json!(format!("{device:?}"));
    report["compute_dtype"] = json!(dtype.as_str());
    report["use_flash_attn"] = json!(args.use_flash_attn);
    report["force_dmmv"] = json!(args.force_dmmv);
    report["prompt"] = json!(args.prompt);
    report["raw_prompt"] = json!(args.raw_prompt);
    report["seed_prompt_token_ids"] = json!(seed_tokens);
    report["synthetic_prompt_method"] = if probe {
        json!("repeat seed token IDs, preserving initial BOS once when seed has other tokens; truncate to exact size")
    } else {
        Value::Null
    };
    report["sampling"] = json!({"seed": args.seed, "temperature": args.temperature.unwrap_or(0.),
        "top_p": args.top_p, "top_k": args.top_k, "repeat_penalty": args.repeat_penalty, "repeat_last_n": args.repeat_last_n});
    report["timings_seconds"]["preparation"] = json!(preparation_seconds);
    report["timings_seconds"]["model_load"] = json!(model_load_seconds);
    if let Some(path) = &args.report_file {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("writing report {}", path.display()))?;
        println!("report written to {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokenizers::{models::wordlevel::WordLevel, processors::template::TemplateProcessing};

    fn dense_config() -> serde_json::Value {
        json!({
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 8,
            "query_pre_attn_scalar": 4,
            "vocab_size": 16,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "attention_k_eq_v": true,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "enable_moe_block": false,
            "use_bidirectional_attention": "vision"
        })
    }

    #[test]
    fn extracts_both_checkpoint_layouts_and_propagates_flash() -> Result<()> {
        let flat = dense_config();
        let nested = json!({"model_type": "gemma4", "text_config": flat.clone()});
        for flash in [false, true] {
            let (standalone, conditional) = parse_text_config(&serde_json::to_vec(&flat)?, flash)?;
            assert!(!conditional);
            let (wrapped, conditional) = parse_text_config(&serde_json::to_vec(&nested)?, flash)?;
            assert!(conditional);
            assert_eq!(standalone, wrapped);
            assert_eq!(wrapped.use_flash_attn, flash);
        }
        Ok(())
    }

    #[test]
    fn rejects_unsupported_configs_in_both_layouts() -> Result<()> {
        for (field, value) in [
            ("hidden_size_per_layer_input", json!(8)),
            ("num_kv_shared_layers", json!(1)),
            ("enable_moe_block", json!(true)),
            ("model_type", json!("gemma4_unified_text")),
        ] {
            let mut flat = dense_config();
            flat[field] = value;
            let nested = json!({"model_type": "gemma4", "text_config": flat.clone()});
            for config in [flat, nested] {
                let err = parse_text_config(&serde_json::to_vec(&config)?, false).unwrap_err();
                assert!(format!("{err:#}").contains(field), "{err:#}");
            }
        }
        for config in [
            json!({"model_type": "gemma4"}),
            json!({"model_type": "gemma4_text", "text_config": dense_config()}),
            json!({"model_type": "gemma4_unified", "text_config": dense_config()}),
            json!({"model_type": "gemma4", "text_config": null}),
        ] {
            assert!(parse_text_config(&serde_json::to_vec(&config)?, false).is_err());
        }
        Ok(())
    }

    fn tiny_tokenizer(specials: &[&str]) -> Result<Tokenizer> {
        let model = WordLevel::builder()
            .vocab([("[UNK]".to_owned(), 0)].into_iter().collect())
            .unk_token("[UNK]".to_owned())
            .build()
            .map_err(E::msg)?;
        let mut tokenizer = Tokenizer::new(model);
        let specials = specials
            .iter()
            .map(|&token| tokenizers::AddedToken::from(token, true))
            .collect::<Vec<_>>();
        tokenizer.add_special_tokens(specials).map_err(E::msg)?;
        Ok(tokenizer)
    }

    #[test]
    fn formats_single_turn_without_thinking_and_preserves_raw_prompt() {
        assert_eq!(
            format_prompt("  Hello!\n", false),
            "<bos><|turn>user\nHello!<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
        let raw = "<bos>  Already formatted\n";
        assert_eq!(format_prompt(raw, true), raw);
    }

    #[test]
    fn prompt_encoding_does_not_add_duplicate_bos() -> Result<()> {
        let mut tokenizer = tiny_tokenizer(&[
            "<bos>",
            "<eos>",
            "<|turn>",
            "<turn|>",
            "<|channel>",
            "<channel|>",
        ])?;
        let bos = tokenizer.token_to_id("<bos>").unwrap();
        let processor = TemplateProcessing::builder()
            .try_single("<bos> $A")
            .map_err(E::msg)?
            .special_tokens(vec![("<bos>", bos)])
            .build()
            .map_err(E::msg)?;
        tokenizer.with_post_processor(Some(processor));
        for (prompt, raw) in [("Hello!", false), ("<bos>Hello!", true)] {
            let tokens = prompt_tokens(&tokenizer, prompt, raw)?;
            assert_eq!(tokens[0], bos);
            assert_eq!(tokens.iter().filter(|&&id| id == bos).count(), 1);
        }
        assert!(!prompt_tokens(&tokenizer, "Hello!", true)?.contains(&bos));
        assert!(prompt_tokens(&tokenizer, "", true).is_err());
        Ok(())
    }

    #[test]
    fn stops_on_eos_turn_end_and_tool_handoff_not_legacy_eos() -> Result<()> {
        let tokenizer = tiny_tokenizer(&["<eos>", "<turn|>", "<|tool_response>", "</s>"])?;
        let stops = stop_tokens(&tokenizer)?;
        for token in ["<eos>", "<turn|>", "<|tool_response>"] {
            assert!(stops.contains(&tokenizer.token_to_id(token).unwrap()));
        }
        assert!(!stops.contains(&tokenizer.token_to_id("</s>").unwrap()));
        for (specials, missing) in [
            (["</s>", "<turn|>", "<|tool_response>"], "<eos>"),
            (["<eos>", "</s>", "<|tool_response>"], "<turn|>"),
            (["<eos>", "<turn|>", "</s>"], "<|tool_response>"),
        ] {
            let err = stop_tokens(&tiny_tokenizer(&specials)?).unwrap_err();
            assert!(err.to_string().contains(missing));
        }
        Ok(())
    }

    #[test]
    fn requires_explicit_model_selection() {
        let error = Args::try_parse_from(["gemma4", "--prompt", "Hello!"]).unwrap_err();
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
        assert!(error.to_string().contains("--model-id"));
        let args = Args::try_parse_from([
            "gemma4",
            "--prompt",
            "Hello!",
            "--model-id",
            "google/gemma-4-31B-it",
        ])
        .unwrap();
        assert_eq!(args.model_id, "google/gemma-4-31B-it");
    }

    #[test]
    fn validates_capacity_probe_cli() -> Result<()> {
        let parse = |extra: &[&str]| {
            Args::try_parse_from(
                ["gemma4", "--model-id", "local/base", "--prompt", "Hello"]
                    .into_iter()
                    .chain(extra.iter().copied()),
            )
        };
        assert!(parse(&["--ignore-eos"]).is_err());
        for extra in [
            vec!["--max-context", "0"],
            vec!["--probe-prompt-tokens", "0"],
            vec!["--probe-prompt-tokens", "4", "--sample-len", "0"],
            vec!["--probe-prompt-tokens", "4", "--sample-len", "1"],
            vec!["--multimodal"],
        ] {
            assert!(parse(&extra)?.validate().is_err(), "{extra:?}");
        }
        for extra in [
            vec![],
            vec!["--sample-len", "0"],
            vec![
                "--adapter-checkpoint",
                "/saved/checkpoint-2",
                "--max-context",
                "128",
                "--probe-prompt-tokens",
                "126",
                "--sample-len",
                "2",
                "--ignore-eos",
                "--report-file",
                "/tmp/report.json",
            ],
        ] {
            parse(&extra)?.validate()?;
        }
        assert!(parse(&["--sample-len", "-1"]).is_err());
        Ok(())
    }

    #[test]
    fn context_budget_is_conservative_downward_only_and_overflow_safe() -> Result<()> {
        assert_eq!(context_budget(128, None, 126, 2)?, 128);
        assert_eq!(context_budget(128, Some(16), 14, 2)?, 16);
        assert_eq!(context_budget(128, None, 128, 0)?, 128);
        for (declared, cap, prompt, sample) in [
            (128, Some(129), 1, 2),
            (128, Some(0), 1, 2),
            (128, Some(16), 15, 2),
            (128, None, 127, 2),
            (128, None, 0, 2),
            (usize::MAX, None, usize::MAX, 1),
        ] {
            assert!(context_budget(declared, cap, prompt, sample).is_err());
        }
        Ok(())
    }

    #[test]
    fn synthetic_prompt_has_exact_reproducible_size_and_preserves_bos() -> Result<()> {
        let seed = [1, 4, 7];
        for n in [1, 2, 3, 4, 17, 4096] {
            let tokens = synthetic_prompt(&seed, n, Some(1))?;
            assert_eq!(tokens.len(), n);
            assert_eq!(tokens[0], 1);
            assert_eq!(tokens.iter().filter(|&&id| id == 1).count(), 1);
            assert_eq!(tokens, synthetic_prompt(&seed, n, Some(1))?);
            assert_eq!(
                &tokens[1..],
                [4, 7].into_iter().cycle().take(n - 1).collect::<Vec<_>>()
            );
        }
        assert_eq!(synthetic_prompt(&seed, 5, None)?, [1, 4, 7, 1, 4]);
        assert_eq!(synthetic_prompt(&[1], 3, Some(1))?, [1, 1, 1]);
        assert_eq!(synthetic_prompt(&[4, 7], 3, Some(1))?, [4, 7, 4]);
        assert!(synthetic_prompt(&[], 3, Some(1)).is_err());
        assert!(synthetic_prompt(&seed, 0, Some(1)).is_err());
        Ok(())
    }

    #[test]
    fn adapter_metadata_rejects_policy_architecture_and_tokenizer_mismatches() -> Result<()> {
        let base = parse_text_config(&serde_json::to_vec(&dense_config())?, false)?;
        let mut trained = base.clone();
        trained.0.max_position_embeddings = 8;
        let state = json!({"format_version": 1, "mode": "quantized-lora", "quantization": "Q4_0",
            "dtype": "bf16", "master_dtype": "f32", "source_checkpoint": "/base",
            "lora_rank": 2, "lora_alpha": 4., "sequence_length": 8});
        let check = |state: &Value,
                     source: &(Gemma4TextConfig, bool),
                     trained: &(Gemma4TextConfig, bool),
                     tokenizer: &[u8]| {
            validate_adapter_metadata(state, &base, source, trained, b"tokenizer", tokenizer)
        };
        assert_eq!(check(&state, &base, &trained, b"tokenizer")?, (2, 4.));
        for (key, value) in [
            ("format_version", json!(2)),
            ("mode", json!("quantized-head")),
            ("quantization", json!("Q8_0")),
            ("dtype", json!("f32")),
            ("master_dtype", json!("bf16")),
            ("source_checkpoint", Value::Null),
            ("lora_rank", json!(0)),
            ("lora_rank", json!(-1)),
            ("lora_alpha", json!(0)),
            ("lora_alpha", Value::Null),
            ("sequence_length", json!(16)),
        ] {
            let mut bad = state.clone();
            bad[key] = value;
            assert!(check(&bad, &base, &trained, b"tokenizer").is_err(), "{key}");
        }
        assert!(check(&state, &base, &trained, b"other tokenizer").is_err());
        let mut bad = trained.clone();
        bad.0.hidden_size *= 2;
        assert!(check(&state, &base, &bad, b"tokenizer").is_err());
        assert!(check(&state, &bad, &trained, b"tokenizer").is_err());
        bad = trained.clone();
        bad.1 = !bad.1;
        assert!(check(&state, &base, &bad, b"tokenizer").is_err());
        let mut shorter_source = base.clone();
        shorter_source.0.max_position_embeddings = 4;
        assert!(check(&state, &shorter_source, &trained, b"tokenizer").is_err());
        Ok(())
    }

    #[test]
    fn adapter_tensors_require_exact_names_shapes_finite_f32_and_all_layers() -> Result<()> {
        use safetensors::tensor::TensorView;
        let (config, _) = parse_text_config(&serde_json::to_vec(&dense_config())?, false)?;
        let expected = adapter_shapes(&config, 2);
        assert_eq!(expected.len(), 26);
        assert!(!expected.contains_key("layers.1.self_attn.v_proj.lora_a"));
        assert_eq!(expected["layers.1.self_attn.o_proj.lora_a"], [2, 16]);
        let data: BTreeMap<_, _> = expected
            .iter()
            .map(|(name, shape)| (name.clone(), vec![0u8; shape.iter().product::<usize>() * 4]))
            .collect();
        let views = || -> Result<Vec<_>> {
            expected
                .iter()
                .map(|(name, shape)| {
                    Ok((
                        name.clone(),
                        TensorView::new(safetensors::Dtype::F32, shape.clone(), &data[name])?,
                    ))
                })
                .collect()
        };
        validate_adapter_tensors(views()?, &expected)?;
        let mut missing = views()?;
        missing.pop();
        assert!(validate_adapter_tensors(missing, &expected).is_err());
        let mut duplicate = views()?;
        duplicate.push(duplicate[0].clone());
        assert!(validate_adapter_tensors(duplicate, &expected).is_err());
        let mut wrong = views()?;
        wrong[0].0 = "lm_head.lora_a".into();
        assert!(validate_adapter_tensors(wrong, &expected).is_err());
        let name = expected.keys().next().unwrap();
        for (dtype, shape) in [
            (safetensors::Dtype::BF16, vec![data[name].len() / 2]),
            (safetensors::Dtype::F32, vec![data[name].len() / 4]),
        ] {
            let mut wrong = views()?;
            wrong[0].1 = TensorView::new(dtype, shape, &data[name])?;
            assert!(validate_adapter_tensors(wrong, &expected).is_err());
        }
        let mut nonfinite = data[name].clone();
        nonfinite[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut wrong = views()?;
        wrong[0].1 = TensorView::new(safetensors::Dtype::F32, expected[name].clone(), &nonfinite)?;
        assert!(validate_adapter_tensors(wrong, &expected).is_err());
        Ok(())
    }

    #[test]
    fn frozen_q4_bf16_adapters_prefill_cached_decode_and_report_positions() -> Result<()> {
        let mut raw = dense_config();
        for (key, value) in [
            ("hidden_size", 32),
            ("intermediate_size", 64),
            ("head_dim", 32),
            ("global_head_dim", 32),
            ("max_position_embeddings", 264),
        ] {
            raw[key] = json!(value);
        }
        let (config, _) = parse_text_config(&serde_json::to_vec(&raw)?, false)?;
        let vars = candle_nn::VarMap::new();
        TextModel::new(
            &config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )?;
        let weights = vars
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| {
                let values: Vec<f32> = (0..var.elem_count())
                    .map(|i| {
                        if var.rank() == 2 {
                            ((i % 17) as f32 - 8.) * 0.01
                        } else {
                            1.
                        }
                    })
                    .collect();
                Ok((
                    name.clone(),
                    Tensor::from_vec(values, var.shape(), &Device::Cpu)?.to_dtype(DType::BF16)?,
                ))
            })
            .collect::<Result<std::collections::HashMap<_, _>>>()?;
        let mut model = TextModel::new_quantized(
            &config,
            VarBuilder::from_tensors(weights, DType::BF16, &Device::Cpu),
            candle::quantized::GgmlDType::Q4_0,
        )?;
        let shapes = adapter_shapes(&config, 2);
        // Serialize nonzero saved adapters and reload through a frozen safetensors backend.
        let adapters = shapes
            .iter()
            .map(|(name, shape)| {
                Ok((
                    name.clone(),
                    Tensor::full(0.02f32, shape.as_slice(), &Device::Cpu)?,
                ))
            })
            .collect::<Result<std::collections::HashMap<_, _>>>()?;
        let bytes = safetensors::serialize(adapters.iter().map(|(n, t)| (n.as_str(), t)), None)?;
        let saved = candle::safetensors::SliceSafetensors::new(&bytes)?;
        validate_adapter_tensors(saved.tensors(), &shapes)?;
        assert_eq!(
            model.add_quantized_lora(
                2,
                4.,
                VarBuilder::from_slice_safetensors(&bytes, DType::F32, &Device::Cpu)?
            )?,
            13
        );
        let logits = model.forward(&Tensor::new(&[[1u32, 4, 7]], &Device::Cpu)?, 0)?;
        assert!(!logits.track_op());
        assert!(logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));
        let long_prompt = vec![4u32; 261];
        model.clear_kv_cache();
        let full = model.forward(
            &Tensor::new(&long_prompt[..], &Device::Cpu)?.unsqueeze(0)?,
            0,
        )?;
        model.clear_kv_cache();
        model.forward(
            &Tensor::new(&long_prompt[..256], &Device::Cpu)?.unsqueeze(0)?,
            0,
        )?;
        let chunked = model.forward(
            &Tensor::new(&long_prompt[256..], &Device::Cpu)?.unsqueeze(0)?,
            256,
        )?;
        let error = (full.to_dtype(DType::F32)? - chunked.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(error < 0.02, "Q4/BF16 chunked prefill error: {error}");
        let tokenizer = tiny_tokenizer(&["<bos>", "<eos>", "<turn|>", "<|tool_response>"])?;
        let mut pipeline =
            TextGeneration::new(model, tokenizer, 42, None, None, None, 1., 64, &Device::Cpu);
        let report = pipeline.run(vec![1, 4, 7, 4, 7], 3, true, true)?;
        assert_eq!(report["prompt_tokens"], 5);
        assert_eq!(report["generated_tokens"], 3);
        assert_eq!(report["forwarded_tokens"], 7);
        assert_eq!(report["max_forwarded_position_zero_based"], 6);
        assert_eq!(report["cached_decode_steps"], 2);
        assert_eq!(report["cached_decode_tested"], true);
        assert!(
            report["timings_seconds"]["prefill_forward"]
                .as_f64()
                .unwrap()
                > 0.
        );
        assert!(
            report["timings_seconds"]["cached_decode_forward"]
                .as_f64()
                .unwrap()
                > 0.
        );
        let report = pipeline.run(long_prompt, 3, true, true)?;
        assert_eq!(report["prompt_tokens"], 261);
        assert_eq!(report["generated_tokens"], 3);
        assert_eq!(report["forwarded_tokens"], 263);
        assert_eq!(report["max_forwarded_position_zero_based"], 262);
        assert_eq!(report["cached_decode_steps"], 2);
        // Subsequent runs reset cache, and zero samples forward nothing.
        let report = pipeline.run(vec![1], 0, false, false)?;
        assert_eq!(report["forwarded_tokens"], 0);
        assert_eq!(report["max_forwarded_position_zero_based"], Value::Null);
        let report = pipeline.run(vec![1], 1, false, false)?;
        assert_eq!(report["forwarded_tokens"], 1);
        assert_eq!(report["max_forwarded_position_zero_based"], 0);
        assert_eq!(report["cached_decode_tested"], false);
        Ok(())
    }
}
