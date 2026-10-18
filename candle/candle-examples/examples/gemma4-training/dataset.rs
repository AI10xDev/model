//! JSONL preprocessing and headerless little-endian u32 token streams.
//!
//! Documents are shuffled and split before packing. Each split concatenates its
//! documents (optional BOS, text, EOS), then keeps `windows * sequence_length + 1`
//! tokens. Windows have length `sequence_length + 1` and stride `sequence_length`,
//! preserving every next-token target across window boundaries without padding.

use anyhow::{ensure, Context, Result};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Debug, clap::Args)]
pub struct PrepareArgs {
    /// JSONL file, with a required nonempty string `text` in every record.
    #[arg(long)]
    pub input: PathBuf,
    /// The actual tokenizer.json used by the model.
    #[arg(long)]
    pub tokenizer: PathBuf,
    /// New output directory; an existing directory is never overwritten.
    #[arg(long)]
    pub output: PathBuf,
    /// Number of next-token targets per window.
    #[arg(long, default_value_t = 512)]
    pub sequence_length: usize,
    /// Model vocabulary size (all token IDs must be strictly smaller).
    #[arg(long)]
    pub vocab_size: usize,
    /// Fraction of documents held out, rounded down; both splits must be usable.
    #[arg(long, default_value_t = 0.1)]
    pub validation_fraction: f64,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Optional explicit BOS prepended to each document.
    #[arg(long)]
    pub bos_token_id: Option<u32>,
    /// Explicit EOS appended to each document.
    #[arg(long)]
    pub eos_token_id: u32,
}

#[derive(Debug, Serialize)]
struct SplitStats {
    documents: usize,
    tokens_before_packing: usize,
    stored_tokens: usize,
    windows: usize,
    dropped_tail_tokens: usize,
}

fn validate_dimensions(sequence_length: usize, vocab_size: usize) -> Result<usize> {
    ensure!(sequence_length >= 1, "sequence_length must be at least 1");
    ensure!(vocab_size >= 1, "vocab_size must be at least 1");
    sequence_length
        .checked_add(1)
        .context("sequence_length + 1 overflows usize")
}

fn split_documents(
    mut documents: Vec<Vec<u32>>,
    validation_fraction: f64,
    seed: u64,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
    ensure!(
        validation_fraction > 0.0 && validation_fraction < 1.0,
        "validation_fraction must be finite and strictly between 0 and 1"
    );
    let validation_count = (documents.len() as f64 * validation_fraction).floor() as usize;
    ensure!(
        validation_count > 0 && validation_count < documents.len(),
        "document split is empty: {} documents and validation_fraction={validation_fraction} \
         give {validation_count} validation documents (floor); provide more documents or change the fraction",
        documents.len()
    );
    documents.shuffle(&mut StdRng::seed_from_u64(seed));
    let train = documents.split_off(validation_count);
    Ok((train, documents))
}

fn pack(
    documents: Vec<Vec<u32>>,
    sequence_length: usize,
    name: &str,
) -> Result<(Vec<u32>, SplitStats)> {
    let document_count = documents.len();
    let mut tokens: Vec<u32> = documents.into_iter().flatten().collect();
    let windows = tokens.len().saturating_sub(1) / sequence_length;
    ensure!(
        windows > 0,
        "{name} split has no usable window: {} tokens, need at least {} (sequence_length + 1)",
        tokens.len(),
        sequence_length + 1
    );
    let stored_tokens = windows * sequence_length + 1;
    let stats = SplitStats {
        documents: document_count,
        tokens_before_packing: tokens.len(),
        stored_tokens,
        windows,
        dropped_tail_tokens: tokens.len() - stored_tokens,
    };
    tokens.truncate(stored_tokens);
    Ok((tokens, stats))
}

pub fn prepare(args: PrepareArgs) -> Result<()> {
    validate_dimensions(args.sequence_length, args.vocab_size)?;
    ensure!(
        args.validation_fraction > 0.0 && args.validation_fraction < 1.0,
        "validation_fraction must be finite and strictly between 0 and 1"
    );
    for (name, id) in [("BOS", args.bos_token_id), ("EOS", Some(args.eos_token_id))] {
        if let Some(id) = id {
            ensure!(
                (id as u64) < args.vocab_size as u64,
                "{name} token ID {id} is outside vocab_size {}",
                args.vocab_size
            );
        }
    }
    ensure!(
        !args.output.try_exists()?,
        "output directory already exists: {}",
        args.output.display()
    );
    let tokenizer_bytes = fs::read(&args.tokenizer)
        .with_context(|| format!("reading tokenizer {}", args.tokenizer.display()))?;
    let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("loading tokenizer {}", args.tokenizer.display()))?;
    // A saved inference tokenizer may truncate or pad; neither is appropriate
    // for document packing. Preserve the original bytes for provenance.
    tokenizer
        .with_truncation(None)
        .map_err(anyhow::Error::msg)?;
    tokenizer.with_padding(None);
    for (token, id) in tokenizer.get_vocab(true) {
        ensure!(
            (id as u64) < args.vocab_size as u64,
            "tokenizer token {token:?} has ID {id} outside vocab_size {}",
            args.vocab_size
        );
    }
    let input = File::open(&args.input)
        .with_context(|| format!("opening JSONL {}", args.input.display()))?;
    let mut documents = Vec::new();
    for (index, line) in BufReader::new(input).lines().enumerate() {
        let location = format!("{}: line {}", args.input.display(), index + 1);
        let line = line.with_context(|| format!("reading {location}"))?;
        let record: serde_json::Value = serde_json::from_str(&line).with_context(|| {
            format!("{location}: expected a JSON object with string field `text`")
        })?;
        let text = record
            .get("text")
            .and_then(serde_json::Value::as_str)
            .with_context(|| {
                format!("{location}: expected a JSON object with string field `text`")
            })?;
        ensure!(
            !text.trim().is_empty(),
            "{location}: `text` must not be empty or whitespace-only"
        );
        let encoding = tokenizer
            .encode(text, false)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("{location}: tokenization failed"))?;
        ensure!(
            !encoding.get_ids().is_empty(),
            "{location}: `text` tokenized to zero tokens"
        );
        for &id in encoding.get_ids() {
            ensure!(
                (id as u64) < args.vocab_size as u64,
                "{location}: token ID {id} is outside vocab_size {}",
                args.vocab_size
            );
        }
        let mut tokens = Vec::new();
        tokens.extend(args.bos_token_id);
        tokens.extend_from_slice(encoding.get_ids());
        tokens.push(args.eos_token_id);
        documents.push(tokens);
    }
    ensure!(
        !documents.is_empty(),
        "{}: no JSONL records",
        args.input.display()
    );
    let (train, valid) = split_documents(documents, args.validation_fraction, args.seed)?;
    let (train, train_stats) = pack(train, args.sequence_length, "train")?;
    let (valid, valid_stats) = pack(valid, args.sequence_length, "valid")?;
    let metadata = serde_json::json!({
        "format_version": 1,
        "token_format": "u32-le",
        "layout": "contiguous token stream, no header or padding",
        "sequence_length": args.sequence_length,
        "window_length": args.sequence_length + 1,
        "stride": args.sequence_length,
        "vocab_size": args.vocab_size,
        "bos_token_id": args.bos_token_id,
        "eos_token_id": args.eos_token_id,
        "seed": args.seed,
        "split_algorithm": "rand 0.9 StdRng seeded shuffle; first floor(documents * validation_fraction) documents are validation",
        "validation_fraction": args.validation_fraction,
        "input": fs::canonicalize(&args.input)?,
        "tokenizer": {
            "source": fs::canonicalize(&args.tokenizer)?,
            "copy": "tokenizer.json",
            "bytes": tokenizer_bytes.len(),
            "vocabulary_entries_including_added": tokenizer.get_vocab_size(true),
            "add_special_tokens": false,
            "truncation": false,
            "padding": false
        },
        "train": train_stats,
        "valid": valid_stats
    });
    // Do not create any output until all records and both splits pass validation.
    // create_dir (not create_dir_all) also prevents overwriting a concurrent run.
    fs::create_dir(&args.output)
        .with_context(|| format!("creating fresh output directory {}", args.output.display()))?;
    for (name, tokens) in [("train.bin", train), ("valid.bin", valid)] {
        let path = args.output.join(name);
        let mut output = BufWriter::new(File::create(&path)?);
        for token in tokens {
            output.write_all(&token.to_le_bytes())?;
        }
        output
            .flush()
            .with_context(|| format!("writing {}", path.display()))?;
    }
    fs::write(args.output.join("tokenizer.json"), tokenizer_bytes)?;
    fs::write(
        args.output.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&metadata)?);
    Ok(())
}

/// Load a complete packed token stream. Reject partial u32 values, incomplete
/// windows, empty datasets, and out-of-vocabulary IDs rather than dropping data.
/// Use the sequence length and vocabulary size recorded in metadata.json.
pub fn load(path: &Path, sequence_length: usize, vocab_size: usize) -> Result<Vec<Vec<u32>>> {
    let window_length = validate_dimensions(sequence_length, vocab_size)?;
    let bytes = fs::read(path).with_context(|| format!("reading dataset {}", path.display()))?;
    ensure!(
        bytes.len() % 4 == 0,
        "{}: truncated u32-le data: {} bytes is not divisible by 4",
        path.display(),
        bytes.len()
    );
    let mut tokens = Vec::with_capacity(bytes.len() / 4);
    for (index, bytes) in bytes.chunks_exact(4).enumerate() {
        let id = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        ensure!(
            (id as u64) < vocab_size as u64,
            "{}: token {index} has ID {id} outside vocab_size {vocab_size}",
            path.display()
        );
        tokens.push(id);
    }
    ensure!(
        tokens.len() >= window_length,
        "{}: no usable window: {} tokens, need at least {window_length}",
        path.display(),
        tokens.len()
    );
    ensure!(
        (tokens.len() - 1) % sequence_length == 0,
        "{}: incomplete final window: {} tokens for stride {sequence_length}; check sequence_length or truncated data",
        path.display(),
        tokens.len()
    );
    Ok(tokens
        .windows(window_length)
        .step_by(sequence_length)
        .map(<[u32]>::to_vec)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "gemma4-dataset-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn args(&self, input: &str) -> PrepareArgs {
            fs::write(self.0.join("input.jsonl"), input).unwrap();
            // TemplateProcessing would add BOS/EOS if encode(..., true) were used.
            let tokenizer = serde_json::json!({
                "version": "1.0", "truncation": null, "padding": null,
                "added_tokens": [], "normalizer": null,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<eos>", "type_id": 0}}],
                    "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                    "special_tokens": {
                        "<bos>": {"id": "<bos>", "ids": [3], "tokens": ["<bos>"]},
                        "<eos>": {"id": "<eos>", "ids": [4], "tokens": ["<eos>"]}
                    }
                },
                "decoder": null,
                "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "<bos>": 3, "<eos>": 4}, "unk_token": "[UNK]"}
            });
            fs::write(
                self.0.join("tokenizer.json"),
                serde_json::to_vec(&tokenizer).unwrap(),
            )
            .unwrap();
            PrepareArgs {
                input: self.0.join("input.jsonl"),
                tokenizer: self.0.join("tokenizer.json"),
                output: self.0.join("prepared"),
                sequence_length: 3,
                vocab_size: 5,
                validation_fraction: 0.5,
                seed: 42,
                bos_token_id: Some(3),
                eos_token_id: 4,
            }
        }

        fn tokens(&self, tokens: &[u32]) -> PathBuf {
            let path = self.0.join("tokens.bin");
            let bytes: Vec<_> = tokens
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect();
            fs::write(&path, bytes).unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn high_u32_ids_and_shift_boundaries() {
        let dir = TempDir::new();
        let ids = [0, 70_000, 262_143, u32::MAX - 1, 42];
        let path = dir.tokens(&ids);
        let windows = load(&path, 2, u32::MAX as usize).unwrap();
        assert_eq!(windows, vec![ids[..3].to_vec(), ids[2..].to_vec()]);
        let targets: Vec<_> = windows
            .iter()
            .flat_map(|w| w[1..].iter().copied())
            .collect();
        assert_eq!(targets, ids[1..]);
        assert!(load(&path, 2, 70_000)
            .unwrap_err()
            .to_string()
            .contains("outside vocab_size"));
        assert_eq!(load(&path, 1, u32::MAX as usize).unwrap().len(), 4);
    }

    #[test]
    fn deterministic_document_split_and_packing() {
        let documents: Vec<_> = (0..10).map(|id| vec![id; 4]).collect();
        let (train, valid) = split_documents(documents.clone(), 0.3, 19).unwrap();
        assert_eq!((train.len(), valid.len()), (7, 3));
        assert_eq!(
            (train.clone(), valid.clone()),
            split_documents(documents.clone(), 0.3, 19).unwrap()
        );
        assert_ne!(
            (train.clone(), valid.clone()),
            split_documents(documents, 0.3, 20).unwrap()
        );
        assert!(train
            .iter()
            .all(|doc| !valid.iter().any(|other| other[0] == doc[0])));
        let expected: Vec<_> = train.iter().flatten().copied().take(26).collect();
        let (tokens, stats) = pack(train, 5, "train").unwrap();
        assert_eq!(tokens, expected);
        assert_eq!(stats.windows, 5);
        assert_eq!(stats.dropped_tail_tokens, 2);
        assert_eq!(stats.tokens_before_packing, 28);
    }

    #[test]
    fn preprocessing_round_trip_and_provenance() {
        let dir = TempDir::new();
        let args = dir.args("{\"text\":\"hello world\"}\n{\"text\":\"hello world\"}\n");
        let output = args.output.clone();
        prepare(args).unwrap();
        for split in ["train", "valid"] {
            assert_eq!(
                load(&output.join(format!("{split}.bin")), 3, 5).unwrap(),
                vec![vec![3, 1, 2, 4]]
            );
        }
        assert_eq!(
            fs::read(output.join("tokenizer.json")).unwrap(),
            fs::read(dir.0.join("tokenizer.json")).unwrap()
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(metadata["seed"], 42);
        assert_eq!(metadata["train"]["dropped_tail_tokens"], 0);
        assert_eq!(metadata["valid"]["documents"], 1);
        let mut args = dir.args("{\"text\":\"hello world\"}\n{\"text\":\"hello world\"}\n");
        args.output = dir.0.join("second");
        prepare(args).unwrap();
        for name in ["train.bin", "valid.bin", "metadata.json", "tokenizer.json"] {
            assert_eq!(
                fs::read(output.join(name)).unwrap(),
                fs::read(dir.0.join("second").join(name)).unwrap()
            );
        }
        assert!(prepare(dir.args("{\"text\":\"hello\"}\n"))
            .unwrap_err()
            .to_string()
            .contains("already exists"));
    }

    #[test]
    fn malformed_and_empty_records_are_not_dropped() {
        for record in [
            "not-json",
            "{}",
            "[\"hello\"]",
            "{\"text\":null}",
            "{\"text\":5}",
            "{\"text\":\"\"}",
            "{\"text\":\"  \"}",
            "",
            "{\"text\":\"hello\"",
        ] {
            let dir = TempDir::new();
            let args = dir.args(&format!("{{\"text\":\"hello world\"}}\n{record}\n"));
            let output = args.output.clone();
            let error = format!("{:#}", prepare(args).unwrap_err());
            assert!(error.contains("line 2"), "{record:?}: {error}");
            assert!(!output.exists());
        }
        let dir = TempDir::new();
        assert!(prepare(dir.args(""))
            .unwrap_err()
            .to_string()
            .contains("no JSONL records"));
    }

    #[test]
    fn rejects_truncation_empty_data_and_bad_dimensions() {
        let dir = TempDir::new();
        let path = dir.tokens(&[1, 2, 3, 4]);
        assert!(load(&path, 2, 5)
            .unwrap_err()
            .to_string()
            .contains("incomplete final window"));
        fs::write(&path, [1, 0, 0]).unwrap();
        assert!(load(&path, 1, 5)
            .unwrap_err()
            .to_string()
            .contains("truncated u32-le"));
        let path = dir.tokens(&[]);
        assert!(load(&path, 1, 5)
            .unwrap_err()
            .to_string()
            .contains("no usable window"));
        assert!(load(&path, 0, 5).is_err());
        assert!(load(&path, usize::MAX, 5).is_err());
        assert!(load(&path, 1, 0).is_err());
    }

    #[test]
    fn preprocessing_keeps_high_special_ids_without_implicit_bos() {
        let dir = TempDir::new();
        let mut args = dir.args("{\"text\":\"hello world\"}\n{\"text\":\"hello world\"}\n");
        args.bos_token_id = None;
        args.eos_token_id = 262_143;
        args.vocab_size = 262_144;
        args.sequence_length = 2;
        let output = args.output.clone();
        prepare(args).unwrap();
        assert_eq!(
            load(&output.join("train.bin"), 2, 262_144).unwrap(),
            vec![vec![1, 2, 262_143]]
        );
    }

    #[test]
    fn saved_padding_and_truncation_do_not_drop_or_add_tokens() {
        let dir = TempDir::new();
        let args = dir.args("{\"text\":\"hello world\"}\n{\"text\":\"hello world\"}\n");
        let mut tokenizer: serde_json::Value =
            serde_json::from_slice(&fs::read(&args.tokenizer).unwrap()).unwrap();
        tokenizer["truncation"] = serde_json::json!({
            "direction": "Right", "max_length": 1, "strategy": "LongestFirst", "stride": 0
        });
        tokenizer["padding"] = serde_json::json!({
            "strategy": {"Fixed": 8}, "direction": "Right", "pad_to_multiple_of": null,
            "pad_id": 0, "pad_type_id": 0, "pad_token": "[UNK]"
        });
        fs::write(&args.tokenizer, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let output = args.output.clone();
        prepare(args).unwrap();
        assert_eq!(
            load(&output.join("train.bin"), 3, 5).unwrap(),
            vec![vec![3, 1, 2, 4]]
        );
    }

    #[test]
    fn rejects_zero_token_records_and_invalid_tokenizer_ids() {
        let dir = TempDir::new();
        let input = "{\"text\":\"hello world\"}\n{\"text\":\"hello world\"}\n";
        let args = dir.args(input);
        let mut tokenizer: serde_json::Value =
            serde_json::from_slice(&fs::read(&args.tokenizer).unwrap()).unwrap();
        tokenizer["normalizer"] = serde_json::json!({
            "type": "Replace", "pattern": {"String": "hello world"}, "content": ""
        });
        fs::write(&args.tokenizer, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let error = prepare(args).unwrap_err().to_string();
        assert!(
            error.contains("line 1") && error.contains("zero tokens"),
            "{error}"
        );
        let mut args = dir.args(input);
        args.vocab_size = 4;
        args.bos_token_id = None;
        args.eos_token_id = 3;
        assert!(prepare(args)
            .unwrap_err()
            .to_string()
            .contains("tokenizer token"));
        assert!(!dir.0.join("prepared").exists());
    }

    #[test]
    fn rejects_missing_usable_splits_and_invalid_specials() {
        for fraction in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(split_documents(vec![vec![1]; 10], fraction, 0).is_err());
        }
        assert!(split_documents(vec![vec![1]; 2], 0.1, 0).is_err());
        for name in ["train", "valid"] {
            assert!(pack(vec![vec![1, 2]], 2, name)
                .unwrap_err()
                .to_string()
                .contains(&format!("{name} split has no usable window")));
        }
        let dir = TempDir::new();
        let input = "{\"text\":\"hello world\"}\n{\"text\":\"hello world\"}\n";
        let mut args = dir.args(input);
        args.sequence_length = 4;
        assert!(prepare(args)
            .unwrap_err()
            .to_string()
            .contains("no usable window"));
        let mut args = dir.args(input);
        args.eos_token_id = 70_000;
        assert!(prepare(args)
            .unwrap_err()
            .to_string()
            .contains("EOS token ID"));
        let mut args = dir.args(input);
        args.bos_token_id = Some(70_000);
        assert!(prepare(args)
            .unwrap_err()
            .to_string()
            .contains("BOS token ID"));
        assert!(!dir.0.join("prepared").exists());
    }
}
