use anyhow::{bail, Context, Result};
use burn::backend::Wgpu;
use burn::prelude::ElementConversion;
use burn::tensor::{Int, Tensor, TensorData};
use std::path::{Path, PathBuf};

use voxtral_mini_realtime::audio::{
    chunk::{chunk_audio, needs_chunking, AudioChunk, ChunkConfig},
    mel::{MelConfig, MelSpectrogram},
    pad::{pad_audio, PadConfig},
    resample::resample_to_16k,
    AudioBuffer,
};
use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;
use voxtral_mini_realtime::tokenizer::VoxtralTokenizer;

type Backend = Wgpu;
type Device = <Backend as burn::tensor::backend::Backend>::Device;

/// Q4 GGUF model hosted by the voxtral-mini-realtime-rs author.
const GGUF_REPO: &str = "TrevorJS/voxtral-mini-realtime-gguf";
const GGUF_FILENAME: &str = "voxtral-q4.gguf";
const TOKENIZER_FILENAME: &str = "tekken.json";

#[allow(clippy::large_enum_variant)]
enum Model {
    F32(voxtral_mini_realtime::models::voxtral::VoxtralModel<Backend>),
    Q4(voxtral_mini_realtime::gguf::model::Q4VoxtralModel),
}

pub struct TranscriptionEngine {
    model: Model,
    tokenizer: VoxtralTokenizer,
    mel_extractor: MelSpectrogram,
    pad_config: PadConfig,
    chunk_config: ChunkConfig,
    time_embedding: TimeEmbedding,
    device: Device,
}

/// F32 SafeTensors from the official Mistral repo.
const F32_REPO: &str = "mistralai/Voxtral-Mini-4B-Realtime-2602";
const F32_WEIGHTS: &str = "consolidated.safetensors";
const F32_CONFIG: &str = "params.json";

/// Download Q4 GGUF model + tokenizer from HuggingFace.
///
/// Uses hf-hub's built-in cache — files are only downloaded once.
pub fn download_model_q4() -> Result<(PathBuf, PathBuf)> {
    let api = hf_hub::api::sync::Api::new().context("failed to create HuggingFace API")?;
    let repo = api.model(GGUF_REPO.to_string());

    let gguf_path = repo
        .get(GGUF_FILENAME)
        .context("failed to download Q4 GGUF model")?;
    let tokenizer_path = repo
        .get(TOKENIZER_FILENAME)
        .context("failed to download tokenizer")?;

    Ok((gguf_path, tokenizer_path))
}

/// Download F32 SafeTensors model from HuggingFace.
pub fn download_model_f32() -> Result<(PathBuf, PathBuf)> {
    let api = hf_hub::api::sync::Api::new().context("failed to create HuggingFace API")?;
    let repo = api.model(F32_REPO.to_string());

    let weights_path = repo
        .get(F32_WEIGHTS)
        .context("failed to download F32 model weights")?;
    // Also need config and tokenizer
    let _ = repo.get(F32_CONFIG);
    let tokenizer_path = repo
        .get(TOKENIZER_FILENAME)
        .context("failed to download tokenizer")?;

    Ok((weights_path, tokenizer_path))
}

fn hf_hub_dir() -> Option<PathBuf> {
    std::env::var_os("HF_HOME")
        .map(|h| PathBuf::from(h).join("hub"))
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
        })
}

/// Check whether the Q4 GGUF model is already in hf-hub's cache.
pub fn model_cached_q4() -> bool {
    hf_hub_dir()
        .map(|hub| {
            hub.join("models--TrevorJS--voxtral-mini-realtime-gguf")
                .join("refs")
                .exists()
        })
        .unwrap_or(false)
}

/// Check whether the F32 SafeTensors model is already in hf-hub's cache.
pub fn model_cached_f32() -> bool {
    hf_hub_dir()
        .map(|hub| {
            hub.join("models--mistralai--Voxtral-Mini-4B-Realtime-2602")
                .join("refs")
                .exists()
        })
        .unwrap_or(false)
}

impl TranscriptionEngine {
    pub fn load(model_path: Option<&Path>, use_f32: bool) -> Result<Self> {
        let device: Device = Default::default();

        if let Some(path) = model_path {
            Self::load_from_path(path, device)
        } else if use_f32 {
            let (weights_path, tokenizer_path) = download_model_f32()?;
            Self::load_f32(&weights_path, &tokenizer_path, device)
        } else {
            // Default: Q4 GGUF (4.2x faster decode, 703 MB vs 9.2 GB memory)
            let (gguf_path, tokenizer_path) = download_model_q4()?;
            Self::load_q4(&gguf_path, &tokenizer_path, device)
        }
    }

    fn build(model: Model, tokenizer_path: &Path, device: Device) -> Result<Self> {
        let tokenizer =
            VoxtralTokenizer::from_file(tokenizer_path).context("failed to load tokenizer")?;
        Ok(Self {
            model,
            tokenizer,
            mel_extractor: MelSpectrogram::new(MelConfig::voxtral()),
            pad_config: PadConfig::voxtral(),
            chunk_config: ChunkConfig::voxtral().with_max_frames(1200),
            time_embedding: TimeEmbedding::new(3072),
            device,
        })
    }

    fn load_q4(gguf_path: &Path, tokenizer_path: &Path, device: Device) -> Result<Self> {
        let mut loader = voxtral_mini_realtime::gguf::loader::Q4ModelLoader::from_file(gguf_path)
            .context("failed to open GGUF file")?;
        let model = Model::Q4(loader.load(&device).context("failed to load Q4 model")?);
        Self::build(model, tokenizer_path, device)
    }

    fn load_f32(weights_path: &Path, tokenizer_path: &Path, device: Device) -> Result<Self> {
        let loader =
            voxtral_mini_realtime::models::loader::VoxtralModelLoader::from_file(weights_path)
                .context("failed to open model weights")?;
        let model = Model::F32(loader.load(&device).context("failed to load F32 model")?);
        Self::build(model, tokenizer_path, device)
    }

    /// Load from a user-provided path: directory = F32 SafeTensors, .gguf = Q4.
    fn load_from_path(path: &Path, device: Device) -> Result<Self> {
        let is_gguf = path.extension().map(|ext| ext == "gguf").unwrap_or(false);

        if is_gguf {
            // Look for tokenizer next to the GGUF file, then fall back to hub
            let dir = path.parent().unwrap_or(Path::new("."));
            let tokenizer_path = if dir.join(TOKENIZER_FILENAME).exists() {
                dir.join(TOKENIZER_FILENAME)
            } else {
                let (_, tp) = download_model_q4()?;
                tp
            };
            Self::load_q4(path, &tokenizer_path, device)
        } else {
            // F32 SafeTensors directory or file
            let dir = if path.is_dir() {
                path.to_path_buf()
            } else {
                path.parent().unwrap_or(Path::new(".")).to_path_buf()
            };
            let weights_path = if path.is_dir() {
                dir.join("consolidated.safetensors")
            } else {
                path.to_path_buf()
            };
            let tokenizer_path = dir.join(TOKENIZER_FILENAME);
            Self::load_f32(&weights_path, &tokenizer_path, device)
        }
    }

    pub fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mut audio = AudioBuffer::new(samples.to_vec(), sample_rate);

        // Resample to 16kHz if needed
        if sample_rate != 16000 {
            audio = resample_to_16k(&audio).context("failed to resample audio")?;
        }

        // Normalize
        audio.peak_normalize(0.95);

        let t_embed = self.time_embedding.embed::<Backend>(6.0, &self.device);

        // Handle chunking for long audio
        let chunks = if needs_chunking(audio.samples.len(), &self.chunk_config) {
            chunk_audio(&audio.samples, &self.chunk_config)
        } else {
            vec![AudioChunk {
                samples: audio.samples.clone(),
                start_sample: 0,
                end_sample: audio.samples.len(),
                index: 0,
                is_last: true,
            }]
        };

        let mut texts = Vec::new();

        for chunk in &chunks {
            let chunk_audio = AudioBuffer::new(chunk.samples.clone(), audio.sample_rate);
            let mel_tensor = self.mel_tensor_from_audio(&chunk_audio)?;

            let generated = match &self.model {
                Model::Q4(model) => model.transcribe_streaming(mel_tensor, t_embed.clone()),
                Model::F32(model) => self.transcribe_f32(model, mel_tensor, t_embed.clone())?,
            };

            let text = self.decode_tokens(&generated)?;
            if !text.trim().is_empty() {
                texts.push(text.trim().to_string());
            }
        }

        Ok(texts.join(" "))
    }

    /// Build a mel spectrogram tensor from an audio buffer.
    ///
    /// `compute_log` returns `[n_frames][n_mels]` — we transpose to `[1, n_mels, n_frames]`
    /// for the model input.
    fn mel_tensor_from_audio(&self, audio: &AudioBuffer) -> Result<Tensor<Backend, 3>> {
        let padded = pad_audio(audio, &self.pad_config);
        let mel = self.mel_extractor.compute_log(&padded.samples);
        let n_frames = mel.len();
        let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };

        if n_frames == 0 {
            bail!("audio too short to produce mel frames");
        }

        // Transpose from [n_frames, n_mels] to [n_mels, n_frames]
        let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
        for (frame_idx, frame) in mel.iter().enumerate() {
            for (mel_idx, &val) in frame.iter().enumerate() {
                mel_transposed[mel_idx][frame_idx] = val;
            }
        }
        let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();

        Ok(Tensor::from_data(
            TensorData::new(mel_flat, [1, n_mels, n_frames]),
            &self.device,
        ))
    }

    /// Decode generated token IDs to text, filtering out control tokens.
    fn decode_tokens(&self, generated: &[i32]) -> Result<String> {
        let text_tokens: Vec<u32> = generated
            .iter()
            .filter(|&&t| t >= 1000)
            .map(|&t| t as u32)
            .collect();
        self.tokenizer
            .decode(&text_tokens)
            .context("failed to decode tokens")
    }

    /// F32 SafeTensors inference using manual autoregressive decoding.
    ///
    /// Based on the reference implementation in voxtral-mini-realtime-rs/src/bin/transcribe.rs.
    fn transcribe_f32(
        &self,
        model: &voxtral_mini_realtime::models::voxtral::VoxtralModel<Backend>,
        mel_tensor: Tensor<Backend, 3>,
        t_embed: Tensor<Backend, 3>,
    ) -> Result<Vec<i32>> {
        let audio_embeds = model.encode_audio(mel_tensor);
        let seq_len = audio_embeds.dims()[1];
        let d_model = audio_embeds.dims()[2];

        const PREFIX_LEN: usize = 38;
        const BOS_TOKEN: i32 = 1;
        const STREAMING_PAD: i32 = 32;

        if seq_len < PREFIX_LEN {
            return Ok(Vec::new());
        }

        let mut decoder_cache = model.create_decoder_cache_preallocated(seq_len, &self.device);

        // Build prefix: [BOS, PAD, PAD, ..., PAD] of length PREFIX_LEN
        let mut prefix: Vec<i32> = vec![BOS_TOKEN];
        prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

        let prefix_tensor = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(prefix.clone(), [1, PREFIX_LEN]),
            &self.device,
        );
        let prefix_text_embeds = model.decoder().embed_tokens(prefix_tensor);

        let prefix_audio = audio_embeds
            .clone()
            .slice([0..1, 0..PREFIX_LEN, 0..d_model]);

        let prefix_inputs = prefix_audio + prefix_text_embeds;
        let hidden = model.decoder().forward_hidden_with_cache(
            prefix_inputs,
            t_embed.clone(),
            &mut decoder_cache,
        );
        let logits = model.decoder().lm_head(hidden);

        let vocab_size = logits.dims()[2];
        let last_logits = logits.slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab_size]);
        let first_pred = last_logits.argmax(2);
        let first_token: i32 = first_pred.into_scalar().elem();

        let mut generated = prefix;
        generated.push(first_token);

        // Autoregressive generation for remaining positions
        for pos in PREFIX_LEN + 1..seq_len {
            let new_token = generated[pos - 1];
            let token_tensor = Tensor::<Backend, 2, Int>::from_data(
                TensorData::new(vec![new_token], [1, 1]),
                &self.device,
            );
            let text_embed = model.decoder().embed_tokens(token_tensor);

            let audio_pos = audio_embeds
                .clone()
                .slice([0..1, (pos - 1)..pos, 0..d_model]);

            let input = audio_pos + text_embed;
            let hidden = model.decoder().forward_hidden_with_cache(
                input,
                t_embed.clone(),
                &mut decoder_cache,
            );
            let logits = model.decoder().lm_head(hidden);

            let pred = logits.argmax(2);
            let next_token: i32 = pred.into_scalar().elem();
            generated.push(next_token);
        }

        Ok(generated.into_iter().skip(PREFIX_LEN).collect())
    }
}
