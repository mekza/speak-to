use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

pub struct AudioCapture {
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
    stream: Option<cpal::Stream>,
    recording: Arc<AtomicBool>,
}

impl AudioCapture {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("no input device available")?;
        let config = device
            .default_input_config()
            .context("failed to get default input config")?;

        Ok(Self {
            device,
            config,
            buffer: Arc::new(Mutex::new(Vec::new())),
            stream: None,
            recording: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn start(&mut self) -> Result<()> {
        let buffer = self.buffer.clone();
        let recording = self.recording.clone();
        recording.store(true, Ordering::SeqCst);

        let channels = self.config.channels() as usize;

        let err_fn = |err| eprintln!("audio stream error: {}", err);

        let stream = match self.config.sample_format() {
            cpal::SampleFormat::F32 => {
                let buffer = buffer.clone();
                let recording = recording.clone();
                self.device.build_input_stream(
                    &self.config.clone().into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if !recording.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut buf = buffer.lock().unwrap();
                        // Mix to mono by averaging channels
                        for chunk in data.chunks(channels) {
                            let sample: f32 = chunk.iter().sum::<f32>() / channels as f32;
                            buf.push(sample);
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            cpal::SampleFormat::I16 => {
                let buffer = buffer.clone();
                let recording = recording.clone();
                self.device.build_input_stream(
                    &self.config.clone().into(),
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        if !recording.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut buf = buffer.lock().unwrap();
                        for chunk in data.chunks(channels) {
                            let sample: f32 = chunk
                                .iter()
                                .map(|&s| s as f32 / i16::MAX as f32)
                                .sum::<f32>()
                                / channels as f32;
                            buf.push(sample);
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            format => anyhow::bail!("unsupported sample format: {:?}", format),
        };

        stream.play().context("failed to start audio stream")?;
        self.stream = Some(stream);
        Ok(())
    }

    pub fn stop(&mut self) -> Result<(Vec<f32>, u32)> {
        self.recording.store(false, Ordering::SeqCst);
        // Drop the stream to stop recording
        self.stream = None;

        let samples = std::mem::take(&mut *self.buffer.lock().unwrap());
        let sample_rate = self.config.sample_rate();

        Ok((samples, sample_rate))
    }
}
