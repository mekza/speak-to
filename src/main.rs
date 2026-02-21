mod audio;
mod feedback;
mod interactive;
mod transcribe;

use anyhow::Result;
use clap::Parser;
use std::io::BufRead;

#[derive(Parser)]
#[command(
    name = "speak-to",
    about = "Voice input for CLI tools via local Voxtral transcription"
)]
struct Cli {
    /// Client CLI to launch with transcribed text (e.g., "claude", "kiro")
    client: Option<String>,

    /// Interactive mode: wrap client in PTY, Ctrl+Space for voice input
    #[arg(short = 'i', long)]
    interactive: bool,

    /// Additional arguments to pass to the client (after --)
    #[arg(last = true)]
    client_args: Vec<String>,

    /// Path to model file (safetensors directory or .gguf file)
    #[arg(long)]
    model_path: Option<std::path::PathBuf>,

    /// Use F32 full-precision model (~9GB, lower WER but needs more memory)
    #[arg(long)]
    f32: bool,

    /// Download/update model and exit
    #[arg(long)]
    download: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    // Ctrl+C: exit immediately
    ctrlc::set_handler(|| {
        eprintln!();
        std::process::exit(130);
    })?;

    let use_f32 = cli.f32;

    // --download: download model and exit
    if cli.download {
        if use_f32 {
            feedback::status("Downloading Voxtral Mini F32 (~9GB)...");
            transcribe::download_model_f32()?;
        } else {
            feedback::status("Downloading Voxtral Mini Q4 (~4GB)...");
            transcribe::download_model_q4()?;
        }
        feedback::success("Model downloaded and cached.");
        return Ok(());
    }

    // Download model if not cached (no spinner — hf-hub prints its own progress)
    if cli.model_path.is_none() {
        if use_f32 && !transcribe::model_cached_f32() {
            feedback::status("Downloading Voxtral Mini F32 (~9GB, first run only)...");
            transcribe::download_model_f32()?;
            feedback::success("Model downloaded.");
        } else if !use_f32 && !transcribe::model_cached_q4() {
            feedback::status("Downloading Voxtral Mini Q4 (~4GB, first run only)...");
            transcribe::download_model_q4()?;
            feedback::success("Model downloaded.");
        }
    }

    // Load model
    let label = if use_f32 { "F32" } else { "Q4" };
    let spinner = feedback::spinner(&format!("Loading Voxtral Mini {}...", label));
    let engine = transcribe::TranscriptionEngine::load(cli.model_path.as_deref(), use_f32)?;
    spinner.finish_and_clear();
    feedback::success(&format!("Model loaded ({}).", label));

    // Interactive mode: wrap client in PTY with voice input via Ctrl+Space
    if cli.interactive {
        let client = cli.client.as_deref().unwrap_or("claude");
        return interactive::run(&engine, client, &cli.client_args);
    }

    // --- One-shot mode (default, unchanged) ---

    // Record audio
    feedback::prompt("Speak now... (press Enter when done)");
    feedback::play_begin_sound();

    let mut capture = audio::AudioCapture::new()?;
    capture.start()?;
    feedback::recording();

    // Wait for Enter keypress on stdin
    let mut line = String::new();
    let _ = std::io::stdin().lock().read_line(&mut line);

    let (samples, sample_rate) = capture.stop()?;
    feedback::play_end_sound();

    if samples.is_empty() {
        return Ok(());
    }

    // Transcribe
    let spinner = feedback::spinner("Transcribing...");
    let text = engine.transcribe(&samples, sample_rate)?;
    spinner.finish_and_clear();

    if text.is_empty() {
        feedback::warning("No speech detected.");
        return Ok(());
    }

    feedback::transcription(&text);

    // Launch client or print to stdout
    if let Some(client) = cli.client {
        feedback::launching(&client, &text, &cli.client_args);

        use std::os::unix::process::CommandExt;
        let mut cmd = std::process::Command::new(&client);
        cmd.arg(&text);
        cmd.args(&cli.client_args);

        // exec() replaces the current process — gives the client full terminal control
        let err = cmd.exec();
        anyhow::bail!("failed to exec {}: {}", client, err);
    } else {
        // No client: print raw transcription to stdout for piping
        println!("{}", text);
    }

    Ok(())
}
