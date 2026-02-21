# speak-to

Voice input for CLI tools via fully local [Voxtral Mini 4B](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) transcription. Everything runs locally on your machine.

## Usage

### One-shot mode

Record a voice prompt, transcribe it, and launch a CLI tool with the result:

```bash
# Transcribe and pass to claude as the initial prompt
speak-to claude

# Same, with extra arguments forwarded to the client
speak-to claude -- --model sonnet

# Transcribe to stdout (for piping)
speak-to
```

The flow: load model, record until you press Enter, transcribe, exec the client with the transcribed text as the first argument.

### Interactive mode

Wrap any interactive CLI tool in a PTY and trigger voice input at any time with `Ctrl+\`:

```bash
speak-to -i claude

speak-to -i kiro-cli

speak-to -i claude -- --model sonnet
```

The client runs normally in your terminal. Press `Ctrl+\` to start recording, speak, then press Enter. The transcribed text is injected into the client's input as a paste. Press `Ctrl+C` during recording to cancel.

The model stays loaded in memory between recordings so subsequent transcriptions are fast.

## Install

Requires a working Rust toolchain.

```bash
cargo install --path .
```

On first run, the Q4 model (~4GB) is downloaded from HuggingFace and cached locally.

### Pre-download the model

```bash
speak-to --download
```

### macOS audio permissions

macOS will prompt for microphone access on first use. Grant it to your terminal emulator (iTerm2, Terminal.app, etc.).

## Model options

The default is Voxtral Mini Q4 (quantized, ~4GB download, ~700MB memory). For lower word error rate at the cost of ~9GB memory:

```bash
speak-to --f32 claude

speak-to --download --f32
```

You can also point to a local model:

```bash
# Q4 GGUF file
speak-to --model-path /path/to/voxtral-q4.gguf claude

# F32 SafeTensors directory
speak-to --model-path /path/to/model-dir/ claude
```

## How it works

**One-shot mode** records from your mic, transcribes locally on the GPU via [burn](https://burn.dev), and execs the target CLI with the result.

**Interactive mode** wraps the client in a PTY so speak-to sits between you and the client. It intercepts the trigger keystroke, records and transcribes, then pastes the text into the client's prompt. The client has no idea anything special happened -- it just sees text appear as if you typed it.

## Requirements

- macOS (uses CoreAudio for recording and system sounds)
- GPU with wgpu support (Metal on macOS)
- ~4GB disk for the Q4 model (~9GB for F32)

## Shoutouts

Transcription is powered by [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) by TrevorS, a Rust implementation of Mistral's Voxtral Mini model using the burn framework.

## License

Apache License 2.0
