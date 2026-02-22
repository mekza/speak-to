use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

const SOUND_DIR: &str =
    "/System/Library/Components/CoreAudio.component/Contents/SharedSupport/SystemSounds/system";

pub fn status(msg: &str) {
    eprintln!("  {} {}", style("○").dim(), msg);
}

pub fn prompt(msg: &str) {
    eprintln!("\n  {}", msg);
}

pub fn recording() {
    eprintln!("  {} {}", style("●").red(), "Recording...");
}

pub fn success(msg: &str) {
    eprintln!("  {} {}", style("✓").green(), msg);
}

pub fn warning(msg: &str) {
    eprintln!("  {} {}", style("!").yellow(), msg);
}

pub fn transcription(text: &str) {
    eprintln!("  {} \"{}\"", style("✓").green(), style(text).bold());
}

pub fn launching(client: &str, text: &str, extra_args: &[String]) {
    let args_str = if extra_args.is_empty() {
        String::new()
    } else {
        format!(" {}", extra_args.join(" "))
    };
    eprintln!(
        "\n  {} {}{} \"{}\"",
        style("→").yellow(),
        client,
        args_str,
        text,
    );
}

pub fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("  {spinner} {msg}")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

pub fn play_begin_sound() {
    play_sound("begin_record.caf");
}

pub fn play_end_sound() {
    play_sound("end_record.caf");
}

pub fn set_title(msg: &str) {
    eprint!("\x1b]0;{}\x07", msg);
}

pub fn clear_title() {
    eprint!("\x1b]0;\x07");
}

fn play_sound(name: &str) {
    let path = format!("{}/{}", SOUND_DIR, name);
    let _ = std::process::Command::new("afplay").arg(&path).spawn();
}
