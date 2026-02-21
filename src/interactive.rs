use anyhow::{Context, Result};
use nix::libc;
use nix::poll::{PollFd, PollFlags, PollTimeout};
use nix::pty::openpty;
use nix::sys::termios;
use nix::unistd::{dup2, execvp, fork, read, setsid, write, ForkResult, Pid};
use std::ffi::CString;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};

use crate::audio::AudioCapture;
use crate::feedback;
use crate::transcribe::TranscriptionEngine;

const CTRL_SPACE: u8 = 0x00; // NUL byte — Ctrl+Space in raw mode
const CTRL_BACKSLASH: u8 = 0x1C; // Ctrl+\ in legacy terminal mode
const CTRL_C: u8 = 0x03;

// Kitty keyboard protocol escape sequences (sent when child enables kitty mode).
// The child's \x1b[>...u sequences get proxied to the real terminal, changing
// how it encodes keystrokes — single bytes become CSI u sequences.
const KITTY_CTRL_SPACE: &[u8] = b"\x1b[32;5u"; // Ctrl+Space in kitty mode
const KITTY_CTRL_BACKSLASH: &[u8] = b"\x1b[92;5u"; // Ctrl+\ in kitty mode
const KITTY_ENTER: &[u8] = b"\x1b[13u"; // Enter in kitty mode (no modifier)
const KITTY_ENTER_MOD: &[u8] = b"\x1b[13;1u"; // Enter in kitty mode (explicit modifier)
const KITTY_CTRL_C: &[u8] = b"\x1b[99;5u"; // Ctrl+C in kitty mode

/// Find a voice trigger in the input byte stream.
/// Returns (start_position, byte_length) of the trigger if found.
/// Checks both legacy single-byte and kitty escape sequence forms.
fn find_voice_trigger(input: &[u8]) -> Option<(usize, usize)> {
    // Check single-byte triggers first (legacy terminal mode)
    if let Some(pos) = input
        .iter()
        .position(|&b| b == CTRL_SPACE || b == CTRL_BACKSLASH)
    {
        return Some((pos, 1));
    }

    // Check kitty-encoded triggers
    for pattern in [KITTY_CTRL_SPACE, KITTY_CTRL_BACKSLASH] {
        if let Some(pos) = input.windows(pattern.len()).position(|w| w == pattern) {
            return Some((pos, pattern.len()));
        }
    }

    None
}

/// Check if input contains Enter (finish recording) in legacy or kitty mode.
fn has_enter(input: &[u8]) -> bool {
    input.contains(&b'\r')
        || input.contains(&b'\n')
        || input.windows(KITTY_ENTER.len()).any(|w| w == KITTY_ENTER)
        || input
            .windows(KITTY_ENTER_MOD.len())
            .any(|w| w == KITTY_ENTER_MOD)
}

/// Check if input contains Ctrl+C (cancel recording) in legacy or kitty mode.
fn has_ctrl_c(input: &[u8]) -> bool {
    input.contains(&CTRL_C) || input.windows(KITTY_CTRL_C.len()).any(|w| w == KITTY_CTRL_C)
}

/// RAII guard that restores the terminal's original termios on drop.
struct TerminalGuard {
    original: termios::Termios,
}

impl TerminalGuard {
    fn new() -> Result<Self> {
        let original =
            termios::tcgetattr(std::io::stdin()).context("failed to get terminal attributes")?;
        let mut raw = original.clone();
        termios::cfmakeraw(&mut raw);
        termios::tcsetattr(std::io::stdin(), termios::SetArg::TCSANOW, &raw)
            .context("failed to set raw mode")?;
        Ok(Self { original })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = termios::tcsetattr(std::io::stdin(), termios::SetArg::TCSANOW, &self.original);
    }
}

/// Get the current terminal window size.
fn get_winsize() -> libc::winsize {
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    unsafe {
        libc::ioctl(
            libc::STDIN_FILENO,
            libc::TIOCGWINSZ as libc::c_ulong,
            &mut ws,
        )
    };
    ws
}

/// Set the window size on a file descriptor (PTY master).
fn set_winsize(fd: RawFd, ws: &libc::winsize) {
    unsafe { libc::ioctl(fd, libc::TIOCSWINSZ as libc::c_ulong, ws) };
}

/// Set up a self-pipe that receives a byte on SIGWINCH.
fn setup_sigwinch_pipe() -> Result<(OwnedFd, OwnedFd)> {
    let (read_fd, write_fd) = nix::unistd::pipe().context("failed to create signal pipe")?;

    // Make the write end non-blocking so the signal handler never blocks
    nix::fcntl::fcntl(
        write_fd.as_raw_fd(),
        nix::fcntl::FcntlArg::F_SETFL(nix::fcntl::OFlag::O_NONBLOCK),
    )?;

    // Register SIGWINCH to write to our pipe
    signal_hook::low_level::pipe::register(signal_hook::consts::SIGWINCH, write_fd.as_raw_fd())?;

    Ok((read_fd, write_fd))
}

/// Spawn the client process inside a new PTY.
/// Returns the master fd and the child's PID.
fn spawn_in_pty(client: &str, args: &[String]) -> Result<(OwnedFd, Pid)> {
    let pty = openpty(None, None).context("openpty failed")?;

    // Set initial window size to match the real terminal
    let ws = get_winsize();
    set_winsize(pty.master.as_raw_fd(), &ws);

    // Safety: we immediately exec or _exit in the child; no async-signal-unsafe
    // operations happen between fork and exec.
    match unsafe { fork() }.context("fork failed")? {
        ForkResult::Child => {
            // Close master side in child
            drop(pty.master);

            // Create new session and set controlling terminal
            setsid().ok();
            unsafe { libc::ioctl(pty.slave.as_raw_fd(), libc::TIOCSCTTY as libc::c_ulong, 0) };

            // Dup slave to stdin/stdout/stderr
            dup2(pty.slave.as_raw_fd(), libc::STDIN_FILENO).ok();
            dup2(pty.slave.as_raw_fd(), libc::STDOUT_FILENO).ok();
            dup2(pty.slave.as_raw_fd(), libc::STDERR_FILENO).ok();
            if pty.slave.as_raw_fd() > 2 {
                drop(pty.slave);
            }

            // Build argv for execvp
            let c_client = CString::new(client).unwrap();
            let mut c_args: Vec<CString> = vec![c_client.clone()];
            for arg in args {
                c_args.push(CString::new(arg.as_str()).unwrap());
            }

            let _ = execvp(&c_client, &c_args);
            // If exec fails, exit immediately
            unsafe { libc::_exit(127) };
        }
        ForkResult::Parent { child } => {
            // Close slave side in parent
            drop(pty.slave);
            Ok((pty.master, child))
        }
    }
}

#[derive(PartialEq)]
enum Mode {
    Proxy,
    Recording,
}

/// Borrow stdin as a `BorrowedFd` without taking ownership.
fn stdin_fd() -> BorrowedFd<'static> {
    // Safety: fd 0 is always valid while the process is running
    unsafe { BorrowedFd::borrow_raw(libc::STDIN_FILENO) }
}

/// Borrow stdout as a `BorrowedFd` without taking ownership.
fn stdout_fd() -> BorrowedFd<'static> {
    // Safety: fd 1 is always valid while the process is running
    unsafe { BorrowedFd::borrow_raw(libc::STDOUT_FILENO) }
}

/// Run the interactive PTY wrapper.
pub fn run(engine: &TranscriptionEngine, client: &str, client_args: &[String]) -> Result<()> {
    eprintln!(
        "  {} Launching {} (interactive, Ctrl+\\ to speak)\n",
        console::style("→").yellow(),
        client,
    );

    let (master_fd, child_pid) = spawn_in_pty(client, client_args)?;

    let (sig_read, _sig_write) = setup_sigwinch_pipe()?;

    // Put the real terminal in raw mode so we can intercept Ctrl+Space
    let _guard = TerminalGuard::new()?;

    let mut mode = Mode::Proxy;
    let mut capture: Option<AudioCapture> = None;
    let mut bracketed_paste = false;
    let mut buf = [0u8; 4096];

    loop {
        let mut poll_fds = vec![
            PollFd::new(master_fd.as_fd(), PollFlags::POLLIN),
            PollFd::new(sig_read.as_fd(), PollFlags::POLLIN),
            PollFd::new(stdin_fd(), PollFlags::POLLIN),
        ];

        let n = nix::poll::poll(&mut poll_fds, PollTimeout::NONE)?;

        if n == 0 {
            continue;
        }

        let master_revents = poll_fds[0].revents().unwrap_or(PollFlags::empty());
        let signal_revents = poll_fds[1].revents().unwrap_or(PollFlags::empty());
        let stdin_revents = poll_fds[2].revents().unwrap_or(PollFlags::empty());

        // --- SIGWINCH: forward terminal resize to PTY ---
        if signal_revents.intersects(PollFlags::POLLIN) {
            // Drain the signal pipe
            let mut drain = [0u8; 64];
            let _ = read(sig_read.as_raw_fd(), &mut drain);

            let ws = get_winsize();
            set_winsize(master_fd.as_raw_fd(), &ws);
        }

        // --- PTY master output: forward child output to real terminal ---
        if master_revents.intersects(PollFlags::POLLIN) {
            match read(master_fd.as_raw_fd(), &mut buf) {
                Ok(0) | Err(_) => break, // child closed PTY
                Ok(n) => {
                    let output = &buf[..n];
                    // Track bracketed paste mode set by the child
                    if output.windows(8).any(|w| w == b"\x1b[?2004h") {
                        bracketed_paste = true;
                    }
                    if output.windows(8).any(|w| w == b"\x1b[?2004l") {
                        bracketed_paste = false;
                    }
                    write_all_fd(stdout_fd(), output)?;
                }
            }
        }

        // Child exited (hangup on master)
        if master_revents.intersects(PollFlags::POLLHUP) {
            // Drain any remaining output
            loop {
                match read(master_fd.as_raw_fd(), &mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        write_all_fd(stdout_fd(), &buf[..n])?;
                    }
                }
            }
            break;
        }

        // --- Stdin: handle user keyboard input ---
        if stdin_revents.intersects(PollFlags::POLLIN) {
            match read(libc::STDIN_FILENO, &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let input = &buf[..n];

                    match mode {
                        Mode::Proxy => {
                            // Scan for voice trigger (legacy or kitty-encoded)
                            if let Some((pos, len)) = find_voice_trigger(input) {
                                // Forward any bytes before the trigger
                                if pos > 0 {
                                    write_all_fd(master_fd.as_fd(), &input[..pos])?;
                                }
                                // Enter recording mode
                                mode = Mode::Recording;
                                feedback::set_title("● Recording... (Enter to finish)");
                                feedback::play_begin_sound();
                                let mut cap = AudioCapture::new()?;
                                cap.start()?;
                                capture = Some(cap);

                                // Handle any bytes after the trigger
                                let rest = &input[pos + len..];
                                if !rest.is_empty() {
                                    if has_enter(rest) {
                                        finish_recording(
                                            &mut capture,
                                            engine,
                                            master_fd.as_fd(),
                                            bracketed_paste,
                                            &mut mode,
                                        )?;
                                    } else if has_ctrl_c(rest) {
                                        cancel_recording(&mut capture, &mut mode);
                                    }
                                }
                            } else {
                                // Normal proxy: forward all input to child
                                write_all_fd(master_fd.as_fd(), input)?;
                            }
                        }
                        Mode::Recording => {
                            if has_enter(input) {
                                finish_recording(
                                    &mut capture,
                                    engine,
                                    master_fd.as_fd(),
                                    bracketed_paste,
                                    &mut mode,
                                )?;
                            } else if has_ctrl_c(input) {
                                cancel_recording(&mut capture, &mut mode);
                            }
                            // Other keystrokes are discarded during recording
                        }
                    }
                }
            }
        }
    }

    // Collect child exit status
    drop(master_fd); // close master before waitpid
    let status = nix::sys::wait::waitpid(child_pid, None)?;
    drop(_guard); // restore terminal

    match status {
        nix::sys::wait::WaitStatus::Exited(_, code) => std::process::exit(code),
        nix::sys::wait::WaitStatus::Signaled(_, sig, _) => std::process::exit(128 + sig as i32),
        _ => std::process::exit(1),
    }
}

/// Stop recording, transcribe, and inject text into the PTY.
fn finish_recording(
    capture: &mut Option<AudioCapture>,
    engine: &TranscriptionEngine,
    master_fd: BorrowedFd<'_>,
    bracketed_paste: bool,
    mode: &mut Mode,
) -> Result<()> {
    if let Some(mut cap) = capture.take() {
        let (samples, sample_rate) = cap.stop()?;
        feedback::play_end_sound();

        if !samples.is_empty() {
            feedback::set_title("○ Transcribing...");
            match engine.transcribe(&samples, sample_rate) {
                Ok(text) if !text.is_empty() => {
                    // Inject transcribed text into child's stdin via PTY master.
                    // Wrap in bracketed paste markers so TUI apps (claude, kiro, etc.)
                    // treat it as pasted text inserted into the prompt, not as
                    // individual keystrokes that trigger UI actions.
                    if bracketed_paste {
                        write_all_fd(master_fd, b"\x1b[200~")?;
                    }
                    write_all_fd(master_fd, text.as_bytes())?;
                    if bracketed_paste {
                        write_all_fd(master_fd, b"\x1b[201~")?;
                    }
                }
                Ok(_) => {} // empty transcription, nothing to inject
                Err(e) => {
                    feedback::set_title(&format!("Transcription error: {}", e));
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
    }

    feedback::clear_title();
    *mode = Mode::Proxy;
    Ok(())
}

/// Cancel recording and return to proxy mode.
fn cancel_recording(capture: &mut Option<AudioCapture>, mode: &mut Mode) {
    if let Some(mut cap) = capture.take() {
        let _ = cap.stop();
    }
    feedback::play_end_sound();
    feedback::clear_title();
    *mode = Mode::Proxy;
}

/// Write all bytes to a fd, handling partial writes.
fn write_all_fd(fd: BorrowedFd<'_>, mut data: &[u8]) -> Result<()> {
    while !data.is_empty() {
        match write(fd, data) {
            Ok(n) => data = &data[n..],
            Err(nix::errno::Errno::EINTR) => continue,
            Err(e) => return Err(e).context("write failed"),
        }
    }
    Ok(())
}
