use criterion::Criterion;

/// Criterion config that works under `cargo ndk-runner`.
///
/// The runner pushes the binary to `/data/local/tmp` and execs it with CWD `/`
/// (read-only). Host env vars such as `CRITERION_HOME` are not forwarded into
/// the `adb shell` process, so Criterion's default `./target/criterion` fails.
/// Point reports at a writable location on device instead.
pub fn configured() -> Criterion {
    #[cfg(target_os = "android")]
    {
        use std::path::PathBuf;

        let dir = std::env::var_os("CRITERION_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/data/local/tmp/infers-benchmarks"));
        let _ = std::fs::create_dir_all(&dir);
        Criterion::default().output_directory(&dir)
    }
    #[cfg(not(target_os = "android"))]
    {
        Criterion::default()
    }
}
