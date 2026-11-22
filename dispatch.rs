use std::{env, os::unix::process::CommandExt, process::Command};

fn selected_solver() -> (&'static str, &'static str) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512cd")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512vl")
        {
            return ("x86-64-v4", "/shopping-sessions");
        }
        ("x86-64-v3", "/shopping-sessions-v3")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        ("native", "/shopping-sessions")
    }
}

fn main() {
    let (isa, path) = selected_solver();
    if env::var_os("SHOPPING_REPORT_CPU").as_deref() == Some(std::ffi::OsStr::new("1")) {
        println!("{isa} {path}");
        return;
    }
    let error = Command::new(path).args(env::args_os().skip(1)).exec();
    eprintln!("Cannot execute {path}: {error}");
    std::process::exit(127);
}
