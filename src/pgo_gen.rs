//! Self-contained synthetic dataset generator for build-time PGO training only.
//! Compiled directly with `rustc -O src/pgo_gen.rs` inside the Docker builder
//! stage and never linked into the runtime binary.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn u(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn b36(prefix: u8, mut v: u64, out: &mut String) {
    const DIG: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    out.clear();
    out.push(prefix as char);
    let mut buf = [0u8; 8];
    for i in (0..8).rev() {
        buf[i] = DIG[(v % 36) as usize];
        v /= 36;
    }
    let start = if buf[0] == b'0' && (v & 1) == 0 { 1 } else { 0 };
    for &c in &buf[start..] {
        out.push(c as char);
    }
}

fn fmt_utc(sec_from_oct1: i64, out: &mut String) {
    let day = 1 + (sec_from_oct1 / 86_400);
    let rem = sec_from_oct1 % 86_400;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let s = rem % 60;
    use std::fmt::Write as _;
    out.clear();
    let _ = write!(out, "2026-10-{day:02}T{h:02}:{m:02}:{s:02}Z");
}

fn q(rng: &mut Rng, s: &str, w: &mut impl Write) {
    if rng.u(5) == 0 {
        let _ = write!(w, "\"{s}\"");
    } else {
        let _ = w.write_all(s.as_bytes());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0x5eed_cafe_1234_5678);
    let mut rng = Rng(seed);

    for sub in ["events", "clicks", "links"] {
        fs::create_dir_all(dir.join(sub)).unwrap();
    }
    fs::write(
        dir.join("manifest.json"),
        "{\"window_start\":\"2026-10-08T00:00:00Z\",\"window_end\":\"2026-10-22T00:00:00Z\"}\n",
    )
    .unwrap();

    // products.csv (4000 products, 25 categories)
    {
        let mut w = BufWriter::new(File::create(dir.join("products.csv")).unwrap());
        writeln!(w, "product_id,category").unwrap();
        for i in 0..4000u32 {
            writeln!(w, "p{i},cat_{:02}", i % 25).unwrap();
        }
    }
    // campaigns.csv (120 campaigns, 6 channels)
    let chans = ["search", "social", "email", "display", "video", "affiliate"];
    {
        let mut w = BufWriter::new(File::create(dir.join("campaigns.csv")).unwrap());
        writeln!(w, "campaign_id,channel").unwrap();
        for i in 0..120usize {
            writeln!(w, "c_{i},{}", chans[i % chans.len()]).unwrap();
        }
    }

    const N_USERS: u64 = 55_000;
    const N_EVENTS: usize = 450_000;
    const N_CLICKS: usize = 110_000;
    const N_LINKS: usize = 50_000;
    const OCT1_MS: i64 = 1_790_812_800_000; // 2026-10-01T00:00:00Z

    let mut s1 = String::with_capacity(32);
    let mut s2 = String::with_capacity(32);
    let mut ts_s = String::with_capacity(32);
    let mut ing_s = String::with_capacity(32);

    // links
    {
        let mut w = BufWriter::new(File::create(dir.join("links/part-00000.csv")).unwrap());
        writeln!(w, "user_id_a,user_id_b").unwrap();
        for _ in 0..N_LINKS {
            let a = rng.u(N_USERS) + 1;
            let b = rng.u(N_USERS) + 1;
            b36(b'u', a.wrapping_mul(0x9e37_79b9) & 0xFFFF_FFFF_FF, &mut s1);
            b36(b'u', b.wrapping_mul(0x9e37_79b9) & 0xFFFF_FFFF_FF, &mut s2);
            q(&mut rng, &s1, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &s2, &mut w);
            let _ = w.write_all(b"\n");
        }
    }

    // clicks
    {
        let mut w = BufWriter::new(File::create(dir.join("clicks/part-00000.csv")).unwrap());
        writeln!(w, "click_id,user_id,campaign_id,ts").unwrap();
        for i in 0..N_CLICKS {
            let u = if rng.u(4) == 0 { rng.u(300) + 1 } else { rng.u(N_USERS) + 1 };
            b36(b'k', (i as u64 + 1).wrapping_mul(0x517c_c1b7) & 0xFFFF_FFFF_FF, &mut s1);
            b36(b'u', u.wrapping_mul(0x9e37_79b9) & 0xFFFF_FFFF_FF, &mut s2);
            let camp = format!("c_{}", rng.u(120));
            let sec = rng.u(21 * 86_400) as i64; // Oct 1..22
            match rng.u(10) {
                0 => {
                    ts_s.clear();
                    use std::fmt::Write as _;
                    let _ = write!(ts_s, "{}", OCT1_MS + sec * 1000);
                }
                1 => {
                    fmt_utc(sec, &mut ts_s);
                    ts_s.pop();
                    ts_s.push_str("+02:00");
                }
                _ => fmt_utc(sec, &mut ts_s),
            }
            q(&mut rng, &s1, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &s2, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &camp, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &ts_s, &mut w);
            let _ = w.write_all(b"\n");
        }
    }

    // events (split across 4 parts so parallel chunking is exercised)
    let per_part = N_EVENTS / 4;
    for part in 0..4usize {
        let mut w = BufWriter::new(File::create(dir.join(format!("events/part-{part:05}.csv"))).unwrap());
        writeln!(w, "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents").unwrap();
        for j in 0..per_part {
            let gid = (part * per_part + j) as u64;
            let eid_num = if rng.u(25) == 0 && gid > 100 { gid - rng.u(100) - 1 } else { gid + 1 };
            // ~28% of events on 120 whales (>256 events each) to train timebins + hot cache
            let u = if rng.u(100) < 28 { rng.u(120) + 1 } else { rng.u(N_USERS) + 1 };
            b36(b'e', eid_num.wrapping_mul(0x632b_e59b) & 0xFFFF_FFFF_FF, &mut s1);
            b36(b'u', u.wrapping_mul(0x9e37_79b9) & 0xFFFF_FFFF_FF, &mut s2);
            let sec = (7 * 86_400 + rng.u(14 * 86_400)) as i64; // Oct 8..22
            let late = if rng.u(40) == 0 { 90_000 } else { rng.u(3600) as i64 };
            let ing_sec = (sec + late).min(29 * 86_400);
            match rng.u(10) {
                0..=2 => {
                    ts_s.clear();
                    use std::fmt::Write as _;
                    let _ = write!(ts_s, "{}", OCT1_MS + sec * 1000);
                }
                3 => {
                    fmt_utc(sec, &mut ts_s);
                    ts_s.pop();
                    ts_s.push_str("-04:00");
                }
                _ => fmt_utc(sec, &mut ts_s),
            }
            fmt_utc(ing_sec, &mut ing_s);
            let kind = rng.u(10);
            let (et, price) = if kind < 6 {
                ("view", "")
            } else if kind < 8 {
                ("cart", "")
            } else {
                ("purchase", "2499")
            };
            let prod = format!("p{}", rng.u(4000));
            q(&mut rng, &s1, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &s2, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &ts_s, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &ing_s, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, et, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &prod, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, price, &mut w);
            let _ = w.write_all(b"\n");
        }
    }
}
