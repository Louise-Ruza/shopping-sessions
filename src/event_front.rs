//! Guarded event-schema frontend. The only fast-path assumptions (96-byte row,
//! first two delimiters within 32 bytes, and common timestamp/type lengths) are
//! checked. Every other physical row goes to the existing exact CSV scanner.
//! No IDs, dates, prices, or event counts are inferred from generator behavior.

use crate::{csv, ts::{DateCache, parse_price}};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event<'a> {
    pub eid: &'a [u8],
    pub uid: &'a [u8],
    pub product: &'a [u8],
    pub timestamp: Option<i64>,
    pub ingest: Option<i64>,
    pub kind: u8, // 0 invalid; 1 view/cart; 2 purchase
    pub price: u32,
}

#[derive(Default, Debug, PartialEq, Eq)]
struct Counts { fast: usize, fallback: usize }

#[inline(always)]
fn fallback<'a, F: FnMut(Event<'a>)>(data: &'a [u8], dates: &DateCache, f: &mut F) {
    csv::scan::<7, _>(data, |fields| {
        let kind = match fields[4] { b"view" | b"cart" => 1, b"purchase" => 2, _ => 0 };
        #[cfg(feature = "fast-fields")]
        let (timestamp, ingest) = dates.parse_event_ingest(fields[2], fields[3]);
        #[cfg(not(feature = "fast-fields"))]
        let (timestamp, ingest) = (dates.parse_ts(fields[2]), dates.parse_f1_ingest(fields[3]));
        f(Event { eid: fields[0], uid: fields[1], product: fields[5], timestamp, ingest, kind,
            price: if kind == 2 { parse_price(fields[6]) } else { 0 } });
    });
}

#[inline(always)]
fn prefix_xor(mut q: u128) -> u128 {
    q ^= q << 1; q ^= q << 2; q ^= q << 4; q ^= q << 8;
    q ^= q << 16; q ^= q << 32; q ^= q << 64;
    q
}

#[inline(always)]
fn scalar_masks(data: &[u8; 96]) -> (u128, u128, u128) {
    const LO7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    #[inline(always)]
    fn equal(word: u64, value: u64) -> u128 {
        let y = word ^ value;
        let high = !(((y & LO7).wrapping_add(LO7)) | y | LO7);
        ((high >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u128
    }
    let (mut c, mut l, mut q) = (0, 0, 0);
    for i in 0..12 {
        let w = u64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap());
        c |= equal(w, 0x2c2c_2c2c_2c2c_2c2c) << (i * 8);
        l |= equal(w, 0x0a0a_0a0a_0a0a_0a0a) << (i * 8);
        q |= equal(w, 0x2222_2222_2222_2222) << (i * 8);
    }
    (c, l, q)
}

#[inline(always)]
fn field(data: &[u8], a: usize, b: usize) -> &[u8] {
    // Caller has proved exact delimiter positions and whole-row quote legality.
    let quoted = unsafe { *data.get_unchecked(a) == b'"' } as usize;
    unsafe { data.get_unchecked(a + quoted..b - quoted) }
}

#[inline(always)]
fn row<'a, P: Fn(u128) -> u128>(data: &'a [u8; 96], end: usize, mut commas: u128,
    quotes: u128, dates: &DateCache, parity: &P) -> Option<Event<'a>>
{
    let below = (1u128 << end) - 1;
    commas &= below;
    if commas.count_ones() != 6 { return None; }
    let low = commas as u64;
    let c0 = low.trailing_zeros() as usize;
    let c1 = (low & low.wrapping_sub(1)).trailing_zeros() as usize;
    // This bound permits unchecked 13/20/25-byte timestamp shape probes within
    // the actual 96-byte array. Longer arbitrary IDs use the exact fallback.
    if c1 > 32 { return None; }
    let q = quotes & below;
    if q != 0 {
        let separators = commas | (1u128 << end);
        let edges = (separators << 1) | 1 | (separators >> 1);
        if q & !edges != 0 || parity(q) & separators != 0 { return None; }
    }
    let tq = (data[c1 + 1] == b'"') as usize;
    let ta = c1 + 1 + tq;
    let (tn, c2) = if data[ta + 19] == b'Z' && data[ta + 20 + tq] == b',' {
        (20, ta + 20 + tq)
    } else if data[ta + 13 + tq] == b',' {
        (13, ta + 13 + tq)
    } else if data[ta + 25 + tq] == b',' {
        (25, ta + 25 + tq)
    } else { return None; };
    let iq = (data[c2 + 1] == b'"') as usize;
    let ia = c2 + 1 + iq;
    let c3 = ia + 20 + iq;
    if data[c3] != b',' { return None; }
    let kq = (data[c3 + 1] == b'"') as usize;
    let ka = c3 + 1 + kq;
    // ka can reach 86; the largest probe (eight-byte purchase + quote/comma)
    // stays inside the actual 96-byte array because c1 <= 32.
    let (kind, c4) = if (&data[ka..ka + 4] == b"view" || &data[ka..ka + 4] == b"cart")
        && data[ka + 4 + kq] == b',' { (1, ka + 4 + kq) }
    else if &data[ka..ka + 8] == b"purchase" && data[ka + 8 + kq] == b',' {
        (2, ka + 8 + kq)
    } else { return None; };
    let c5 = 127 - commas.leading_zeros() as usize;
    // Six commas, first two and final one from their bit positions, plus three
    // distinct predicted commas establish every boundary without three more
    // dynamic delimiter extractions. In particular probes into a later row can
    // never pass this ordering check.
    if c4 >= c5 { return None; }
    let (timestamp, ingest) = match tn {
        20 => {
            #[cfg(feature = "fast-fields")]
            { dates.parse_event_ingest(&data[ta..ta + 20], &data[ia..ia + 20]) }
            #[cfg(not(feature = "fast-fields"))]
            { (dates.parse_ts(&data[ta..ta + 20]), dates.parse_f1_ingest(&data[ia..ia + 20])) }
        }
        13 => (dates.parse_ts(&data[ta..ta + 13]), dates.parse_f1_ingest(&data[ia..ia + 20])),
        _ => (dates.parse_ts(&data[ta..ta + 25]), dates.parse_f1_ingest(&data[ia..ia + 20])),
    };
    Some(Event { eid: field(data, 0, c0), uid: field(data, c0 + 1, c1),
        product: field(data, c4 + 1, c5), timestamp, ingest, kind,
        price: if kind == 2 { parse_price(field(data, c5 + 1, end)) } else { 0 } })
}

#[inline(always)]
fn scan_with<'a, const COUNT: bool, F: FnMut(Event<'a>), M: Fn(&[u8; 96]) -> (u128, u128, u128), P: Fn(u128) -> u128>(
    mut data: &'a [u8], dates: &DateCache, mut f: F, masks: M, parity: P,
) -> Counts {
    let mut counts = Counts::default();
    while data.len() >= 96 {
        // The array reference proves all three SIMD loads and shape probes fit.
        let bytes: &'a [u8; 96] = unsafe { &*data.as_ptr().cast::<[u8; 96]>() };
        let (commas, lines, quotes) = masks(bytes);
        let end = lines.trailing_zeros() as usize;
        if end < 96 {
            if let Some(event) = row(bytes, end, commas, quotes, dates, &parity) {
                f(event);
                if COUNT { counts.fast += 1; }
            } else {
                fallback(&data[..end + 1], dates, &mut f);
                if COUNT { counts.fallback += 1; }
            }
            data = &data[end + 1..];
        } else {
            let end = data[96..].iter().position(|&b| b == b'\n')
                .map_or(data.len(), |p| 96 + p + 1);
            fallback(&data[..end], dates, &mut f);
            if COUNT { counts.fallback += 1; }
            data = &data[end..];
        }
    }
    if COUNT {
        counts.fallback += data.iter().filter(|&&b| b == b'\n').count()
            + usize::from(!data.is_empty() && data.last() != Some(&b'\n'));
    }
    fallback(data, dates, &mut f);
    counts
}

#[inline(always)]
pub fn scan<'a, F: FnMut(Event<'a>)>(data: &'a [u8], dates: &DateCache, f: F) {
    #[cfg(target_arch = "x86_64")]
    if csv::simd_backend() == "avx2" && is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("popcnt") {
        unsafe { scan_avx2::<false, F>(data, dates, f) };
        return;
    }
    scan_with::<false, F, _, _>(data, dates, f, scalar_masks, prefix_xor);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,popcnt,pclmulqdq")]
unsafe fn scan_avx2<'a, const COUNT: bool, F: FnMut(Event<'a>)>(data: &'a [u8], dates: &DateCache, f: F) -> Counts {
    scan_with::<COUNT, F, _, _>(data, dates, f, |b| avx2_masks(b), |q| clmul_parity(q))
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx2_masks(data: &[u8; 96]) -> (u128, u128, u128) {
    use core::arch::x86_64::*;
    let (mut c, mut l, mut q) = (0, 0, 0);
    for i in 0..3 {
        let bytes = _mm256_loadu_si256(data.as_ptr().add(32 * i).cast());
        c |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(bytes, _mm256_set1_epi8(b',' as i8))) as u32 as u128) << (32 * i);
        l |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(bytes, _mm256_set1_epi8(b'\n' as i8))) as u32 as u128) << (32 * i);
        q |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(bytes, _mm256_set1_epi8(b'"' as i8))) as u32 as u128) << (32 * i);
    }
    (c, l, q)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn clmul_parity(q: u128) -> u128 {
    use core::arch::x86_64::*;
    let bits = _mm_set_epi64x((q >> 64) as i64, q as i64);
    let ones = _mm_set1_epi64x(-1);
    let lo = _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(bits, ones)) as u64;
    let hi = _mm_cvtsi128_si64(_mm_clmulepi64_si128::<1>(bits, ones)) as u64;
    lo as u128 | (((hi ^ 0u64.wrapping_sub(lo >> 63)) as u128) << 64)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dates() -> DateCache { DateCache::new(1_791_417_600_000, 1_792_627_200_000) }

    fn expected<'a>(input: &'a [u8], dates: &DateCache) -> Vec<Event<'a>> {
        let mut out = Vec::new();
        csv::scan_byte_oracle::<7, _>(input, |f| {
            out.push(Event { eid: f[0], uid: f[1], product: f[5],
                timestamp: dates.parse_ts(f[2]), ingest: dates.parse_f1_ingest(f[3]),
                kind: match f[4] { b"view" | b"cart" => 1, b"purchase" => 2, _ => 0 },
                price: if f[4] == b"purchase" { parse_price(f[6]) } else { 0 } });
        });
        out
    }

    fn check(input: &[u8]) -> Counts {
        let dates = dates();
        let oracle = expected(input, &dates);
        let mut actual = Vec::new();
        let counts = scan_with::<true, _, _, _>(input, &dates, |e| actual.push(e), scalar_masks, prefix_xor);
        assert_eq!(actual, oracle, "input={input:?}");
        actual.clear();
        scan(input, &dates, |e| actual.push(e));
        assert_eq!(actual, oracle, "selected input={input:?}");
        counts
    }

    #[test]
    fn bit_primitives_every_lane_and_byte() {
        for lane in 0..96 {
            for byte in 0..=255 {
                let mut block = [0u8; 96]; block[lane] = byte;
                let expected = (if byte == b',' { 1u128 << lane } else { 0 },
                    if byte == b'\n' { 1u128 << lane } else { 0 },
                    if byte == b'"' { 1u128 << lane } else { 0 });
                assert_eq!(scalar_masks(&block), expected);
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") { assert_eq!(unsafe { avx2_masks(&block) }, expected); }
            }
        }
        for lane in 0..128 {
            let q = 1u128 << lane;
            assert_eq!(prefix_xor(q), u128::MAX << lane);
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("pclmulqdq") { assert_eq!(unsafe { clmul_parity(q) }, prefix_xor(q)); }
        }
    }

    #[test]
    fn timestamp_forms_quotes_and_every_truncation() {
        for ts in ["2026-10-08T12:34:56Z", "2026-10-08T12:34:56-04:00", "1791462896000",
            "0001791462896000", "86400000", "2026-02-30T12:34:56Z", "2026-10-08T24:00:00Z",
            "2026-10-08T12:34:56+24:00", "2026-10-08T12:34:56Z\r", "999999999999999999999999", ""] {
            for quote_bits in 0..128 {
                let values = ["e1", "u1", ts, "2026-10-08T13:00:00Z", "purchase", "p1", "00250"];
                let row = values.iter().enumerate().map(|(i, s)| {
                    if quote_bits & (1 << i) != 0 { format!("\"{s}\"") } else { (*s).to_owned() }
                }).collect::<Vec<_>>().join(",") + "\n";
                let input = row.repeat(3).into_bytes();
                check(&input);
                for end in 0..row.len() { check(&input[..row.len() + end]); }
            }
        }
    }

    #[test]
    fn arbitrary_mutations_offsets_long_ids_and_seams() {
        let valid = b"e1,u1,2026-10-08T12:34:56Z,2026-10-08T13:00:00Z,purchase,p1,250\n";
        for offset in 0..128 {
            for pos in 0..valid.len() {
                for byte in [0, b'"', b',', b'\n', b'\r', b' ', 255] {
                    let mut input = vec![b'!'; offset];
                    input.extend_from_slice(valid);
                    input[offset + pos] = byte;
                    input.extend_from_slice(valid);
                    check(&input[offset..]);
                }
            }
        }
        for size in [0, 31, 32, 33, 63, 64, 95, 96, 97, 128, 16_385] {
            let mut input = vec![b'x'; size]; input.extend_from_slice(&valid[2..]);
            input.extend_from_slice(valid); check(&input);
        }
    }

    #[test]
    fn optional_public_samples_exact_and_counted() {
        let Some(dir) = std::env::var_os("EVENTFRONT_SAMPLE_DIR") else { return; };
        for name in ["dev", "seed19-5m", "seed29-20000000"] {
            let path = std::path::Path::new(&dir).join(format!("{name}.sample.csv"));
            let input = std::fs::read(path).unwrap();
            let counts = check(&input);
            eprintln!("event-front sample {name}: fast={} fallback={}", counts.fast, counts.fallback);
        }
    }
}
