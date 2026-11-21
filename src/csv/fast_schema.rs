//! Fixed-arity structural index, with a row-wide proof of quote legality.
//!
//! Quotes must touch a field edge, and quote parity must be even at every field
//! separator. Together these conditions allow exactly zero quotes or a pair at
//! the field's two ends. Parity is normalized at each LF, including malformed
//! rows, so an unmatched quote cannot affect the next row. Commas always split
//! fields; this is the competition dialect, not general RFC 4180 CSV.

const BATCH: usize = 16 * 1024;

#[derive(Clone, Copy, Default)]
struct Quotes {
    bad: u64,
    odd: u64,
    // Cumulative bad-quote count in high 32 bits, odd-separator count in low 32.
    before: u64,
}

struct Index {
    commas: Vec<u32>,
    lines: Vec<u32>,
    quotes: Vec<Quotes>,
    nc: usize,
    nl: usize,
}

impl Index {
    fn new(len: usize) -> Self {
        let size = len.min(BATCH);
        Self {
            // Slack permits fixed-width spilling and N-field sentinel reads.
            commas: vec![0; size + 16],
            lines: vec![0; size + 16],
            quotes: vec![Quotes::default(); size.div_ceil(64) + 1],
            nc: 0,
            nl: 0,
        }
    }

    #[inline(always)]
    fn counts(&self, pos: usize) -> u64 {
        // Called only for an indexed LF + 1, including the terminal sentinel.
        let q = unsafe { *self.quotes.get_unchecked(pos / 64) };
        let mask = (1u64 << (pos % 64)) - 1;
        q.before + (((q.bad & mask).count_ones() as u64) << 32)
            + (q.odd & mask).count_ones() as u64
    }

    #[inline(always)]
    fn build<M: Fn(&[u8; 64]) -> (u64, u64, u64)>(&mut self, data: &[u8], masks: &M) {
        self.nc = 0;
        self.nl = 0;
        let mut before = 0u64;
        let mut carry = 0u64;
        let mut previous_delimiter = 1u64;
        for block in 0..data.len().div_ceil(64) {
            let off = block * 64;
            let mut tail = [0u8; 64];
            let bytes: &[u8; 64] = if off + 64 <= data.len() {
                // The full block is within the input slice, including unaligned inputs.
                unsafe { &*data.as_ptr().add(off).cast::<[u8; 64]>() }
            } else {
                tail[..data.len() - off].copy_from_slice(&data[off..]);
                &tail
            };
            let (commas, lines, quotes) = masks(bytes);
            spill::<4>(&mut self.commas, &mut self.nc, off, commas);
            spill::<1>(&mut self.lines, &mut self.nl, off, lines);
            let delimiters = commas | lines;
            let next_delimiter = match data.get(off + 64) {
                Some(b',' | b'\n') | None => 1u64 << 63,
                _ => 0,
            };
            let edges = (delimiters << 1) | previous_delimiter
                | (delimiters >> 1) | next_delimiter;
            let bad = quotes & !edges;
            let mut parity = quotes;
            if quotes != 0 {
                parity ^= parity << 1;
                parity ^= parity << 2;
                parity ^= parity << 4;
                parity ^= parity << 8;
                parity ^= parity << 16;
                parity ^= parity << 32;
            }
            parity ^= carry;
            let odd = parity & delimiters;
            self.quotes[block] = Quotes { bad, odd, before };
            before += ((bad.count_ones() as u64) << 32) + odd.count_ones() as u64;
            carry = 0u64.wrapping_sub(parity >> 63);
            previous_delimiter = delimiters >> 63;
        }
        self.quotes[data.len().div_ceil(64)] = Quotes { before, ..Quotes::default() };
        self.commas[self.nc..self.nc + 16].fill(u32::MAX);
    }

    #[inline(always)]
    fn emit<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(&self, data: &'a [u8], f: &mut F) -> usize {
        let mut ci = 0usize;
        let mut start = 0usize;
        let mut previous_counts = 0u64;
        let mut row_parity = false;
        for &line in &self.lines[..self.nl] {
            let end = line as usize;
            let counts = self.counts(end + 1);
            let row_counts = counts - previous_counts;
            // Both counters are monotone and < 2^32, so packed subtraction has
            // no borrow between counters. Exactly N separators include the LF.
            // ci never exceeds nc; 16 initialized sentinels cover both arities.
            let shape = unsafe {
                *self.commas.get_unchecked(ci + N - 2) < line
                    && *self.commas.get_unchecked(ci + N - 1) > line
            };
            if shape {
                let expected_odd = if row_parity { N as u32 } else { 0 };
                if row_counts >> 32 == 0 && row_counts as u32 == expected_odd {
                    let fields: [&'a [u8]; N] = std::array::from_fn(|k| {
                        let a = if k == 0 { start } else {
                            unsafe { *self.commas.get_unchecked(ci + k - 1) as usize + 1 }
                        };
                        let b = if k == N - 1 { end } else {
                            unsafe { *self.commas.get_unchecked(ci + k) as usize }
                        };
                        // Every start points at an input byte (possibly a separator).
                        // The row proof guarantees a quote here has its mate at b-1.
                        let quoted = unsafe { *data.get_unchecked(a) == b'"' } as usize;
                        unsafe { data.get_unchecked(a + quoted..b - quoted) }
                    });
                    f(&fields);
                }
                ci += N - 1;
            } else {
                while unsafe { *self.commas.get_unchecked(ci) } < line { ci += 1; }
            }
            row_parity = (unsafe { self.quotes.get_unchecked(end / 64).odd } >> (end % 64)) & 1 != 0;
            previous_counts = counts;
            start = end + 1;
        }
        start
    }
}

#[inline(always)]
fn spill<const FIRST: usize>(out: &mut [u32], count: &mut usize, base: usize, mut bits: u64) {
    let n = bits.count_ones() as usize;
    let mut i = *count;
    // Writes after the real population are ignored and overwritten; Index owns
    // 16 spare entries, and FIRST <= 4. This avoids a branch for each early hit.
    for _ in 0..FIRST {
        // Previous populations plus this block's hits cannot exceed the byte
        // count. Any speculative stores are covered by the 16 spare entries.
        unsafe { *out.get_unchecked_mut(i) = (base + bits.trailing_zeros() as usize) as u32; }
        i += 1;
        bits &= bits.wrapping_sub(1);
    }
    while bits != 0 {
        unsafe { *out.get_unchecked_mut(i) = (base + bits.trailing_zeros() as usize) as u32; }
        i += 1;
        bits &= bits - 1;
    }
    *count += n;
}

#[inline(always)]
fn scalar_masks(bytes: &[u8; 64]) -> (u64, u64, u64) {
    #[inline(always)]
    fn pack(mask: u64) -> u64 {
        (mask >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56
    }
    let mut commas = 0;
    let mut lines = 0;
    let mut quotes = 0;
    for i in 0..8 {
        let w = u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
        commas |= pack(super::eqz(w ^ 0x2c2c_2c2c_2c2c_2c2c)) << (i * 8);
        lines |= pack(super::eqz(w ^ 0x0a0a_0a0a_0a0a_0a0a)) << (i * 8);
        quotes |= pack(super::eqz(w ^ 0x2222_2222_2222_2222)) << (i * 8);
    }
    (commas, lines, quotes)
}

#[inline(always)]
fn scan_with<'a, const N: usize, F: FnMut(&[&'a [u8]; N]), M: Fn(&[u8; 64]) -> (u64, u64, u64)>(
    mut data: &'a [u8], mut f: F, masks: M,
) {
    debug_assert!(N == 4 || N == 7);
    let mut index = Index::new(data.len());
    while !data.is_empty() {
        let size = data.len().min(BATCH);
        index.build(&data[..size], &masks);
        let consumed = index.emit::<N, F>(&data[..size], &mut f);
        if consumed == 0 {
            // Unbounded rows and final non-LF tails stay on the validated general
            // scanner. Looking beyond this batch does not enlarge the index.
            let end = data[size..].iter().position(|&b| b == b'\n')
                .map_or(data.len(), |p| size + p + 1);
            super::scan_baseline::<N, _>(&data[..end], &mut f);
            data = &data[end..];
        } else {
            data = &data[consumed..];
        }
    }
}

pub(super) fn scan<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(data: &'a [u8], f: F) {
    #[cfg(target_arch = "x86_64")]
    if super::simd_backend() == "avx2" {
        if is_x86_feature_detected!("popcnt") {
            unsafe { scan_avx2::<N, F>(data, f) };
        } else {
            super::scan_baseline::<N, F>(data, f);
        }
        return;
    }
    scan_with::<N, F, _>(data, f, scalar_masks);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,popcnt")]
unsafe fn scan_avx2<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(data: &'a [u8], f: F) {
    scan_with::<N, F, _>(data, f, |bytes| unsafe { avx2_masks(bytes) });
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx2_masks(bytes: &[u8; 64]) -> (u64, u64, u64) {
    use core::arch::x86_64::*;
    let lo = _mm256_loadu_si256(bytes.as_ptr().cast());
    let hi = _mm256_loadu_si256(bytes.as_ptr().add(32).cast());
    let mask = |byte: i8| {
        let value = _mm256_set1_epi8(byte);
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, value)) as u32 as u64
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, value)) as u32 as u64) << 32)
    };
    (mask(b',' as i8), mask(b'\n' as i8), mask(b'"' as i8))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<const N: usize>(input: &[u8]) {
        let mut expected = Vec::new();
        super::super::scan_byte_oracle::<N, _>(input, |f| {
            expected.push(f.iter().map(|x| x.to_vec()).collect::<Vec<_>>());
        });
        let mut baseline = Vec::new();
        super::super::scan_baseline::<N, _>(input, |f| {
            baseline.push(f.iter().map(|x| x.to_vec()).collect::<Vec<_>>());
        });
        assert_eq!(baseline, expected, "baseline input={input:?}");
        let mut scalar = Vec::new();
        scan_with::<N, _, _>(input, |f| {
            scalar.push(f.iter().map(|x| x.to_vec()).collect::<Vec<_>>());
        }, scalar_masks);
        assert_eq!(scalar, expected, "structural scalar input={input:?}");
        let mut selected = Vec::new();
        scan::<N, _>(input, |f| {
            selected.push(f.iter().map(|x| x.to_vec()).collect::<Vec<_>>());
        });
        assert_eq!(selected, expected, "structural selected input={input:?}");
    }

    #[test]
    fn masks_match_bytes_at_every_offset() {
        for lane in 0..64 {
            for byte in 0..=255 {
                let mut block = [0u8; 64];
                block[lane] = byte;
                let expected = (
                    if byte == b',' { 1u64 << lane } else { 0 },
                    if byte == b'\n' { 1u64 << lane } else { 0 },
                    if byte == b'"' { 1u64 << lane } else { 0 },
                );
                assert_eq!(scalar_masks(&block), expected);
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    assert_eq!(unsafe { avx2_masks(&block) }, expected);
                }
            }
        }
        for (byte, expected) in [(b',', (u64::MAX, 0, 0)), (b'\n', (0, u64::MAX, 0)), (b'"', (0, 0, u64::MAX))] {
            assert_eq!(scalar_masks(&[byte; 64]), expected);
        }
    }

    #[test]
    fn exhaustive_quote_placements_and_row_recovery() {
        // Each field receives every short byte sequence over this alphabet;
        // following rows check parity recovery after malformed physical rows.
        let alphabet = b"a\",\n\r\0";
        for len in 0..=5u32 {
            for mut code in 0..alphabet.len().pow(len) {
                let mut part = Vec::new();
                for _ in 0..len {
                    part.push(alphabet[code % alphabet.len()]);
                    code /= alphabet.len();
                }
                for field in 0..7 {
                    let mut row = Vec::new();
                    for k in 0..7 {
                        if k != 0 { row.push(b','); }
                        row.extend_from_slice(if k == field { &part } else { b"\"v\"" });
                    }
                    row.extend_from_slice(b"\nx,y,z,w\na,b,c,d,e,f,g\n");
                    check::<7>(&row);
                    check::<4>(&row);
                }
            }
        }
    }

    #[test]
    fn offsets_tails_binary_and_batch_seams() {
        let rows: &[&[u8]] = &[
            b"a,b,c,d\n", b"\"a\",\"\",\"c\",d\n", b"a,b,c,\n",
            b"a,b,c,d\r\n", b"a,\0\xff,c,d\n", b"\"a,b\",c,d\n",
            b"\",b,c,d\n", b"\"\"\",b,c,d\n", b"a,b,c,d,e\n",
            b"a,b,c\n", b"\"\"\"\",b,c,d\n", b"a,b,c,\"d\"\r\n",
            b"a,b,c,d,e,f,g\n", b"\"\",b,c,d,e,f,\"\"\n",
        ];
        for offset in 0..128 {
            for row in rows {
                let mut input = vec![b'!'; offset];
                let start = input.len();
                input.extend_from_slice(&vec![b'\n'; offset]);
                input.extend_from_slice(row);
                input.extend_from_slice(b"a,b,c,d\na,b,c,d,e,f,g\n");
                for tail in 0..=1 {
                    check::<4>(&input[start..input.len() - tail]);
                    check::<7>(&input[start..input.len() - tail]);
                }
            }
        }
        for size in [63, 64, 65, 127, 128, BATCH - 1, BATCH, BATCH + 1, BATCH * 2 + 17] {
            for quoted in [false, true] {
                let mut input = b"\",bad,row,quote\n".to_vec();
                if quoted { input.push(b'"'); }
                input.extend(std::iter::repeat(b'x').take(size));
                if quoted { input.push(b'"'); }
                input.extend_from_slice(b",b,c,d\n\"a\",b,c,\"\"\na,b,c,d,e,f,g\na,b,c,");
                check::<4>(&input);
                check::<7>(&input);
            }
        }
        for offset in 0..128 {
            let mut input = vec![b'\n'; BATCH - 64 + offset];
            input.extend_from_slice(b"\"a\",\"\",c,\"d\"\n\",b,c,d\na,b,c,d,e,f,g\n");
            check::<4>(&input);
            check::<7>(&input);
        }
    }
}
