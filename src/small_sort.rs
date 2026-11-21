//! Sort 5–32 packed event words in fixed eight-lane compare/exchange networks.
use super::Ev8;

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Ev8;
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn step<const DIST: i32, const MAX: u8>(x: __m512i) -> __m512i {
        let y = match DIST {
            1 => _mm512_permutex_epi64::<0xb1>(x),
            2 => _mm512_permutex_epi64::<0x4e>(x),
            4 => _mm512_shuffle_i64x2::<0x4e>(x, x),
            _ => unreachable!(),
        };
        _mm512_mask_max_epu64(_mm512_min_epu64(x, y), MAX, x, y)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn finish(x: __m512i) -> __m512i {
        step::<1, 0xaa>(step::<2, 0xcc>(step::<4, 0xf0>(x)))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sort8(x: __m512i) -> __m512i {
        finish(step::<1, 0x5a>(step::<2, 0x3c>(step::<1, 0x66>(x))))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn reverse(x: __m512i) -> __m512i {
        _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), x)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn split(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
        (_mm512_min_epu64(x, y), _mm512_max_epu64(x, y))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn merge16(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
        let (low, high) = split(x, reverse(y));
        (finish(low), finish(high))
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn sort(rows: &mut [Ev8]) {
        // Inactive lanes are padded with the maximum word. A masked tail load
        // starts inside the slice and never accesses memory beyond its end.
        let n = rows.len();
        let ptr = rows.as_mut_ptr().cast::<i64>();
        let pad = _mm512_set1_epi64(-1);
        let read = |i: usize| {
            if i >= n { pad } else {
                let mask = (1u16 << (n - i).min(8)) - 1;
                _mm512_mask_loadu_epi64(pad, mask as u8, ptr.add(i))
            }
        };
        let write = |i: usize, value| {
            if i < n {
                let mask = (1u16 << (n - i).min(8)) - 1;
                _mm512_mask_storeu_epi64(ptr.add(i), mask as u8, value);
            }
        };
        let a = sort8(read(0));
        if n <= 8 { write(0, a); return; }
        let (a, b) = merge16(a, sort8(read(8)));
        if n <= 16 { write(0, a); write(8, b); return; }
        let (c, d) = merge16(sort8(read(16)), sort8(read(24)));
        let (al, ah) = split(a, reverse(d));
        let (bl, bh) = split(b, reverse(c));
        let (a, b) = split(al, bl);
        let (c, d) = split(ah, bh);
        write(0, finish(a)); write(8, finish(b));
        write(16, finish(c)); write(24, finish(d));
    }
}

#[inline(always)]
pub(super) fn try_sort(rows: &mut [Ev8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if (5..=32).contains(&rows.len()) && is_x86_feature_detected!("avx512f") {
        unsafe { x86::sort(rows); }
        return true;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = rows;
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sorts_every_tail_and_preserves_full_payloads() {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        for n in 0..=35 {
            for iteration in 0..120 {
                let mut storage = vec![Ev8(0x9876); n + 2];
                for x in &mut storage[1..=n] {
                    rng ^= rng << 7; rng ^= rng >> 9;
                    x.0 = match iteration % 5 { 0 => u64::MAX, 1 => rng & 1,
                        2 => rng & 0xffff, 3 => rng | (u32::MAX as u64) << 32, _ => rng };
                }
                let mut gold = storage.clone();
                gold[1..=n].sort_unstable();
                let rows = &mut storage[1..=n];
                if !try_sort(rows) { rows.sort_unstable(); }
                assert!(storage == gold, "length {n} iteration {iteration}");
            }
        }
    }
}
