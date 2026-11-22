//! Fast non-cryptographic 64-bit string hash (wyhash-style mixing).

const P0: u64 = 0xa076_1d64_78bd_642f;
const P1: u64 = 0xe703_7ed1_a0b4_28db;

#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

#[inline(always)]
unsafe fn r8(p: *const u8) -> u64 {
    u64::from_le((p as *const u64).read_unaligned())
}

#[inline(always)]
unsafe fn r4(p: *const u8) -> u64 {
    u32::from_le((p as *const u32).read_unaligned()) as u64
}

/// Exact 3-mum depth-2 hash for 33-byte v5 hex IDs (4 non-overlapping u64 loads).
/// Identical output to `hash(s, seed)` for `s.len() == 33`.
#[inline(always)]
pub fn hash33(s: &[u8], seed: u64) -> u64 {
    debug_assert_eq!(s.len(), 33);
    let p = s.as_ptr();
    let seed = seed ^ P0;
    unsafe {
        let s0 = mum(r8(p.add(1)) ^ P1, r8(p.add(9)) ^ seed);
        let s1 = mum(r8(p.add(17)) ^ P0, r8(p.add(25)) ^ (seed ^ (*p as u64)));
        mum(s0 ^ P1, s1 ^ (33u64 << 32))
    }
}

#[inline(always)]
pub fn hash(s: &[u8], seed: u64) -> u64 {
    let len = s.len();
    let p = s.as_ptr();
    let seed = seed ^ P0;
    unsafe {
        if len <= 16 {
            let a;
            let b;
            if len >= 4 {
                let off = (len >> 3) << 2;
                a = (r4(p) << 32) | r4(p.add(off));
                b = (r4(p.add(len - 4)) << 32) | r4(p.add(len - 4 - off));
            } else if len > 0 {
                a = ((*p as u64) << 16) | ((*p.add(len >> 1) as u64) << 8) | (*p.add(len - 1) as u64);
                b = 0;
            } else {
                a = 0;
                b = 0;
            }
            return mum(P1 ^ (len as u64), mum(a ^ P1, b ^ seed));
        }
        if len == 33 {
            // Exact 3-mum depth-2 fast path for 33-byte v5 hex IDs (4 non-overlapping u64 loads).
            let s0 = mum(r8(p.add(1)) ^ P1, r8(p.add(9)) ^ seed);
            let s1 = mum(r8(p.add(17)) ^ P0, r8(p.add(25)) ^ (seed ^ (*p as u64)));
            return mum(s0 ^ P1, s1 ^ (33u64 << 32));
        }
        if len <= 48 {
            // 2-way ILP fast path for 17..=48 byte IDs (zero loop).
            let s0 = mum(r8(p) ^ P1, r8(p.add(8)) ^ seed);
            let s1 = if len > 32 {
                mum(r8(p.add(16)) ^ P0, r8(p.add(24)) ^ seed)
            } else {
                seed
            };
            let a = r8(p.add(len - 16));
            let b = r8(p.add(len - 8));
            return mum(P1 ^ (len as u64), mum(a ^ s0, b ^ s1));
        }
        let mut s = seed;
        let mut i = len;
        let mut q = p;
        while i > 16 {
            s = mum(r8(q) ^ P1, r8(q.add(8)) ^ s);
            q = q.add(16);
            i -= 16;
        }
        let a = r8(p.add(len - 16));
        let b = r8(p.add(len - 8));
        mum(P1 ^ (len as u64), mum(a ^ P1, b ^ s))
    }
}
