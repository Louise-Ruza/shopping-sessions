//! Exact 8-byte keys for IDs (`Key8`).
//!
//! Any ID of length 1..=10 using the 63-character sorted ASCII alphabet
//! `0-9`, `A-Z`, `_`, `a-z` is packed inline into 60 big-endian 6-bit slots
//! (codes 1..=63; 0 = end-of-string padding; bit 63 = 0). Because the 6-bit
//! codes are strictly monotonically increasing in ASCII order and 0 < 1,
//! comparing two inline `Key8` values as plain `u64` is **100% identical**
//! to byte-wise lexicographic comparison, and equality is plain `u64` equality.
//!
//! Any longer ID (>10 bytes) or ID containing any byte outside `[0-9A-Za-z_]`
//! (embedded NULs, hyphens, Unicode, spaces, etc.) is interned in a global,
//! thread-safe sharded `OverflowStore` with bit 63 = 1 (`OVF_BIT`). Equal
//! overflow IDs always receive the exact same index, so equality remains a
//! single `u64` comparison everywhere, while `cmp_bytes` falls back to exact
//! byte slice comparison whenever either operand has `OVF_BIT` set.
//!
//! With `ascii-keys`, inline keys instead contain nine big-endian 7-bit slots
//! for nonzero ASCII bytes. Lengths above nine, NULs, and high bytes use overflow.
//! Zero padding and the unchanged bit-63 tag preserve exact ordering/equality.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub const P0: u64 = 0xa076_1d64_78bd_642f;
pub const P1: u64 = 0xe703_7ed1_a0b4_28db;

#[inline(always)]
pub fn mum(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

const OVF_BIT: u64 = 1u64 << 63;
const OVF_SHARD_BITS: u32 = 6;
const OVF_SHARDS: usize = 1 << OVF_SHARD_BITS;





#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
#[repr(transparent)]
pub struct Key8(pub u64);

/// Pack eight big-endian nonzero ASCII bytes directly into 56 ordered bits.

#[inline(always)]
fn pack8_ascii(word: u64) -> Option<u64> {
    // A valid byte is 1..127, so subtracting one cannot borrow across its lane.
    // A zero causes a high bit in the subtraction; high input bytes fail directly.
    if (word | word.wrapping_sub(0x0101_0101_0101_0101)) & 0x8080_8080_8080_8080 != 0 {
        return None;
    }
    let pairs = (word & 0x007F_007F_007F_007F) | ((word >> 1) & 0x3F80_3F80_3F80_3F80);
    let quads = (pairs & 0x0000_3FFF_0000_3FFF) | ((pairs >> 2) & 0x0FFF_C000_0FFF_C000);
    Some((quads & 0x0000_0000_0FFF_FFFF) | ((quads >> 4) & 0x00FF_FFFF_F000_0000))
}



impl Key8 {
    pub const EMPTY: Key8 = Key8(0);

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub fn is_inline(&self) -> bool {
        (self.0 & OVF_BIT) == 0
    }


    #[inline(always)]
    pub fn new(s: &[u8]) -> Key8 {
        let n = s.len();
        if n == 8 || n == 9 {
            let word = u64::from_be_bytes(s[..8].try_into().unwrap());
            if let Some(packed) = pack8_ascii(word) {
                if n == 8 {
                    return Key8(packed << 7);
                }
                if s[8].wrapping_sub(1) < 127 {
                    return Key8((packed << 7) | s[8] as u64);
                }
            }
            return Key8::overflow(s);
        }
        if n < 8 {
            let mut packed = 0u64;
            for &byte in s {
                if byte.wrapping_sub(1) >= 127 {
                    return Key8::overflow(s);
                }
                packed = (packed << 7) | byte as u64;
            }
            return Key8(packed << ((9 - n) * 7));
        }
        Key8::overflow(s)
    }





    #[cold]
    fn overflow(s: &[u8]) -> Key8 {
        let idx = overflow_store().intern(s);
        Key8(OVF_BIT | idx)
    }

    /// Single-multiply wyhash-grade 64-bit hash with full avalanche across all 64 bits.
    #[inline(always)]
    pub fn hash(&self, seed: u64) -> u64 {
        mum(self.0 ^ seed ^ P0, self.0.wrapping_add(P1) ^ seed)
    }

    /// The ID bytes; inline keys are decoded into `buf`.
    #[inline(always)]
    pub fn as_slice<'a>(&self, buf: &'a mut [u8; 16]) -> &'a [u8] {
        if self.is_inline() {

            {
                let mut value = self.0 << 1;
                let mut len = 0;
                while value != 0 {
                    buf[len] = (value >> 57) as u8;
                    len += 1;
                    value <<= 7;
                }
                &buf[..len]
            }

        } else {
            overflow_store().get(self.0 & !OVF_BIT)
        }
    }

    /// Exact byte-wise lexicographic comparison of the two IDs.
    #[inline(always)]
    pub fn cmp_bytes(&self, other: &Key8) -> Ordering {
        if ((self.0 | other.0) & OVF_BIT) == 0 {
            return self.0.cmp(&other.0);
        }
        self.cmp_slow(other)
    }

    #[cold]
    fn cmp_slow(&self, other: &Key8) -> Ordering {
        if self == other {
            return Ordering::Equal;
        }
        let (mut a, mut b) = ([0u8; 16], [0u8; 16]);
        self.as_slice(&mut a).cmp(other.as_slice(&mut b))
    }
}

// ---------------------------------------------------------------------------

struct OvfShard {
    map: HashMap<&'static [u8], u64>,
    strs: Vec<&'static [u8]>,
}

pub struct OverflowStore {
    shards: Vec<Mutex<OvfShard>>,
}

pub fn overflow_store() -> &'static OverflowStore {
    static S: OnceLock<OverflowStore> = OnceLock::new();
    S.get_or_init(|| OverflowStore {
        shards: (0..OVF_SHARDS).map(|_| Mutex::new(OvfShard { map: HashMap::new(), strs: Vec::new() })).collect(),
    })
}

impl OverflowStore {
    fn intern(&self, s: &[u8]) -> u64 {
        let sh = (crate::hash::hash(s, P1) >> (64 - OVF_SHARD_BITS)) as usize;
        let mut g = self.shards[sh].lock().unwrap();
        if let Some(&i) = g.map.get(s) {
            return i;
        }
        let owned: &'static [u8] = Box::leak(s.to_vec().into_boxed_slice());
        let local = g.strs.len() as u64;
        assert!(local < (1 << (62 - OVF_SHARD_BITS)), "overflow store full");
        let idx = (local << OVF_SHARD_BITS) | sh as u64;
        g.strs.push(owned);
        g.map.insert(owned, idx);
        idx
    }

    fn get(&self, idx: u64) -> &'static [u8] {
        let sh = (idx as usize) & (OVF_SHARDS - 1);
        self.shards[sh].lock().unwrap().strs[(idx >> OVF_SHARD_BITS) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn ascii_word_packing_matches_every_adjacent_byte_pair() {
        for fill in [1u8, 127] {
            for position in 0..7 {
                for pair in 0..=u16::MAX {
                    let mut bytes = [fill; 8];
                    bytes[position..position + 2].copy_from_slice(&pair.to_le_bytes());
                    let expected = bytes.iter().try_fold(0u64, |acc, &c| {
                        (c != 0 && c < 128).then_some((acc << 7) | c as u64)
                    });
                    assert_eq!(pack8_ascii(u64::from_be_bytes(bytes)), expected, "{bytes:?}");
                }
            }
        }
    }


    #[test]
    fn ascii_key_boundaries_roundtrip_without_aliasing() {
        assert_eq!(std::mem::size_of::<Key8>(), 8);
        assert_eq!(Key8::new(&[]), Key8::EMPTY);
        assert_eq!(Key8::new(&[127; 9]).0, OVF_BIT - 1);
        let background = [1, 127, b'A', b' ', b'_', b'z', b'\r', b'a', b'X', b'3', b'4'];
        for n in 1..=11 {
            for position in 0..n {
                for byte in 0..=255u8 {
                    let mut text = background[..n].to_vec();
                    text[position] = byte;
                    let key = Key8::new(&text);
                    let inline = n <= 9 && text.iter().all(|&c| c != 0 && c < 128);
                    assert_eq!(key.is_inline(), inline, "{text:?}");
                    if inline {
                        let value = text.iter().fold(0u64, |acc, &c| (acc << 7) | c as u64);
                        assert_eq!(key.0, value << ((9 - n) * 7));
                    }
                    assert_eq!(rt(&key), text);
                    assert_eq!(key, Key8::new(&text));
                }
            }
        }
        let mut values: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        values.extend([vec![], vec![127; 8], vec![127; 9], vec![127; 10],
                       b"aaaaaaaaa".to_vec(), b"aaaaaaaaa\0".to_vec(), b"aaaaaaaaaa".to_vec()]);
        for a in &values {
            for b in &values {
                let (ka, kb) = (Key8::new(a), Key8::new(b));
                assert_eq!(ka == kb, a == b);
                assert_eq!(ka.cmp_bytes(&kb), a.cmp(b));
            }
        }
    }


    #[test]
    fn ascii_keys_sort_arbitrary_strings_exactly() {
        let mut seed = 1729u64;
        let mut rows = Vec::new();
        for i in 0..10_000 {
            let mut text = Vec::new();
            for _ in 0..i % 25 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                text.push((seed >> 32) as u8);
            }
            let key = Key8::new(&text);
            assert_eq!(rt(&key), text);
            rows.push((key, text));
        }
        rows.sort_by(|a, b| a.0.cmp_bytes(&b.0));
        for pair in rows.windows(2) {
            assert!(pair[0].1 <= pair[1].1);
            assert_eq!(pair[0].0 == pair[1].0, pair[0].1 == pair[1].1);
        }
    }



    fn rt(k: &Key8) -> Vec<u8> {
        let mut b = [0u8; 16];
        k.as_slice(&mut b).to_vec()
    }

    #[test]
    fn nul_ordering_is_bytewise() {
        let ids: Vec<&[u8]> = vec![
            b"", b"\0", b"\0\0", b"a", b"a\0", b"a\0\0", b"a\0b", b"a\x01", b"ab", b"b", b"\xff",
            b"0", b"9", b"A", b"Z", b"_", b"z", b"c_1", b"c_1958", b"C_1", b"p13649",
            b"e42huhoug", b"uduffvac", b"u8w84o3t0", b"u8w84o3t01", b"u8w84o3t012",
            b"aaaaaaa", b"aaaaaaaa", b"aaaaaaaa\0", b"aaaaaaaa\0\0\0\0\0\0\0", b"aaaaaaab",
            b"aaaaaaaaaa", b"aaaaaaaaab", b"aaaaaaaaaaa", b"aaaaaaaaaaaaaaa", b"aaaaaaaaaaaaaaaa",
            b"aaaaaaaaaaaaaaaaaaaaaaaaaa", b"aaaaaaaaaaaaaaaaaaaaaaaaab", b"aaaaaaab\0\0\0\0\0\0\0\0\0\0",
            b"zzzzzzzzzzzzzzzzzzzz", b"U1", b"u1", b" u1", b"u1 ", b"u-1", b"u_1",
        ];
        let keys: Vec<Key8> = ids.iter().map(|s| Key8::new(s)).collect();
        for (i, a) in ids.iter().enumerate() {
            assert_eq!(rt(&keys[i]), a.to_vec(), "roundtrip {a:?}");
            for (j, b) in ids.iter().enumerate() {
                assert_eq!(keys[i].cmp_bytes(&keys[j]), a.cmp(b), "{a:?} vs {b:?}");
                assert_eq!(keys[i] == keys[j], a == b, "eq {a:?} vs {b:?}");
                if keys[i].is_inline() && keys[j].is_inline() {
                    assert_eq!(keys[i].cmp(&keys[j]), a.cmp(b));
                }
            }
        }
    }

    #[test]
    fn overflow_is_exact_and_interned() {
        let a = vec![b'x'; 300];
        let mut b = a.clone();
        b[299] = b'y';
        let ka = Key8::new(&a);
        let kb = Key8::new(&b);
        assert!(!ka.is_inline() && !kb.is_inline());
        assert_ne!(ka, kb);
        assert_eq!(ka, Key8::new(&a.clone()));
        assert_eq!(rt(&ka), a);
        assert_eq!(ka.cmp_bytes(&kb), Ordering::Less);
        // Concurrent interning returns one index per distinct string.
        let ks: Vec<Vec<Key8>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..8)
                .map(|_| s.spawn(|| (0..500).map(|i| Key8::new(format!("long-identifier-{i:08}").as_bytes())).collect()))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for v in &ks[1..] {
            assert_eq!(v, &ks[0]);
        }
    }

    #[test]
    fn page_edge_reads() {
        let mut buf = vec![0u8; 3 * 4096];
        let base = (buf.as_ptr() as usize + 4095) & !4095;
        let off = base - buf.as_ptr() as usize + 4096;
        for n in 0..=15 {
            for (i, c) in buf[off - n..off].iter_mut().enumerate() {
                *c = b'a' + i as u8;
            }
            let s = &buf[off - n..off];
            assert_eq!(rt(&Key8::new(s)), s.to_vec());
        }
    }



    #[test]
    fn hash_distinct_keys_never_merge() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..200_000u32 {
            let s = format!("u{i:x}");
            assert!(seen.insert(Key8::new(s.as_bytes())));
        }
        let mut cnt = [0u32; 256];
        for i in 0..256_000u32 {
            cnt[(Key8::new(format!("e{i}").as_bytes()).hash(7) >> 56) as usize] += 1;
        }
        assert!(cnt.iter().all(|&c| c > 700 && c < 1300), "{cnt:?}");
    }
}
