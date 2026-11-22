//! Exact 8-byte keys for IDs (`Key8`, `EKey`, `CKey`).
//!
//! All three use prefix-stripped little-endian raw 8-byte loads for the
//! common 8- and 9-byte IDs (`'u'` for `Key8`, `'e'` for `EKey`, `'k'` for `CKey`),
//! skipping `bswap`, 7-bit ASCII validation, and `_pext_u64`/`_pdep_u64` entirely.
//!
//! Arbitrary variable-width IDs (`> 9` bytes or non-prefixed `8..=9` bytes) from
//! the contiguous input mmap region are encoded zero-copy as 8-byte slice refs
//! (`REF_TAG | (len << 36) | off`) with ZERO mutexes, ZERO `HashMap` lookups, and
//! ZERO heap allocations. Non-mmap strings (e.g. unit tests, `products.csv`) fall
//! back to `OverflowStore` (`OVF_TAG | idx`).
//!
//! Tag spaces (by top byte `hi = val >> 56`) are strictly disjoint:
//! - `val == 0`: empty slice `b""` (`EMPTY`)
//! - `hi <= 0x7F` (`val != 0`): 9-byte prefix + 8 raw bytes (exact, verbatim)
//! - `hi == 0x80`: 8-byte prefix + 7 raw bytes (exact, verbatim)
//! - `hi in 0x81..=0x87`: raw 1..=7 byte string (any bytes, verbatim)
//! - `hi == 0xC0`: interned in global sharded `OverflowStore` (`OVF_TAG | idx`)
//! - `hi == 0xD0`: zero-copy slice into contiguous input mmap (`REF_TAG | (len << 36) | off`)
//! - `hi == 0xD1`: 33-byte Key8 mmap ref (`tag | (fingerprint20 << 36) | off`)
//! - `val == u64::MAX` (`hi == 0xFF`): `Key8::TOMBSTONE` (impossible for real IDs)

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering as AOrd};
use std::sync::{Mutex, OnceLock};

pub const P0: u64 = 0xa076_1d64_78bd_642f;
pub const P1: u64 = 0xe703_7ed1_a0b4_28db;

#[inline(always)]
pub fn mum(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

const OVF_TAG: u64 = 0xC000_0000_0000_0000;
const REF_TAG: u64 = 0xD000_0000_0000_0000;
const REF_TAG_BYTE: u64 = 0xD0;
const USER_REF_TAG_BYTE: u64 = 0xD1;
const REF_OFF_MASK: u64 = (1u64 << 36) - 1; // up to 64 GiB of contiguous mapped CSVs
const REF_MAX_LEN: usize = 1 << 20;         // up to 1 MiB per single ID string
const OVF_SHARD_BITS: u32 = 6;
const OVF_SHARDS: usize = 1 << OVF_SHARD_BITS;

static INPUT_BASE: AtomicUsize = AtomicUsize::new(0);
static INPUT_LEN: AtomicUsize = AtomicUsize::new(0);

/// Registers the contiguous virtual address range containing all mapped input CSVs.
/// Must be called once during single-threaded init before parsing threads start.
pub fn register_input_region(base: usize, len: usize) {
    INPUT_BASE.store(base, AOrd::Release);
    INPUT_LEN.store(len, AOrd::Release);
}

#[inline(always)]
fn is_non_inline_raw(v: u64) -> bool {
    // Includes overflow, ordinary refs, and cached user refs; excludes inline and tombstone.
    ((v >> 56) & 0xE0) == 0xC0
}

#[inline(always)]
fn is_ref_raw(v: u64) -> bool {
    (v >> 56) == REF_TAG_BYTE || (v >> 56) == USER_REF_TAG_BYTE
}

#[inline(always)]
unsafe fn ref_bytes(v: u64) -> &'static [u8] {
    let base = INPUT_BASE.load(AOrd::Relaxed);
    let off = (v & REF_OFF_MASK) as usize;
    let len = ((v >> 36) & ((REF_MAX_LEN - 1) as u64)) as usize;

    let len = if (v >> 56) == USER_REF_TAG_BYTE { 33 } else { len };
    std::slice::from_raw_parts((base + off) as *const u8, len)
}

#[inline(always)]
fn non_inline_bytes(v: u64) -> &'static [u8] {
    if is_ref_raw(v) {
        unsafe { ref_bytes(v) }
    } else {
        overflow_store().get(v & !OVF_TAG)
    }
}

#[inline(never)]
fn key_eq_slow(a: u64, b: u64) -> bool {

    if (a >> 56) == USER_REF_TAG_BYTE && (b >> 56) == USER_REF_TAG_BYTE {
        // Different cached fingerprints reject without reading either mmap slice.
        // Equal fingerprints are only a filter: compare all 33 bytes, even on collision.
        if ((a ^ b) >> 36) != 0 {
            return false;
        }
        return unsafe { ref_bytes(a) == ref_bytes(b) };
    }
    if (a >> 56) == REF_TAG_BYTE && (b >> 56) == REF_TAG_BYTE {
        if ((a ^ b) >> 36) != 0 {
            return false;
        }
        return unsafe { ref_bytes(a) == ref_bytes(b) };
    }
    non_inline_bytes(a) == non_inline_bytes(b)
}

#[inline(always)]
fn key_eq(a: u64, b: u64) -> bool {
    if a == b {
        return true;
    }
    if is_non_inline_raw(a) && is_non_inline_raw(b) {
        return key_eq_slow(a, b);
    }
    false
}

#[inline(never)]
fn key_hash_slow(v: u64, seed: u64) -> u64 {
    crate::hash::hash(non_inline_bytes(v), seed)
}

#[inline(always)]
fn key_hash(v: u64, seed: u64) -> u64 {
    if !is_non_inline_raw(v) {
        mum(v ^ seed, P0)
    } else {
        key_hash_slow(v, seed)
    }
}


#[inline(always)]
fn user_fingerprint(s: &[u8]) -> u64 {
    debug_assert_eq!(s.len(), 33);
    crate::hash::hash(s, P1) & ((1 << 20) - 1)
}


#[inline(always)]
fn fingerprint_hash(fingerprint: u64, seed: u64) -> u64 {
    mum(fingerprint ^ seed, P0)
}


#[inline(always)]
fn pack_user33(s: &[u8], fingerprint: u64) -> u64 {
    let ilen = INPUT_LEN.load(AOrd::Relaxed);
    let base = INPUT_BASE.load(AOrd::Relaxed);
    let off = (s.as_ptr() as usize).wrapping_sub(base);
    if off < ilen && 33 <= ilen - off && (off as u64) <= REF_OFF_MASK {
        (USER_REF_TAG_BYTE << 56) | (fingerprint << 36) | off as u64
    } else {
        OVF_TAG | overflow_store().intern(s)
    }
}


#[inline(always)]
fn user_key_hash(v: u64, seed: u64) -> u64 {
    if (v >> 56) == USER_REF_TAG_BYTE {
        fingerprint_hash((v >> 36) & ((1 << 20) - 1), seed)
    } else if !is_non_inline_raw(v) {
        mum(v ^ seed, P0)
    } else {
        let s = non_inline_bytes(v);
        if s.len() == 33 {
            // Owned and ordinary-ref copies must use the same hash for every seed.
            fingerprint_hash(user_fingerprint(s), seed)
        } else {
            crate::hash::hash(s, seed)
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
pub struct Key8(pub u64);

impl PartialEq for Key8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        key_eq(self.0, other.0)
    }
}
impl Eq for Key8 {}

impl Hash for Key8 {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        if !is_non_inline_raw(self.0) {
            self.0.hash(state);
        } else {
            non_inline_bytes(self.0).hash(state);
        }
    }
}

/// Exact 8-byte equality+hash key for `event_id` (`EKey`), fast-pathed for `'e'`-prefixed 8/9B IDs.
#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
pub struct EKey(pub u64);

impl PartialEq for EKey {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        key_eq(self.0, other.0)
    }
}
impl Eq for EKey {}

/// Exact 8-byte ordered key for `click_id` (`CKey`), fast-pathed for `'k'`-prefixed 8/9B IDs.
#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
pub struct CKey(pub u64);

impl PartialEq for CKey {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        key_eq(self.0, other.0)
    }
}
impl Eq for CKey {}

#[inline(always)]
fn pack_prefixed<const PREFIX: u8>(s: &[u8]) -> u64 {
    let n = s.len();
    let p = s.as_ptr();
    if n == 9 {
        unsafe {
            if *p == PREFIX {
                let v = (p.add(1) as *const u64).read_unaligned();
                if (v as i64) > 0 {
                    return v;
                }
            }
        }
        return pack_slow(s);
    }
    if n == 8 {
        unsafe {
            let w0 = (p as *const u64).read_unaligned();
            if (w0 as u8) == PREFIX {
                return (w0 >> 8) | (0x80u64 << 56);
            }
        }
        return pack_slow(s);
    }
    pack_slow(s)
}

#[inline(always)]
fn pack_prefixed_and_hash<const PREFIX: u8>(s: &[u8], seed: u64) -> (u64, u64) {
    let n = s.len();
    let p = s.as_ptr();
    if n == 9 {
        unsafe {
            if *p == PREFIX {
                let v = (p.add(1) as *const u64).read_unaligned();
                if (v as i64) > 0 {
                    return (v, mum(v ^ seed, P0));
                }
            }
        }
        return pack_slow_and_hash(s, seed);
    }
    if n == 8 {
        unsafe {
            let w0 = (p as *const u64).read_unaligned();
            if (w0 as u8) == PREFIX {
                let v = (w0 >> 8) | (0x80u64 << 56);
                return (v, mum(v ^ seed, P0));
            }
        }
        return pack_slow_and_hash(s, seed);
    }
    pack_slow_and_hash(s, seed)
}

#[inline(never)]
fn pack_slow(s: &[u8]) -> u64 {
    let n = s.len();
    if n == 0 {
        return 0;
    }
    if n <= 7 {
        let mut v = (0x80u64 | n as u64) << 56;
        for (i, &b) in s.iter().enumerate() {
            v |= (b as u64) << (i * 8);
        }
        return v;
    }
    let ilen = INPUT_LEN.load(AOrd::Relaxed);
    if ilen != 0 && n < REF_MAX_LEN {
        let base = INPUT_BASE.load(AOrd::Relaxed);
        let off = (s.as_ptr() as usize).wrapping_sub(base);
        if off < ilen && n <= ilen - off {
            return REF_TAG | ((n as u64) << 36) | (off as u64);
        }
    }
    OVF_TAG | overflow_store().intern(s)
}

#[inline(never)]
fn pack_slow_and_hash(s: &[u8], seed: u64) -> (u64, u64) {
    let v = pack_slow(s);
    let h = if !is_non_inline_raw(v) {
        mum(v ^ seed, P0)
    } else {
        crate::hash::hash(s, seed)
    };
    (v, h)
}

#[inline(always)]
fn unpack_prefixed<'a, const PREFIX: u8>(v: u64, buf: &'a mut [u8; 16]) -> &'a [u8] {
    let hi = v >> 56;
    if hi <= 0x80 && v != 0 {
        unsafe {
            *buf.get_unchecked_mut(0) = PREFIX;
            (buf.as_mut_ptr().add(1) as *mut u64).write_unaligned(v);
            let len = 9 - (hi >> 7) as usize;
            return buf.get_unchecked(..len);
        }
    }
    unpack_slow(v, buf)
}

#[inline(never)]
fn unpack_slow<'a>(v: u64, buf: &'a mut [u8; 16]) -> &'a [u8] {
    if v == 0 {
        return &[];
    }
    let hi = v >> 56;
    if hi < 0xC0 {
        let len = (hi - 0x80) as usize;
        unsafe {
            (buf.as_mut_ptr() as *mut u64).write_unaligned(v);
            return buf.get_unchecked(..len);
        }
    }
    non_inline_bytes(v)
}

#[inline(always)]
fn cmp_prefixed<const PREFIX: u8>(a: u64, b: u64) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    // Fast path: both 9-byte prefixed (bit 63 == 0, nonzero) -> bswap + cmp.
    if ((a | b) >> 63) == 0 && a != 0 && b != 0 {
        return a.swap_bytes().cmp(&b.swap_bytes());
    }
    // Fast path: both 8-byte prefixed (top byte == 0x80) -> bswap + cmp.
    if (a >> 56) == 0x80 && (b >> 56) == 0x80 {
        return a.swap_bytes().cmp(&b.swap_bytes());
    }
    cmp_prefixed_slow::<PREFIX>(a, b)
}

#[inline(never)]
fn cmp_prefixed_slow<const PREFIX: u8>(a: u64, b: u64) -> Ordering {
    let (mut ba, mut bb) = ([0u8; 16], [0u8; 16]);
    unpack_prefixed::<PREFIX>(a, &mut ba).cmp(unpack_prefixed::<PREFIX>(b, &mut bb))
}

impl EKey {
    #[allow(dead_code)]
    #[inline(always)]
    pub fn new(s: &[u8]) -> EKey {
        EKey(pack_prefixed::<b'e'>(s))
    }

    #[inline(always)]
    pub fn new_and_hash(s: &[u8], seed: u64) -> (EKey, u64) {
        let (v, h) = pack_prefixed_and_hash::<b'e'>(s, seed);
        (EKey(v), h)
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn is_non_inline(&self) -> bool {
        is_non_inline_raw(self.0)
    }

    #[inline(always)]
    pub fn hash(&self, seed: u64) -> u64 {
        key_hash(self.0, seed)
    }
}

impl CKey {
    pub const EMPTY: CKey = CKey(0);

    #[inline(always)]
    pub fn new(s: &[u8]) -> CKey {
        CKey(pack_prefixed::<b'k'>(s))
    }

    #[inline(always)]
    pub fn cmp_bytes(&self, other: &CKey) -> Ordering {
        cmp_prefixed::<b'k'>(self.0, other.0)
    }
}

impl Key8 {
    pub const EMPTY: Key8 = Key8(0);
    /// Impossible sentinel key used to mark deduplicated-away `PEv` tombstones in place
    /// (real overflow/ref keys always have top byte `0xC0`/`0xD0`/`0xD1`, never `0xFF`).
    pub const TOMBSTONE: Key8 = Key8(u64::MAX);

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn is_non_inline(&self) -> bool {
        is_non_inline_raw(self.0)
    }

    #[inline(always)]
    pub fn prefetch_ref(&self) {
        if is_ref_raw(self.0) {
            let base = INPUT_BASE.load(AOrd::Relaxed);
            let off = (self.0 & REF_OFF_MASK) as usize;
            crate::links::prefetch((base + off) as *const u8);
        }
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn is_inline(&self) -> bool {
        (self.0 >> 56) < 0xC0
    }

    #[inline(always)]
    pub fn new(s: &[u8]) -> Key8 {

        if s.len() == 33 {
            return Key8(pack_user33(s, user_fingerprint(s)));
        }
        Key8(pack_prefixed::<b'u'>(s))
    }

    #[inline(always)]
    pub fn new_and_hash(s: &[u8], seed: u64) -> (Key8, u64) {

        if s.len() == 33 {
            let fingerprint = user_fingerprint(s);
            return (Key8(pack_user33(s, fingerprint)), fingerprint_hash(fingerprint, seed));
        }
        let (v, h) = pack_prefixed_and_hash::<b'u'>(s, seed);
        (Key8(v), h)
    }

    /// Inline mixing, or full byte hashing; user-fingerprint caches a 20-bit 33-byte digest.
    #[inline(always)]
    pub fn hash(&self, seed: u64) -> u64 {
        user_key_hash(self.0, seed)
    }

    /// The ID bytes; inline keys are decoded into `buf` with a single unaligned u64 store.
    #[inline(always)]
    pub fn as_slice<'a>(&self, buf: &'a mut [u8; 16]) -> &'a [u8] {
        unpack_prefixed::<b'u'>(self.0, buf)
    }

    /// Exact byte-wise lexicographic comparison of the two IDs.
    #[inline(always)]
    pub fn cmp_bytes(&self, other: &Key8) -> Ordering {
        cmp_prefixed::<b'u'>(self.0, other.0)
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

    fn rt(k: &Key8) -> Vec<u8> {
        let mut b = [0u8; 16];
        k.as_slice(&mut b).to_vec()
    }

    #[test]
    fn key8_boundaries_and_roundtrip() {
        assert_eq!(std::mem::size_of::<Key8>(), 8);
        assert_eq!(std::mem::size_of::<CKey>(), 8);
        assert_eq!(Key8::new(&[]), Key8::EMPTY);
        assert_ne!(Key8::new(b"u\0\0\0\0\0\0\0\0"), Key8::EMPTY);
        assert_eq!(rt(&Key8::new(b"u\0\0\0\0\0\0\0\0")), b"u\0\0\0\0\0\0\0\0".to_vec());
        let bg = [b'u', 127, b'A', b' ', b'_', b'z', b'\r', b'a', b'X', b'3', b'4'];
        for n in 1..=11 {
            for pos in 0..n {
                for b in 0..=255u8 {
                    let mut s = bg[..n].to_vec();
                    s[pos] = b;
                    let k = Key8::new(&s);
                    assert_ne!(k, Key8::EMPTY);
                    assert_ne!(k, Key8::TOMBSTONE);
                    assert_eq!(rt(&k), s);
                    assert_eq!(k, Key8::new(&s));
                }
            }
        }
    }

    #[test]
    fn nul_ordering_is_bytewise() {
        let ids: Vec<&[u8]> = vec![
            b"", b"\0", b"\0\0", b"a", b"a\0", b"a\0\0", b"a\0b", b"a\x01", b"ab", b"b", b"\xff",
            b"0", b"9", b"A", b"Z", b"_", b"z", b"c_1", b"c_1958", b"C_1", b"p13649",
            b"e42huhoug", b"uduffvac", b"u8w84o3t0", b"u8w84o3t01", b"u8w84o3t012",
            b"u0000000", b"u00000000", b"u0000000\0", b"u\0\0\0\0\0\0\0", b"u\0\0\0\0\0\0\0\0",
            b"k1234567", b"k12345678", b"k12345670", b"k1234567\0", b"k1234568",
            b"aaaaaaa", b"aaaaaaaa", b"aaaaaaaa\0", b"aaaaaaaa\0\0\0\0\0\0\0", b"aaaaaaab",
            b"aaaaaaaaaa", b"aaaaaaaaab", b"aaaaaaaaaaa", b"aaaaaaaaaaaaaaa", b"aaaaaaaaaaaaaaaa",
            b"aaaaaaaaaaaaaaaaaaaaaaaaaa", b"aaaaaaaaaaaaaaaaaaaaaaaaab", b"aaaaaaab\0\0\0\0\0\0\0\0\0\0",
            b"zzzzzzzzzzzzzzzzzzzz", b"U1", b"u1", b" u1", b"u1 ", b"u-1", b"u_1",
        ];
        let keys: Vec<Key8> = ids.iter().map(|s| Key8::new(s)).collect();
        let ckeys: Vec<CKey> = ids.iter().map(|s| CKey::new(s)).collect();
        for (i, a) in ids.iter().enumerate() {
            assert_eq!(rt(&keys[i]), a.to_vec(), "roundtrip {a:?}");
            for (j, b) in ids.iter().enumerate() {
                assert_eq!(keys[i].cmp_bytes(&keys[j]), a.cmp(b), "Key8 {a:?} vs {b:?}");
                assert_eq!(keys[i] == keys[j], a == b, "Key8 eq {a:?} vs {b:?}");
                assert_eq!(ckeys[i].cmp_bytes(&ckeys[j]), a.cmp(b), "CKey {a:?} vs {b:?}");
                assert_eq!(ckeys[i] == ckeys[j], a == b, "CKey eq {a:?} vs {b:?}");
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
    fn ekey_is_exact_and_collision_free() {
        let ids: Vec<&[u8]> = vec![
            b"", b"e", b"e1", b"e123456", b"e1234567", b"e12345678", b"e123456789",
            b"u1234567", b"u12345678", b"e1234567\0", b"e123456\0", b"e1234567\x80",
            b"e123456\x80", b"e1234567\xff", b"\x80\x81\x82\x83\x84\x85\x86",
            b"e42huhoug", b"e42huhou", b"e42huhou\x88", b"long-event-id-1234567890",
        ];
        let keys: Vec<EKey> = ids.iter().map(|s| EKey::new(s)).collect();
        for i in 0..ids.len() {
            assert_eq!(keys[i], EKey::new(ids[i]));
            for j in 0..ids.len() {
                assert_eq!(keys[i] == keys[j], ids[i] == ids[j], "{:?} vs {:?}", ids[i], ids[j]);
            }
        }
        let buf = vec![b'e'; 3 * 4096];
        let base = (buf.as_ptr() as usize + 4095) & !4095;
        let off = base - buf.as_ptr() as usize + 4096;
        for n in 0..=15 {
            let s = &buf[off - n..off];
            assert_eq!(EKey::new(s), EKey::new(&s.to_vec()));
        }
    }

    #[cfg(unix)]
    #[test]
    fn mapped_user_fingerprints_match_owned_refs_and_exact_collisions() {
        struct Mapping { ptr: *mut libc::c_void, len: usize, old_base: usize, old_len: usize }
        impl Drop for Mapping {
            fn drop(&mut self) {
                register_input_region(self.old_base, self.old_len);
                assert_eq!(unsafe { libc::munmap(self.ptr, self.len) }, 0);
            }
        }
        let control = b"u0123456789abcdef0123456789abcdef";
        let mut fingerprints = HashMap::new();
        let (one, two) = (0..).find_map(|i: u32| {
            let id = format!("u{i:032x}");
            let fingerprint = user_fingerprint(id.as_bytes());
            if fingerprint == user_fingerprint(control) { return None; }
            fingerprints.insert(fingerprint, i).map(|prev| (format!("u{prev:032x}"), id))
        }).unwrap();
        let mut ids = vec![
            control.to_vec(), one.into_bytes(), two.into_bytes(),
            "\u{00e9}\u{6c49}\u{1f600}abcdefghijklmnopqrstuvwx".as_bytes().to_vec(),
        ];
        for i in 0..8u16 {
            ids.push((0..33u16).map(|j| (i * 33 + j) as u8).collect());
        }
        for len in [0, 1, 7, 8, 9, 32, 34, 300] { ids.push(vec![b'X'; len]); }
        assert!(ids[..12].iter().all(|s| s.len() == 33));
        assert_ne!(ids[1], ids[2]);
        assert_eq!(user_fingerprint(&ids[1]), user_fingerprint(&ids[2]));

        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let needed: usize = ids.iter().map(|s| (s.len() + 1) * 2).sum();
        let len = (needed + page - 1) & !(page - 1);
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE,
                                     libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0) };
        assert_ne!(ptr, libc::MAP_FAILED);
        let mapping = Mapping { ptr, len, old_base: INPUT_BASE.load(AOrd::Relaxed), old_len: INPUT_LEN.load(AOrd::Relaxed) };
        let mut offsets = Vec::new();
        let mut off = 0;
        for s in &ids {
            offsets.push(off);
            unsafe {
                std::ptr::copy_nonoverlapping(s.as_ptr(), ptr.cast::<u8>().add(off), s.len());
                std::ptr::copy_nonoverlapping(s.as_ptr(), ptr.cast::<u8>().add(off + s.len() + 1), s.len());
            }
            off += (s.len() + 1) * 2;
        }
        assert_eq!(unsafe { libc::mprotect(ptr, len, libc::PROT_READ) }, 0);
        register_input_region(ptr as usize, len);
        let trait_hash = |k: &Key8| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            Hash::hash(k, &mut hasher);
            hasher.finish()
        };
        let mut mapped_keys = Vec::new();
        for (i, s) in ids.iter().enumerate() {
            let mapped = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>().add(offsets[i]), s.len()) };
            let second = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>().add(offsets[i] + s.len() + 1), s.len()) };
            let key = Key8::new(mapped);
            let owned = Key8::new(s);
            let duplicate = Key8::new(second);
            assert_eq!(key, owned);
            assert_eq!(key, duplicate);
            assert_eq!(rt(&key), *s);
            assert_eq!(key.cmp_bytes(&owned), Ordering::Equal);
            assert_eq!(trait_hash(&key), trait_hash(&owned));
            key.prefetch_ref();
            if s.len() == 33 {
                assert_eq!(key.0 >> 56, USER_REF_TAG_BYTE);
                assert_eq!(owned.0 >> 56, OVF_TAG >> 56);
                assert!(key.is_non_inline() && !key.is_inline());
                assert_eq!(key.0 & REF_OFF_MASK, offsets[i] as u64);
                let ordinary = Key8(REF_TAG | (33 << 36) | offsets[i] as u64);
                assert_eq!(key, ordinary);
                assert_eq!(trait_hash(&key), trait_hash(&ordinary));
                assert_eq!(rt(&ordinary), *s);
                for bit in 0..64 {
                    for seed in [0, u64::MAX, 1u64 << bit, !(1u64 << bit)] {
                        assert_eq!(key.hash(seed), owned.hash(seed));
                        assert_eq!(key.hash(seed), ordinary.hash(seed));
                        let (combined, hash) = Key8::new_and_hash(mapped, seed);
                        let (fallback, fallback_hash) = Key8::new_and_hash(s, seed);
                        assert_eq!(combined, key);
                        assert_eq!(fallback, key);
                        assert_eq!(hash, key.hash(seed));
                        assert_eq!(hash, fallback_hash);
                        // Event and click keys retain ordinary refs and full-byte hashing.
                        let (event, event_hash) = EKey::new_and_hash(mapped, seed);
                        assert_eq!(event.0 >> 56, REF_TAG_BYTE);
                        assert_eq!(event_hash, crate::hash::hash(s, seed));
                        assert_eq!(event_hash, EKey::new(s).hash(seed));
                    }
                }
                assert_eq!(CKey::new(mapped).0 >> 56, REF_TAG_BYTE);
                assert_eq!(CKey::new(mapped), CKey::new(s));
            } else {
                assert_ne!(key.0 >> 56, USER_REF_TAG_BYTE);
                for seed in [0, P0, P1, u64::MAX] {
                    assert_eq!(key.hash(seed), owned.hash(seed));
                    assert_eq!(Key8::new_and_hash(mapped, seed), (key, key.hash(seed)));
                }
            }
            mapped_keys.push(key);
        }
        // Exact collisions do not merge IDs; all cross-representation ordering stays bytewise.
        assert_ne!(mapped_keys[1], mapped_keys[2]);
        assert_eq!(mapped_keys[1].hash(P0), mapped_keys[2].hash(P0));
        for (i, a) in mapped_keys.iter().enumerate() {
            for (j, b) in mapped_keys.iter().enumerate() {
                assert_eq!(a == b, ids[i] == ids[j]);
                assert_eq!(a.cmp_bytes(b), ids[i].cmp(&ids[j]));
                assert_eq!(a.cmp_bytes(&Key8::new(&ids[j])), ids[i].cmp(&ids[j]));
            }
        }
        // A differing cached fingerprint must reject before any mmap read; cached hashing too.
        assert_ne!(user_fingerprint(&ids[0]), user_fingerprint(&ids[1]));
        let expected = mapped_keys[0].hash(P1);
        assert_eq!(unsafe { libc::mprotect(ptr, len, libc::PROT_NONE) }, 0);
        assert_ne!(mapped_keys[0], mapped_keys[1]);
        assert_eq!(mapped_keys[0].hash(P1), expected);
        assert_eq!(unsafe { libc::mprotect(ptr, len, libc::PROT_READ) }, 0);
        drop(mapping);
    }

    #[test]
    fn hash_distinct_keys_never_merge() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..200_000u32 {
            let s = format!("u{i:x}");
            assert!(seen.insert(Key8::new(s.as_bytes())));
        }
        let mut cnt_hi = [0u32; 256];
        let mut cnt_lo = [0u32; 256];
        let mut cnt_tag = [0u32; 128];
        for i in 0..256_000u32 {
            let h = Key8::new(format!("u{i:08x}").as_bytes()).hash(0x9e37_79b9_7f4a_7c15);
            cnt_hi[(h >> 56) as usize] += 1;
            cnt_lo[(h & 255) as usize] += 1;
            cnt_tag[((h >> 17) & 127) as usize] += 1;
        }
        assert!(cnt_hi.iter().all(|&c| c > 700 && c < 1300), "hi: {cnt_hi:?}");
        assert!(cnt_lo.iter().all(|&c| c > 700 && c < 1300), "lo: {cnt_lo:?}");
        assert!(cnt_tag.iter().all(|&c| c > 1500 && c < 2500), "tag: {cnt_tag:?}");
    }
}
