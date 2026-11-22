//! Single-pass guarded AVX2+PCLMULQDQ frontends for events (96B, 7 fields) and
//! clicks (64B, 4 fields).
//!
//! Scans each row directly into bitmasks (commas, newlines, quotes) without
//! intermediate batch offset arrays. Every fast-path assumption is checked before
//! emitting; any other physical row delegates to exact `csv::scan` so semantics
//! are 100% identical on all inputs.

use crate::csv;
use crate::key::{EKey, Key8};
use crate::links::SEED_A;
use crate::ts::{parse_price, DateCache};
use crate::SEED_ID;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event<'a> {
    pub ek: EKey,
    pub eh: u64,
    pub uid_k: Key8,
    pub uh: u64,
    pub uid_bytes: &'a [u8],
    pub product: &'a [u8],
    pub timestamp: Option<i64>,
    pub ing_day_ms: i64,
    pub ing_bcd: u32,
    pub kind: u8, // 0 invalid (then `product` is empty); 1 view/cart; 2 purchase
    pub price: u32,
}

pub trait EventSink<'a> {
    #[inline(always)]
    fn prefetch_keys(&self, _ek_long: bool, _eh: u64, _uid_long: bool, _uh: u64) {}
    fn on_event(&mut self, ev: Event<'a>);
}

impl<'a, F: FnMut(Event<'a>)> EventSink<'a> for F {
    #[inline(always)]
    fn on_event(&mut self, ev: Event<'a>) {
        self(ev);
    }
}

pub trait ClickSink<'a> {
    fn on_click(&mut self, fields: &[&'a [u8]; 4]);
}

impl<'a, F: FnMut(&[&'a [u8]; 4])> ClickSink<'a> for F {
    #[inline(always)]
    fn on_click(&mut self, fields: &[&'a [u8]; 4]) {
        self(fields);
    }
}

pub trait LinkSink<'a> {
    fn on_link(&mut self, a: &'a [u8], b: &'a [u8]);
}

impl<'a, F: FnMut(&'a [u8], &'a [u8])> LinkSink<'a> for F {
    #[inline(always)]
    fn on_link(&mut self, a: &'a [u8], b: &'a [u8]) {
        self(a, b);
    }
}

#[cold]
#[inline(never)]
fn fallback<'a, S: EventSink<'a>>(data: &'a [u8], dates: &DateCache, f: &mut S) {
    csv::scan::<7, _>(data, |fields| {
        let kind = match fields[4] {
            b"view" | b"cart" => 1,
            b"purchase" => 2,
            _ => 0,
        };
        let (timestamp, ing_day_ms, ing_bcd) = dates.parse_event_ingest_bcd(fields[2], fields[3]);
        let (ek, eh) = EKey::new_and_hash(fields[0], SEED_ID);
        let (uid_k, uh) = Key8::pack_inline_or_sentinel(fields[1], SEED_A);
        f.on_event(Event {
            ek,
            eh,
            uid_k,
            uh,
            uid_bytes: fields[1],
            // An unknown kind empties `product`, so the sink's empty-product check rejects it
            // and the per-row handler never tests `kind == 0`.
            product: if kind == 0 { &[] } else { fields[5] },
            timestamp,
            ing_day_ms,
            ing_bcd,
            kind,
            price: if kind == 2 { parse_price(fields[6]) } else { 0 },
        });
    });
}

#[inline(always)]
fn scalar_masks32(p: *const u8) -> (u32, u32, u32) {
    const LO7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    #[inline(always)]
    fn eq_mask(word: u64, value: u64) -> u64 {
        let y = word ^ value;
        let high = !(((y & LO7).wrapping_add(LO7)) | y | LO7);
        (high >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56
    }
    let (mut c, mut l, mut q) = (0u32, 0u32, 0u32);
    for i in 0..4 {
        let w = unsafe { (p.add(i * 8) as *const u64).read_unaligned() };
        c |= (eq_mask(w, 0x2c2c_2c2c_2c2c_2c2c) as u32) << (i * 8);
        l |= (eq_mask(w, 0x0a0a_0a0a_0a0a_0a0a) as u32) << (i * 8);
        q |= (eq_mask(w, 0x2222_2222_2222_2222) as u32) << (i * 8);
    }
    (c, l, q)
}

#[inline(always)]
fn scalar_masks96(data: &[u8; 96]) -> (u64, u32, u64, u32, u64, u32) {
    let p = data.as_ptr();
    let (c0, l0, q0) = scalar_masks32(p);
    let (c1, l1, q1) = scalar_masks32(unsafe { p.add(32) });
    let (c2, l2, q2) = scalar_masks32(unsafe { p.add(64) });
    (
        (c0 as u64) | ((c1 as u64) << 32),
        c2,
        (l0 as u64) | ((l1 as u64) << 32),
        l2,
        (q0 as u64) | ((q1 as u64) << 32),
        q2,
    )
}

#[inline(always)]
fn next_comma3(c_lo: &mut u64, c_mid: &mut u64, c_top: &mut u32) -> usize {
    if *c_lo != 0 {
        let pos = c_lo.trailing_zeros() as usize;
        *c_lo &= *c_lo - 1;
        pos
    } else if *c_mid != 0 {
        let pos = 64 + c_mid.trailing_zeros() as usize;
        *c_mid &= *c_mid - 1;
        pos
    } else {
        let pos = 128 + c_top.trailing_zeros() as usize;
        *c_top &= *c_top - 1;
        pos
    }
}

#[inline(always)]
fn parse_row_tail<'a, S: EventSink<'a>>(
    data: &'a [u8],
    end: usize,
    c0: usize,
    c1: usize,
    c2: usize,
    c5: usize,
    quotes: Option<u8>,
    dates: &DateCache,
    sink: &S,
) -> Option<Event<'a>> {
    let quoted = |field: u32, read: bool| -> usize {
        match quotes {
            Some(bits) => usize::from((bits >> field) & 1),
            None => usize::from(read),
        }
    };
    let p = data.as_ptr();
    unsafe {
        // Callers pass exactly six commas, so `c2 + 3 <= c5` and every read below up to the
        // ingest field stays before `c5`. An empty raw ts (`c1 + 1 == c2`) reads the comma at
        // `c1 + 1` and gets `tlen == 0`, which the timestamp dispatch sends to `fallback`.
        let tq = quoted(2, *p.add(c1 + 1) == b'"');
        let ta = c1 + 1 + tq;
        let tlen = (c2 - c1 - 1).wrapping_sub(tq << 1);
        let iq = quoted(3, *p.add(c2 + 1) == b'"');
        let ia = c2 + 1 + iq;
        let c3 = ia + 20 + iq;
        // One branch: `c3 >= c5` alone rejects; otherwise the byte at `c3` must be a comma.
        if (c3 >= c5) | (*p.add(c3.min(c5)) != b',') {
            return None;
        }
        let q0 = quoted(0, c0 > 0 && *p == b'"');
        let q1 = quoted(1, c0 + 1 < c1 && *p.add(c0 + 1) == b'"');
        let eid = data.get_unchecked(q0..c0 - q0);
        let uid = data.get_unchecked(c0 + 1 + q1..c1 - q1);
        let (ek, eh) = EKey::new_and_hash(eid, SEED_ID);
        let (uid_k, uh) = Key8::pack_inline_or_sentinel(uid, SEED_A);
        sink.prefetch_keys(ek.is_non_inline(), eh, uid_k.0 == Key8::NON_INLINE_SENTINEL.0, uh);

        let (timestamp, ing_day_ms, ing_bcd) = if tlen == 20 {
            dates.parse_event_ingest_bcd(data.get_unchecked(ta..ta + 20), data.get_unchecked(ia..ia + 20))
        } else if tlen == 13 {
            let (id, ib) = dates.parse_f1_ingest_bcd(data.get_unchecked(ia..ia + 20));
            (DateCache::parse_ts_13(data.get_unchecked(ta..ta + 13)), id, ib)
        } else if tlen == 25 {
            dates.parse_f2_ingest_bcd(data.get_unchecked(ta..ta + 25), data.get_unchecked(ia..ia + 20))
        } else {
            return None;
        };
        let kq = quoted(4, *p.add(c3 + 1) == b'"');
        let ka = c3 + 1 + kq;
        // No `ka + 4 + kq < c5` bound: a matching `view`/`cart` followed by a comma makes that
        // comma the first one after `c3`, i.e. `c4 < c5`. Reads reach at most `end + 3`, which
        // stays inside the input (`INPUT_TAIL` slack after every file).
        let (kind, c4) = {
            let k4 = (p.add(ka) as *const u32).read_unaligned();
            let is_vc = ((k4 == u32::from_ne_bytes(*b"view")) | (k4 == u32::from_ne_bytes(*b"cart")))
                & (*p.add(ka + 4 + kq) == b',');
            if is_vc {
                (1u8, ka + 4 + kq)
            } else if ka + 8 + kq < c5
                && (p.add(ka) as *const u64).read_unaligned() == u64::from_ne_bytes(*b"purchase")
                && *p.add(ka + 8 + kq) == b','
            {
                (2u8, ka + 8 + kq)
            } else {
                return None;
            }
        };
        let pq = quoted(5, c4 + 1 < c5 && *p.add(c4 + 1) == b'"');
        let price = if kind == 2 {
            let prq = quoted(6, c5 + 1 < end && *p.add(c5 + 1) == b'"');
            parse_price(data.get_unchecked(c5 + 1 + prq..end - prq))
        } else {
            0
        };
        Some(Event {
            ek,
            eh,
            uid_k,
            uh,
            uid_bytes: uid,
            product: data.get_unchecked(c4 + 1 + pq..c5 - pq),
            timestamp,
            ing_day_ms,
            ing_bcd,
            kind,
            price,
        })
    }
}

#[inline(always)]
fn parse_row_hi<'a, S: EventSink<'a>, P: Fn(u64) -> u64>(
    data: &'a [u8; 96],
    end_hi: usize,
    c_lo: u64,
    cm2: u32,
    q_lo: u64,
    qt2: u32,
    dates: &DateCache,
    parity64: &P,
    sink: &S,
) -> Option<Event<'a>> {
    let end_bit_hi = 1u32 << end_hi;
    let below_hi = end_bit_hi - 1;
    let c_hi = cm2 & below_hi;
    if c_lo.count_ones() + c_hi.count_ones() != 6 {
        return None;
    }
    let q_hi = qt2 & below_hi;
    let p_lo = parity64(q_lo);
    let p_hi_raw = parity64(q_hi as u64) as u32;
    let seps_lo = c_lo;
    let seps_hi = c_hi | end_bit_hi;
    let edges_lo = (seps_lo << 1) | 1 | (seps_lo >> 1) | ((seps_hi as u64) << 63);
    let edges_hi = (seps_hi << 1) | ((seps_lo >> 63) as u32) | (seps_hi >> 1);
    let p_hi = p_hi_raw ^ 0u32.wrapping_sub((p_lo >> 63) as u32);
    if ((q_lo & !edges_lo) | (p_lo & seps_lo)) != 0 || ((q_hi & !edges_hi) | (p_hi & seps_hi)) != 0 {
        return None;
    }
    let (c0, c1, c2) = if c_lo != 0 {
        let c0 = c_lo.trailing_zeros() as usize;
        let low1 = c_lo & c_lo.wrapping_sub(1);
        if low1 != 0 {
            let c1 = low1.trailing_zeros() as usize;
            let low2 = low1 & low1.wrapping_sub(1);
            if low2 != 0 {
                (c0, c1, low2.trailing_zeros() as usize)
            } else {
                (c0, c1, 64 + c_hi.trailing_zeros() as usize)
            }
        } else {
            let mut clo = 0u64;
            let mut cmid = c_hi as u64;
            let mut ctop = 0u32;
            (c0, next_comma3(&mut clo, &mut cmid, &mut ctop), next_comma3(&mut clo, &mut cmid, &mut ctop))
        }
    } else {
        let mut clo = 0u64;
        let mut cmid = c_hi as u64;
        let mut ctop = 0u32;
        (
            next_comma3(&mut clo, &mut cmid, &mut ctop),
            next_comma3(&mut clo, &mut cmid, &mut ctop),
            next_comma3(&mut clo, &mut cmid, &mut ctop),
        )
    };
    let c5 = if c_hi != 0 {
        64 + (31 - c_hi.leading_zeros() as usize)
    } else {
        63 - c_lo.leading_zeros() as usize
    };
    parse_row_tail(data, 64 + end_hi, c0, c1, c2, c5, None, dates, sink)
}

#[inline(always)]
fn parse_row_lo<'a, S: EventSink<'a>, P: Fn(u64) -> u64>(
    data: &'a [u8; 96],
    end: usize,
    c_lo: u64,
    q_lo: u64,
    dates: &DateCache,
    parity64: &P,
    sink: &S,
) -> Option<Event<'a>> {
    let end_bit = 1u64 << end;
    let below = end_bit - 1;
    let c = c_lo & below;
    if c.count_ones() != 6 {
        return None;
    }
    let c0 = c.trailing_zeros() as usize;
    let low1 = c & c.wrapping_sub(1);
    let c1 = low1.trailing_zeros() as usize;
    let low2 = low1 & low1.wrapping_sub(1);
    let c2 = low2.trailing_zeros() as usize;
    let q = q_lo & below;
    let seps = c | end_bit;
    let edges = (seps << 1) | 1 | (seps >> 1);
    if ((q & !edges) | (parity64(q) & seps)) != 0 {
        return None;
    }
    let c5 = 63 - c.leading_zeros() as usize;
    parse_row_tail(data, end, c0, c1, c2, c5, None, dates, sink)
}

#[inline(always)]
fn parse_row_mid128<'a, S: EventSink<'a>, P: Fn(u64) -> u64>(
    data: &'a [u8],
    end_mid: usize,
    c_lo: u64,
    c_mid_raw: u64,
    q_lo: u64,
    q_mid_raw: u64,
    dates: &DateCache,
    parity64: &P,
    sink: &S,
) -> Option<Event<'a>> {
    let end_bit_mid = 1u64 << end_mid;
    let below_mid = end_bit_mid - 1;
    let c_mid = c_mid_raw & below_mid;
    let q_mid = q_mid_raw & below_mid;
    let seps_lo = c_lo;
    let seps_mid = c_mid | end_bit_mid;
    let edges_lo = (seps_lo << 1) | 1 | (seps_lo >> 1) | (seps_mid << 63);
    let edges_mid = (seps_mid << 1) | (seps_lo >> 63) | (seps_mid >> 1);
    let p_lo = parity64(q_lo);
    let p_mid = parity64(q_mid) ^ 0u64.wrapping_sub(p_lo >> 63);
    // Field count and quote legality share one (never-taken on clean rows) branch.
    if (c_lo.count_ones() + c_mid.count_ones() != 6)
        | (((q_lo & !edges_lo) | (p_lo & seps_lo) | (q_mid & !edges_mid) | (p_mid & seps_mid)) != 0)
    {
        return None;
    }
    // 98.3% of v5 rows have >= 2 commas in c_lo (`low1 != 0`); branchless cmov picks c2 from low2 vs c_mid!
    let low1 = c_lo & c_lo.wrapping_sub(1);
    let (c0, c1, c2) = if low1 != 0 {
        let c0 = c_lo.trailing_zeros() as usize;
        let c1 = low1.trailing_zeros() as usize;
        let low2 = low1 & low1.wrapping_sub(1);
        // A (98%-predictable) branch rather than a cmov: the `c_mid` tzcnt competes for port 1
        // with the row's other tzcnt/lzcnt/popcnt/pext/mulx ops.
        let c2 = if low2 != 0 {
            low2.trailing_zeros() as usize
        } else {
            64 + core::hint::black_box(c_mid).trailing_zeros() as usize
        };
        (c0, c1, c2)
    } else {
        let mut clo = c_lo;
        let mut cmid = c_mid;
        let mut ctop = 0u32;
        (
            next_comma3(&mut clo, &mut cmid, &mut ctop),
            next_comma3(&mut clo, &mut cmid, &mut ctop),
            next_comma3(&mut clo, &mut cmid, &mut ctop),
        )
    };
    let c5 = if c_mid != 0 {
        64 + (63 - c_mid.leading_zeros() as usize)
    } else {
        63 - c_lo.leading_zeros() as usize
    };
    // Extract one opening-quote bit per field from the validated masks.
    // This removes dependent byte loads between discovering separators and
    // decoding the two IDs, timestamps and event type.
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    let quotes = unsafe {
        use core::arch::x86_64::_pext_u64;
        let starts_lo = (c_lo << 1) | 1;
        let starts_hi = (c_mid << 1) | (c_lo >> 63);
        let a = _pext_u64(q_lo, starts_lo);
        let b = _pext_u64(q_mid, starts_hi);
        // popcnt(starts_lo) from the field-count popcnt(c_lo): one less port-1 op per row.
        let n_lo = c_lo.count_ones() + 1 - (c_lo >> 63) as u32;
        Some((a | (b << n_lo)) as u8)
    };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    let quotes = None;
    parse_row_tail(data, 64 + end_mid, c0, c1, c2, c5, quotes, dates, sink)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn parse_row_top160<'a, S: EventSink<'a>, P: Fn(u64) -> u64>(
    data: &'a [u8],
    end_top: usize,
    c_lo: u64,
    c_mid: u64,
    cm4: u32,
    q_lo: u64,
    q_mid: u64,
    qt4: u32,
    dates: &DateCache,
    parity64: &P,
    sink: &S,
) -> Option<Event<'a>> {
    let end_bit_top = 1u32 << end_top;
    let below_top = end_bit_top - 1;
    let c_top = cm4 & below_top;
    if c_lo.count_ones() + c_mid.count_ones() + c_top.count_ones() != 6 {
        return None;
    }
    let q_top = qt4 & below_top;
    let seps_lo = c_lo;
    let seps_mid = c_mid;
    let seps_top = c_top | end_bit_top;
    let edges_lo = (seps_lo << 1) | 1 | (seps_lo >> 1) | (seps_mid << 63);
    let edges_mid = (seps_mid << 1) | (seps_lo >> 63) | (seps_mid >> 1) | ((seps_top as u64) << 63);
    let edges_top = (seps_top << 1) | ((seps_mid >> 63) as u32) | (seps_top >> 1);
    let p_lo = parity64(q_lo);
    let p_mid = parity64(q_mid) ^ 0u64.wrapping_sub(p_lo >> 63);
    let p_top = (parity64(q_top as u64) as u32) ^ 0u32.wrapping_sub((p_mid >> 63) as u32);
    if ((q_lo & !edges_lo) | (p_lo & seps_lo)) != 0
        || ((q_mid & !edges_mid) | (p_mid & seps_mid)) != 0
        || ((q_top & !edges_top) | (p_top & seps_top)) != 0
    {
        return None;
    }
    let mut clo = c_lo;
    let mut cmid = c_mid;
    let mut ctop = c_top;
    let c0 = next_comma3(&mut clo, &mut cmid, &mut ctop);
    let c1 = next_comma3(&mut clo, &mut cmid, &mut ctop);
    let c2 = next_comma3(&mut clo, &mut cmid, &mut ctop);
    let c5 = if c_top != 0 {
        128 + (31 - c_top.leading_zeros() as usize)
    } else if c_mid != 0 {
        64 + (63 - c_mid.leading_zeros() as usize)
    } else {
        63 - c_lo.leading_zeros() as usize
    };
    parse_row_tail(data, 128 + end_top, c0, c1, c2, c5, None, dates, sink)
}

#[inline(always)]
fn scan_with<'a, S: EventSink<'a>, M: Fn(&[u8; 96]) -> (u64, u32, u64, u32, u64, u32), P: Fn(u64) -> u64>(
    mut data: &'a [u8],
    dates: &DateCache,
    mut f: S,
    masks: M,
    parity64: P,
) {
    while data.len() >= 96 {
        let bytes: &'a [u8; 96] = unsafe { &*data.as_ptr().cast::<[u8; 96]>() };
        let (c_lo, cm2, l_lo, nl2, q_lo, qt2) = masks(bytes);
        if l_lo == 0 {
            if nl2 != 0 {
                let end_hi = nl2.trailing_zeros() as usize;
                let end = 64 + end_hi;
                if let Some(ev) = parse_row_hi(bytes, end_hi, c_lo, cm2, q_lo, qt2, dates, &parity64, &f) {
                    f.on_event(ev);
                } else {
                    fallback(unsafe { data.get_unchecked(..end + 1) }, dates, &mut f);
                }
                data = unsafe { data.get_unchecked(end + 1..) };
            } else if data.len() >= 128 {
                let (cm3, nl3, qt3) = scalar_masks32(unsafe { data.as_ptr().add(96) });
                let c_mid = (cm2 as u64) | ((cm3 as u64) << 32);
                let q_mid = (qt2 as u64) | ((qt3 as u64) << 32);
                if nl3 != 0 {
                    let end_mid = 32 + nl3.trailing_zeros() as usize;
                    let end = 64 + end_mid;
                    if let Some(ev) =
                        parse_row_mid128(data, end_mid, c_lo, c_mid, q_lo, q_mid, dates, &parity64, &f)
                    {
                        f.on_event(ev);
                    } else {
                        fallback(unsafe { data.get_unchecked(..end + 1) }, dates, &mut f);
                    }
                    data = unsafe { data.get_unchecked(end + 1..) };
                } else if data.len() >= 160 {
                    let (cm4, nl4, qt4) = scalar_masks32(unsafe { data.as_ptr().add(128) });
                    if nl4 != 0 {
                        let end_top = nl4.trailing_zeros() as usize;
                        let end = 128 + end_top;
                        if let Some(ev) = parse_row_top160(
                            data, end_top, c_lo, c_mid, cm4, q_lo, q_mid, qt4, dates, &parity64, &f,
                        ) {
                            f.on_event(ev);
                        } else {
                            fallback(unsafe { data.get_unchecked(..end + 1) }, dates, &mut f);
                        }
                        data = unsafe { data.get_unchecked(end + 1..) };
                    } else {
                        let end = data[160..]
                            .iter()
                            .position(|&b| b == b'\n')
                            .map_or(data.len(), |p| 160 + p + 1);
                        fallback(&data[..end], dates, &mut f);
                        data = &data[end..];
                    }
                } else {
                    let end = data[128..]
                        .iter()
                        .position(|&b| b == b'\n')
                        .map_or(data.len(), |p| 128 + p + 1);
                    fallback(&data[..end], dates, &mut f);
                    data = &data[end..];
                }
            } else {
                let end = data[96..]
                    .iter()
                    .position(|&b| b == b'\n')
                    .map_or(data.len(), |p| 96 + p + 1);
                fallback(&data[..end], dates, &mut f);
                data = &data[end..];
            }
        } else {
            let end = l_lo.trailing_zeros() as usize;
            if let Some(ev) = parse_row_lo(bytes, end, c_lo, q_lo, dates, &parity64, &f) {
                f.on_event(ev);
            } else {
                fallback(unsafe { data.get_unchecked(..end + 1) }, dates, &mut f);
            }
            data = unsafe { data.get_unchecked(end + 1..) };
        }
    }
    if !data.is_empty() {
        fallback(data, dates, &mut f);
    }
}

#[inline(always)]
pub fn scan<'a, S: EventSink<'a>>(data: &'a [u8], dates: &DateCache, f: S) {
    #[cfg(target_arch = "x86_64")]
    if csv::simd_backend() == "avx2"
        && is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("popcnt")
    {
        unsafe { scan_avx2(data, dates, f) };
        return;
    }
    scan_with(data, dates, f, scalar_masks96, prefix_xor64);
}

#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn handle_row_lo_avx2<'a, S: EventSink<'a>>(
    data: &'a [u8],
    bytes: &'a [u8; 96],
    c_lo: u64,
    q_lo: u64,
    dates: &DateCache,
    f: &mut S,
) -> usize {
    use core::arch::x86_64::*;
    let p = bytes.as_ptr();
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let b0 = _mm256_loadu_si256(p.cast());
    let b1 = _mm256_loadu_si256(p.add(32).cast());
    let l_lo = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_nl)) as u32 as u64)
        | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_nl)) as u32 as u64) << 32);
    let end = l_lo.trailing_zeros() as usize;
    let v_ones = _mm_set1_epi64x(-1);
    let parity64 = |q: u64| -> u64 {
        _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(q as i64), v_ones)) as u64
    };
    if let Some(ev) = parse_row_lo(bytes, end, c_lo, q_lo, dates, &parity64, f) {
        f.on_event(ev);
    } else {
        fallback(data.get_unchecked(..end + 1), dates, f);
    }
    end + 1
}

/// Cold AVX2 handler for rare event rows >= 128 bytes (~1.5% of v5 rows: both IDs 33B).
#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn handle_row_160_avx2<'a, S: EventSink<'a>>(
    data: &'a [u8],
    c_lo: u64,
    c_mid: u64,
    q_lo: u64,
    q_mid: u64,
    dates: &DateCache,
    f: &mut S,
) -> usize {
    use core::arch::x86_64::*;
    if data.len() >= 160 {
        let p = data.as_ptr();
        let v_cm = _mm256_set1_epi8(b',' as i8);
        let v_nl = _mm256_set1_epi8(b'\n' as i8);
        let v_qt = _mm256_set1_epi8(b'"' as i8);
        let v_ones = _mm_set1_epi64x(-1);
        let parity64 = |q: u64| -> u64 {
            _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(q as i64), v_ones)) as u64
        };
        let b4 = _mm256_loadu_si256(p.add(128).cast());
        let cm4 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b4, v_cm)) as u32;
        let nl4 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b4, v_nl)) as u32;
        let qt4 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b4, v_qt)) as u32;
        if nl4 != 0 {
            let end_top = nl4.trailing_zeros() as usize;
            let end = 128 + end_top;
            if let Some(ev) =
                parse_row_top160(data, end_top, c_lo, c_mid, cm4, q_lo, q_mid, qt4, dates, &parity64, f)
            {
                f.on_event(ev);
            } else {
                fallback(data.get_unchecked(..end + 1), dates, f);
            }
            return end + 1;
        }
        let end = data[160..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(data.len(), |pos| 160 + pos + 1);
        fallback(&data[..end], dates, f);
        return end;
    }
    let end = data[128..]
        .iter()
        .position(|&b| b == b'\n')
        .map_or(data.len(), |pos| 128 + pos + 1);
    fallback(&data[..end], dates, f);
    end
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn scan_avx2<'a, S: EventSink<'a>>(mut data: &'a [u8], dates: &DateCache, mut f: S) {
    use core::arch::x86_64::*;
    let v_cm = _mm256_set1_epi8(b',' as i8);
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let v_qt = _mm256_set1_epi8(b'"' as i8);
    let v_ones = _mm_set1_epi64x(-1);
    let parity64 = |q: u64| -> u64 {
        _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(q as i64), v_ones)) as u64
    };
    // Unified 4-vector (128B) hot loop: covers 98.4% of all v5 rows in ONE path!
    // Eliminates the 23.4%-mispredicted `nl2 != 0` branch (~4.7M mispredicts @ 20M)
    // AND halves loop code size (single `parse_row_tail` + `on_event` -> 100% DSB uOp cache hit).
    while data.len() >= 128 {
        let p = data.as_ptr();
        _mm_prefetch(p.add(512) as *const i8, _MM_HINT_T0);
        _mm_prefetch(p.add(576) as *const i8, _MM_HINT_T0);
        let b0 = _mm256_loadu_si256(p.cast());
        let b1 = _mm256_loadu_si256(p.add(32).cast());
        let b2 = _mm256_loadu_si256(p.add(64).cast());
        let b3 = _mm256_loadu_si256(p.add(96).cast());
        let c_lo = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_cm)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_cm)) as u32 as u64) << 32);
        let c_mid = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, v_cm)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b3, v_cm)) as u32 as u64) << 32);
        let l_lo_any = _mm256_movemask_epi8(_mm256_or_si256(
            _mm256_cmpeq_epi8(b0, v_nl),
            _mm256_cmpeq_epi8(b1, v_nl),
        ));
        let l_mid = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, v_nl)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b3, v_nl)) as u32 as u64) << 32);
        let q_lo = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_qt)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_qt)) as u32 as u64) << 32);
        let q_mid = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, v_qt)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b3, v_qt)) as u32 as u64) << 32);
        if l_lo_any == 0 {
            if l_mid != 0 {
                let end_mid = l_mid.trailing_zeros() as usize;
                let end = 64 + end_mid;
                if let Some(ev) =
                    parse_row_mid128(data, end_mid, c_lo, c_mid, q_lo, q_mid, dates, &parity64, &f)
                {
                    f.on_event(ev);
                } else {
                    fallback(data.get_unchecked(..end + 1), dates, &mut f);
                }
                data = data.get_unchecked(end + 1..);
            } else {
                let adv = handle_row_160_avx2(data, c_lo, c_mid, q_lo, q_mid, dates, &mut f);
                data = data.get_unchecked(adv..);
            }
        } else {
            let bytes: &'a [u8; 96] = &*p.cast::<[u8; 96]>();
            let adv = handle_row_lo_avx2(data, bytes, c_lo, q_lo, dates, &mut f);
            data = data.get_unchecked(adv..);
        }
    }
    if !data.is_empty() {
        fallback(data, dates, &mut f);
    }
}

// ---------------------------------------------------------------------------
// Single-pass 64B u64 AVX2+PCLMULQDQ click frontend (4 fields, zero Tok arrays)

#[cold]
#[inline(never)]
fn fallback_clicks<'a, S: ClickSink<'a>>(data: &'a [u8], f: &mut S) {
    csv::scan::<4, _>(data, |fields| f.on_click(fields));
}

#[inline(always)]
fn prefix_xor64(mut q: u64) -> u64 {
    q ^= q << 1;
    q ^= q << 2;
    q ^= q << 4;
    q ^= q << 8;
    q ^= q << 16;
    q ^= q << 32;
    q
}

#[inline(always)]
fn scalar_masks64(data: &[u8; 64]) -> (u64, u64, u64) {
    const LO7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    #[inline(always)]
    fn eq_mask(word: u64, value: u64) -> u64 {
        let y = word ^ value;
        let high = !(((y & LO7).wrapping_add(LO7)) | y | LO7);
        (high >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56
    }
    let (mut c, mut l, mut q) = (0u64, 0u64, 0u64);
    let p = data.as_ptr();
    for i in 0..8 {
        let w = unsafe { (p.add(i * 8) as *const u64).read_unaligned() };
        c |= eq_mask(w, 0x2c2c_2c2c_2c2c_2c2c) << (i * 8);
        l |= eq_mask(w, 0x0a0a_0a0a_0a0a_0a0a) << (i * 8);
        q |= eq_mask(w, 0x2222_2222_2222_2222) << (i * 8);
    }
    (c, l, q)
}

#[inline(always)]
fn parse_click_row<'a, P: Fn(u64) -> u64>(
    data: &'a [u8; 64],
    end: usize,
    commas: u64,
    quotes: u64,
    parity: &P,
) -> Option<[&'a [u8]; 4]> {
    let below = (1u64 << end) - 1;
    let c = commas & below;
    if c.count_ones() != 3 {
        return None;
    }
    let q = quotes & below;
    let seps = c | (1u64 << end);
    let edges = (seps << 1) | 1 | (seps >> 1);
    if ((q & !edges) | (parity(q) & seps)) != 0 {
        return None;
    }
    let c0 = c.trailing_zeros() as usize;
    let c_1 = c & c.wrapping_sub(1);
    let c1 = c_1.trailing_zeros() as usize;
    let c2 = (c_1 & c_1.wrapping_sub(1)).trailing_zeros() as usize;
    let q0 = (q & 1) as usize;
    let q1 = ((q >> (c0 + 1)) & 1) as usize;
    let q2 = ((q >> (c1 + 1)) & 1) as usize;
    let q3 = ((q >> (c2 + 1)) & 1) as usize;
    unsafe {
        Some([
            data.get_unchecked(q0..c0 - q0),
            data.get_unchecked(c0 + 1 + q1..c1 - q1),
            data.get_unchecked(c1 + 1 + q2..c2 - q2),
            data.get_unchecked(c2 + 1 + q3..end - q3),
        ])
    }
}

#[inline(always)]
fn parse_click_row_128<'a, P: Fn(u64) -> u64>(
    data: &'a [u8],
    end_hi: usize,
    c_lo: u64,
    c_hi_raw: u64,
    q_lo: u64,
    q_hi_raw: u64,
    parity: &P,
) -> Option<[&'a [u8]; 4]> {
    let end_bit_hi = 1u64 << end_hi;
    let below_hi = end_bit_hi - 1;
    let c_hi = c_hi_raw & below_hi;
    if c_lo.count_ones() + c_hi.count_ones() != 3 {
        return None;
    }
    let q_hi = q_hi_raw & below_hi;
    let seps_lo = c_lo;
    let seps_hi = c_hi | end_bit_hi;
    let edges_lo = (seps_lo << 1) | 1 | (seps_lo >> 1) | (seps_hi << 63);
    let edges_hi = (seps_hi << 1) | (seps_lo >> 63) | (seps_hi >> 1);
    let p_lo = parity(q_lo);
    let p_hi = parity(q_hi) ^ 0u64.wrapping_sub(p_lo >> 63);
    if ((q_lo & !edges_lo) | (p_lo & seps_lo)) != 0 || ((q_hi & !edges_hi) | (p_hi & seps_hi)) != 0 {
        return None;
    }
    let mut clo = c_lo;
    let mut chi = c_hi;
    let mut ctop = 0u32;
    let c0 = next_comma3(&mut clo, &mut chi, &mut ctop);
    let c1 = next_comma3(&mut clo, &mut chi, &mut ctop);
    let c2 = next_comma3(&mut clo, &mut chi, &mut ctop);
    let end = 64 + end_hi;
    let p = data.as_ptr();
    unsafe {
        let q0 = (c0 > 0 && *p == b'"') as usize;
        let q1 = (c0 + 1 < c1 && *p.add(c0 + 1) == b'"') as usize;
        let q2 = (c1 + 1 < c2 && *p.add(c1 + 1) == b'"') as usize;
        let q3 = (c2 + 1 < end && *p.add(c2 + 1) == b'"') as usize;
        Some([
            data.get_unchecked(q0..c0 - q0),
            data.get_unchecked(c0 + 1 + q1..c1 - q1),
            data.get_unchecked(c1 + 1 + q2..c2 - q2),
            data.get_unchecked(c2 + 1 + q3..end - q3),
        ])
    }
}

#[inline(always)]
fn scan_clicks_with<'a, S: ClickSink<'a>, M: Fn(&[u8; 64]) -> (u64, u64, u64), P: Fn(u64) -> u64>(
    mut data: &'a [u8],
    mut f: S,
    masks: M,
    parity: P,
) {
    while data.len() >= 64 {
        let bytes: &'a [u8; 64] = unsafe { &*data.as_ptr().cast::<[u8; 64]>() };
        let (commas, lines, quotes) = masks(bytes);
        let end = lines.trailing_zeros() as usize;
        if end < 64 {
            if let Some(fields) = parse_click_row(bytes, end, commas, quotes, &parity) {
                f.on_click(&fields);
            } else {
                fallback_clicks(unsafe { data.get_unchecked(..end + 1) }, &mut f);
            }
            data = unsafe { data.get_unchecked(end + 1..) };
        } else if data.len() >= 128 {
            let bytes_hi: &'a [u8; 64] = unsafe { &*data.as_ptr().add(64).cast::<[u8; 64]>() };
            let (c_hi, l_hi, q_hi) = masks(bytes_hi);
            if l_hi != 0 {
                let end_hi = l_hi.trailing_zeros() as usize;
                let end = 64 + end_hi;
                if let Some(fields) = parse_click_row_128(data, end_hi, commas, c_hi, quotes, q_hi, &parity) {
                    f.on_click(&fields);
                } else {
                    fallback_clicks(unsafe { data.get_unchecked(..end + 1) }, &mut f);
                }
                data = unsafe { data.get_unchecked(end + 1..) };
            } else {
                let end = data[128..]
                    .iter()
                    .position(|&b| b == b'\n')
                    .map_or(data.len(), |pos| 128 + pos + 1);
                fallback_clicks(&data[..end], &mut f);
                data = &data[end..];
            }
        } else {
            let end = data[64..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(data.len(), |pos| 64 + pos + 1);
            fallback_clicks(&data[..end], &mut f);
            data = &data[end..];
        }
    }
    if !data.is_empty() {
        fallback_clicks(data, &mut f);
    }
}

#[inline(always)]
pub fn scan_clicks<'a, S: ClickSink<'a>>(data: &'a [u8], f: S) {
    #[cfg(target_arch = "x86_64")]
    if csv::simd_backend() == "avx2"
        && is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("popcnt")
    {
        unsafe { scan_clicks_avx2(data, f) };
        return;
    }
    scan_clicks_with(data, f, scalar_masks64, prefix_xor64);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn scan_clicks_avx2<'a, S: ClickSink<'a>>(mut data: &'a [u8], mut f: S) {
    use core::arch::x86_64::*;
    let v_cm = _mm256_set1_epi8(b',' as i8);
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let v_qt = _mm256_set1_epi8(b'"' as i8);
    let v_ones = _mm_set1_epi64x(-1);
    let parity = |q: u64| -> u64 {
        _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(q as i64), v_ones)) as u64
    };
    // 64B fast path (77% of v5 clicks: 2 YMM loads, u64 masks) + inline 64..128B extension (23%).
    while data.len() >= 64 {
        let p = data.as_ptr();
        _mm_prefetch(p.add(384) as *const i8, _MM_HINT_T0);
        let bytes: &'a [u8; 64] = &*p.cast::<[u8; 64]>();
        let b0 = _mm256_loadu_si256(p.cast());
        let b1 = _mm256_loadu_si256(p.add(32).cast());
        let commas = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_cm)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_cm)) as u32 as u64) << 32);
        let lines = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_nl)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_nl)) as u32 as u64) << 32);
        let quotes = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_qt)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_qt)) as u32 as u64) << 32);
        let end = lines.trailing_zeros() as usize;
        if end < 64 {
            if let Some(fields) = parse_click_row(bytes, end, commas, quotes, &parity) {
                f.on_click(&fields);
            } else {
                fallback_clicks(data.get_unchecked(..end + 1), &mut f);
            }
            data = data.get_unchecked(end + 1..);
        } else if data.len() >= 128 {
            let b2 = _mm256_loadu_si256(p.add(64).cast());
            let b3 = _mm256_loadu_si256(p.add(96).cast());
            let c_hi = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, v_cm)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b3, v_cm)) as u32 as u64) << 32);
            let l_hi = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, v_nl)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b3, v_nl)) as u32 as u64) << 32);
            let q_hi = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, v_qt)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b3, v_qt)) as u32 as u64) << 32);
            if l_hi != 0 {
                let end_hi = l_hi.trailing_zeros() as usize;
                let end = 64 + end_hi;
                if let Some(fields) = parse_click_row_128(data, end_hi, commas, c_hi, quotes, q_hi, &parity) {
                    f.on_click(&fields);
                } else {
                    fallback_clicks(data.get_unchecked(..end + 1), &mut f);
                }
                data = data.get_unchecked(end + 1..);
            } else {
                let end = data[128..]
                    .iter()
                    .position(|&b| b == b'\n')
                    .map_or(data.len(), |pos| 128 + pos + 1);
                fallback_clicks(&data[..end], &mut f);
                data = &data[end..];
            }
        } else {
            let end = data[64..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(data.len(), |pos| 64 + pos + 1);
            fallback_clicks(&data[..end], &mut f);
            data = &data[end..];
        }
    }
    if !data.is_empty() {
        fallback_clicks(data, &mut f);
    }
}

// ---------------------------------------------------------------------------
// Multi-row-per-64B-load AVX2+PCLMULQDQ link frontend (2 fields, ~3 rows/load)

#[cold]
#[inline(never)]
fn fallback_links<'a, S: LinkSink<'a>>(data: &'a [u8], f: &mut S) {
    csv::scan::<2, _>(data, |fields| f.on_link(fields[0], fields[1]));
}

#[inline(always)]
fn scan_links_with<'a, S: LinkSink<'a>, M: Fn(&[u8; 64]) -> (u64, u64, u64), P: Fn(u64) -> u64>(
    mut data: &'a [u8],
    mut f: S,
    masks: M,
    parity: P,
) {
    while data.len() >= 64 {
        let bytes: &'a [u8; 64] = unsafe { &*data.as_ptr().cast::<[u8; 64]>() };
        let (mut commas, mut lines, mut quotes) = masks(bytes);
        if lines == 0 {
            let end = data[64..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(data.len(), |pos| 64 + pos + 1);
            fallback_links(&data[..end], &mut f);
            data = &data[end..];
            continue;
        }
        let mut off = 0usize;
        while lines != 0 {
            let end = lines.trailing_zeros() as usize;
            let end_bit = 1u64 << end;
            let below = end_bit - 1;
            let c = commas & below;
            let q = quotes & below;
            let seps = c | end_bit;
            let edges = (seps << 1) | 1 | (seps >> 1);
            if c.count_ones() == 1 && ((q & !edges) | (parity(q) & seps)) == 0 {
                let c0 = c.trailing_zeros() as usize;
                let q0 = (q & 1) as usize;
                let q1 = ((q >> (c0 + 1)) & 1) as usize;
                unsafe {
                    f.on_link(
                        data.get_unchecked(off + q0..off + c0 - q0),
                        data.get_unchecked(off + c0 + 1 + q1..off + end - q1),
                    );
                }
            } else {
                fallback_links(unsafe { data.get_unchecked(off..off + end + 1) }, &mut f);
            }
            let adv = end + 1;
            off += adv;
            if adv >= 64 {
                break;
            }
            commas >>= adv;
            lines >>= adv;
            quotes >>= adv;
        }
        data = unsafe { data.get_unchecked(off..) };
    }
    if !data.is_empty() {
        fallback_links(data, &mut f);
    }
}

#[inline(always)]
pub fn scan_links<'a, S: LinkSink<'a>>(data: &'a [u8], f: S) {
    #[cfg(target_arch = "x86_64")]
    if csv::simd_backend() == "avx2"
        && is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("popcnt")
    {
        unsafe { scan_links_avx2(data, f) };
        return;
    }
    scan_links_with(data, f, scalar_masks64, prefix_xor64);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn scan_links_avx2<'a, S: LinkSink<'a>>(mut data: &'a [u8], mut f: S) {
    use core::arch::x86_64::*;
    let v_cm = _mm256_set1_epi8(b',' as i8);
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let v_qt = _mm256_set1_epi8(b'"' as i8);
    let v_ones = _mm_set1_epi64x(-1);
    let parity = |q: u64| -> u64 {
        _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(q as i64), v_ones)) as u64
    };
    while data.len() >= 64 {
        let p = data.as_ptr();
        _mm_prefetch(p.add(384) as *const i8, _MM_HINT_T0);
        let b0 = _mm256_loadu_si256(p.cast());
        let b1 = _mm256_loadu_si256(p.add(32).cast());
        let mut commas = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_cm)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_cm)) as u32 as u64) << 32);
        let mut lines = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_nl)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_nl)) as u32 as u64) << 32);
        let mut quotes = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_qt)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_qt)) as u32 as u64) << 32);
        if lines == 0 {
            let end = data[64..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(data.len(), |pos| 64 + pos + 1);
            fallback_links(&data[..end], &mut f);
            data = &data[end..];
            continue;
        }
        let mut off = 0usize;
        while lines != 0 {
            let end = lines.trailing_zeros() as usize;
            let end_bit = 1u64 << end;
            let below = end_bit - 1;
            let c = commas & below;
            let q = quotes & below;
            let seps = c | end_bit;
            let edges = (seps << 1) | 1 | (seps >> 1);
            if c.count_ones() == 1 && ((q & !edges) | (parity(q) & seps)) == 0 {
                let c0 = c.trailing_zeros() as usize;
                let q0 = (q & 1) as usize;
                let q1 = ((q >> (c0 + 1)) & 1) as usize;
                f.on_link(
                    data.get_unchecked(off + q0..off + c0 - q0),
                    data.get_unchecked(off + c0 + 1 + q1..off + end - q1),
                );
            } else {
                fallback_links(data.get_unchecked(off..off + end + 1), &mut f);
            }
            let adv = end + 1;
            off += adv;
            if adv >= 64 {
                break;
            }
            commas >>= adv;
            lines >>= adv;
            quotes >>= adv;
        }
        data = data.get_unchecked(off..);
    }
    if !data.is_empty() {
        fallback_links(data, &mut f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dates() -> DateCache {
        DateCache::new(1_791_417_600_000, 1_792_627_200_000)
    }

    fn expected<'a>(input: &'a [u8], dates: &DateCache) -> Vec<Event<'a>> {
        let mut out = Vec::new();
        csv::scan::<7, _>(input, |f| {
            let kind = match f[4] {
                b"view" | b"cart" => 1,
                b"purchase" => 2,
                _ => 0,
            };
            let (ing_day_ms, ing_bcd) = dates.parse_f1_ingest_bcd(f[3]);
            let ek = EKey::new(f[0]);
            let eh = ek.hash(SEED_ID);
            let (uid_k, uh) = Key8::pack_inline_or_sentinel(f[1], SEED_A);
            out.push(Event {
                ek,
                eh,
                uid_k,
                uh,
                uid_bytes: f[1],
                product: if kind == 0 { &[] } else { f[5] },
                timestamp: dates.parse_ts(f[2]),
                ing_day_ms,
                ing_bcd,
                kind,
                price: if kind == 2 { parse_price(f[6]) } else { 0 },
            });
        });
        out
    }

    fn check(input: &[u8]) {
        let d = dates();
        let want = expected(input, &d);
        let mut got = Vec::new();
        scan_with(input, &d, |e| got.push(e), scalar_masks96, prefix_xor64);
        assert_eq!(got, want, "scalar input={input:?}");
        got.clear();
        scan(input, &d, |e| got.push(e));
        assert_eq!(got, want, "selected input={input:?}");
    }

    #[test]
    fn bit_masks_every_lane_and_byte() {
        for lane in 0..96 {
            for byte in 0..=255u8 {
                let mut block = [0u8; 96];
                block[lane] = byte;
                let (c_lo, c_hi, l_lo, l_hi, q_lo, q_hi) = scalar_masks96(&block);
                let got = (
                    (c_lo as u128) | ((c_hi as u128) << 64),
                    (l_lo as u128) | ((l_hi as u128) << 64),
                    (q_lo as u128) | ((q_hi as u128) << 64),
                );
                let want = (
                    if byte == b',' { 1u128 << lane } else { 0 },
                    if byte == b'\n' { 1u128 << lane } else { 0 },
                    if byte == b'"' { 1u128 << lane } else { 0 },
                );
                assert_eq!(got, want);
            }
        }
        for lane in 0..64 {
            assert_eq!(prefix_xor64(1u64 << lane), u64::MAX << lane);
        }
    }

    #[test]
    fn timestamps_quotes_and_truncations() {
        for ts in [
            "2026-10-08T12:34:56Z",
            "2026-10-08T12:34:56-04:00",
            "1791462896000",
            "0001791462896000",
            "86400000",
            "2026-02-30T12:34:56Z",
            "2026-10-08T24:00:00Z",
            "2026-10-08T12:34:56+24:00",
            "2026-10-08T12:34:56Z\r",
            "999999999999999999999999",
            "",
        ] {
            for qmask in 0..128 {
                let vals = ["e1", "u1", ts, "2026-10-08T13:00:00Z", "purchase", "p1", "00250"];
                let row = vals
                    .iter()
                    .enumerate()
                    .map(|(i, s)| if qmask & (1 << i) != 0 { format!("\"{s}\"") } else { (*s).to_owned() })
                    .collect::<Vec<_>>()
                    .join(",")
                    + "\n";
                let input = row.repeat(3).into_bytes();
                check(&input);
                for end in 0..row.len() {
                    check(&input[..row.len() + end]);
                }
            }
        }
    }

    #[test]
    fn mutations_offsets_long_ids_and_seams() {
        let valid = b"e1,u1,2026-10-08T12:34:56Z,2026-10-08T13:00:00Z,purchase,p1,250\n";
        for offset in 0..64 {
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
        for size in [0, 31, 32, 33, 63, 64, 65, 95, 96, 97, 128, 16_385] {
            let mut input = vec![b'x'; size];
            input.extend_from_slice(&valid[2..]);
            input.extend_from_slice(valid);
            check(&input);
        }
        // v5 33-byte hex IDs: 1 long ID (104-126B) and 2 long IDs (128-159B), all quote/ts combos.
        let long_e = "e0123456789abcdef0123456789abcdef";
        let long_u = "u02423c3d84ab053af78764ce49f23b5c";
        for (eid, uid) in [(long_e, "u12345678"), ("e12345678", long_u), (long_e, long_u)] {
            for ts in ["2026-10-08T12:34:56Z", "2026-10-08T12:34:56-04:00", "1791462896000"] {
                for qmask in [0u8, 1, 2, 3, 7, 0x7f] {
                    let vals = [eid, uid, ts, "2026-10-08T13:00:00Z", "purchase", "p22306", "130"];
                    let row = vals
                        .iter()
                        .enumerate()
                        .map(|(i, s)| if qmask & (1 << i) != 0 { format!("\"{s}\"") } else { (*s).to_owned() })
                        .collect::<Vec<_>>()
                        .join(",")
                        + "\n";
                    let input = row.repeat(3).into_bytes();
                    check(&input);
                    for end in [0, 63, 64, 95, 96, 127, 128, row.len() - 1] {
                        if end < row.len() {
                            check(&input[..row.len() + end]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn clicks_quotes_mutations_and_long_lines() {
        let check_clicks = |input: &[u8]| {
            let mut want: Vec<[Vec<u8>; 4]> = Vec::new();
            csv::scan_byte_oracle::<4, _>(input, |f| {
                want.push([f[0].to_vec(), f[1].to_vec(), f[2].to_vec(), f[3].to_vec()]);
            });
            let mut got: Vec<[Vec<u8>; 4]> = Vec::new();
            scan_clicks_with(
                input,
                |f: &[&[u8]; 4]| got.push([f[0].to_vec(), f[1].to_vec(), f[2].to_vec(), f[3].to_vec()]),
                scalar_masks64,
                prefix_xor64,
            );
            assert_eq!(got, want, "scalar clicks input={input:?}");
            got.clear();
            scan_clicks(input, |f: &[&[u8]; 4]| got.push([f[0].to_vec(), f[1].to_vec(), f[2].to_vec(), f[3].to_vec()]));
            assert_eq!(got, want, "selected clicks input={input:?}");
        };
        for ts in ["2026-10-08T12:34:56Z", "2026-10-08T12:34:56-04:00", "1791462896000", ""] {
            for qmask in 0..16 {
                let vals = ["c12345678", "u12345678", "c_1958", ts];
                let row = vals
                    .iter()
                    .enumerate()
                    .map(|(i, s)| if qmask & (1 << i) != 0 { format!("\"{s}\"") } else { (*s).to_owned() })
                    .collect::<Vec<_>>()
                    .join(",")
                    + "\n";
                let input = row.repeat(3).into_bytes();
                check_clicks(&input);
                for end in 0..row.len() {
                    check_clicks(&input[..row.len() + end]);
                }
            }
        }
        let valid = b"c12345678,u12345678,c_1234,2026-10-08T12:34:56Z\n";
        for offset in 0..32 {
            for pos in 0..valid.len() {
                for byte in [0, b'"', b',', b'\n', b'\r', b' ', 255] {
                    let mut input = vec![b'!'; offset];
                    input.extend_from_slice(valid);
                    input[offset + pos] = byte;
                    input.extend_from_slice(valid);
                    check_clicks(&input[offset..]);
                }
            }
        }
        for size in [0, 31, 32, 63, 64, 65, 128] {
            let mut input = vec![b'x'; size];
            input.extend_from_slice(b",u1,c_1,2026-10-08T12:34:56Z\n");
            input.extend_from_slice(valid);
            check_clicks(&input);
        }
        // v5 33-byte hex click/user IDs (72..112B click rows)
        let long_k = "k0123456789abcdef0123456789abcdef";
        let long_u = "u02423c3d84ab053af78764ce49f23b5c";
        for (cid, uid) in [(long_k, "u12345678"), ("k12345678", long_u), (long_k, long_u)] {
            for ts in ["2026-10-08T12:34:56Z", "2026-10-08T12:34:56-04:00", "1791462896000"] {
                for qmask in 0..16u8 {
                    let vals = [cid, uid, "c_1958", ts];
                    let row = vals
                        .iter()
                        .enumerate()
                        .map(|(i, s)| if qmask & (1 << i) != 0 { format!("\"{s}\"") } else { (*s).to_owned() })
                        .collect::<Vec<_>>()
                        .join(",")
                        + "\n";
                    let input = row.repeat(3).into_bytes();
                    check_clicks(&input);
                    for end in [0, 63, 64, 95, 96, row.len() - 1] {
                        if end < row.len() {
                            check_clicks(&input[..row.len() + end]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn links_quotes_mutations_and_long_lines() {
        let check_links = |input: &[u8]| {
            let mut want: Vec<[Vec<u8>; 2]> = Vec::new();
            csv::scan_byte_oracle::<2, _>(input, |f| {
                want.push([f[0].to_vec(), f[1].to_vec()]);
            });
            let mut got: Vec<[Vec<u8>; 2]> = Vec::new();
            scan_links_with(
                input,
                |a: &[u8], b: &[u8]| got.push([a.to_vec(), b.to_vec()]),
                scalar_masks64,
                prefix_xor64,
            );
            assert_eq!(got, want, "scalar links input={input:?}");
            got.clear();
            scan_links(input, |a: &[u8], b: &[u8]| got.push([a.to_vec(), b.to_vec()]));
            assert_eq!(got, want, "selected links input={input:?}");
        };
        for qmask in 0..4 {
            let vals = ["u12345678", "u87654321"];
            let row = vals
                .iter()
                .enumerate()
                .map(|(i, s)| if qmask & (1 << i) != 0 { format!("\"{s}\"") } else { (*s).to_owned() })
                .collect::<Vec<_>>()
                .join(",")
                + "\n";
            let input = row.repeat(6).into_bytes();
            check_links(&input);
            for end in 0..row.len() {
                check_links(&input[..row.len() * 3 + end]);
            }
        }
        let valid = b"u12345678,u87654321\n";
        for offset in 0..32 {
            for pos in 0..valid.len() {
                for byte in [0, b'"', b',', b'\n', b'\r', b' ', 255] {
                    let mut input = vec![b'!'; offset];
                    input.extend_from_slice(valid);
                    input[offset + pos] = byte;
                    input.extend_from_slice(valid);
                    input.extend_from_slice(valid);
                    input.extend_from_slice(valid);
                    check_links(&input[offset..]);
                }
            }
        }
        for size in [0, 31, 32, 62, 63, 64, 65, 128] {
            let mut input = vec![b'x'; size];
            input.extend_from_slice(b",u1\n");
            input.extend_from_slice(valid);
            input.extend_from_slice(valid);
            check_links(&input);
        }
    }
}
