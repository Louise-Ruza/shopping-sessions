//! Grouped controls for the existing exact event-ID winner table.
//!
//! Slots retain their original u32/u64 representation. A control byte is either
//! EMPTY or seven high hash bits. Sixteen adjacent controls share a home group;
//! groups wrap linearly. No deletion occurs, so finding an empty control after
//! checking every matching fingerprint proves the key is absent. Replacements
//! only update the winner slot; they never remove its occupied control.


use super::{arena::Segs, ingest_less, prefetch, RawEv, Slot};


const EMPTY: u8 = 0x80;

/// `ptr` names 16 initialized controls. x86_64 always supports SSE2. The scalar
/// version keeps native ARM correctness checks independent of that instruction.

#[inline(always)]
unsafe fn masks(ptr: *const u8, tag: u8) -> (u16, u16) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let bytes = _mm_loadu_si128(ptr.cast());
        let matching = _mm_movemask_epi8(_mm_cmpeq_epi8(bytes, _mm_set1_epi8(tag as i8)));
        // Occupied tags have bit7 clear; empty controls have bit7 set.
        (matching as u16, _mm_movemask_epi8(bytes) as u16)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let (mut matching, mut empty) = (0u16, 0u16);
        for i in 0..16 {
            let c = *ptr.add(i);
            matching |= ((c == tag) as u16) << i;
            empty |= ((c == EMPTY) as u16) << i;
        }
        (matching, empty)
    }
}


pub(super) fn mark_losers<S: Slot + Default + Clone>(
    segs: &Segs<RawEv>, table: &mut Vec<S>, controls: &mut Vec<u8>,
    cap: usize, loser: &mut [u64], base: i64, oob: &[i64],
) {
    debug_assert!(cap >= 16 && cap.is_power_of_two());
    debug_assert!(segs.len() < cap);
    // Only controls are reset for each shard. Stale slots are never read unless
    // this call has first written their occupied control and complete slot.
    if table.len() < cap { table.resize(cap, S::default()); }
    controls.clear();
    controls.resize(cap, EMPTY);
    let group_mask = (cap - 1) & !15;
    let tptr = table.as_mut_ptr();
    let cptr = controls.as_mut_ptr();
    let rec = |idx: u32| -> &RawEv {
        unsafe { segs.seg((idx >> 9) as usize).get_unchecked((idx & 511) as usize) }
    };

    const PF: usize = 16;
    for si in 0..segs.segs.len() {
        let seg = segs.seg(si);
        for ri in 0..seg.len() {
            if ri + PF < seg.len() {
                let group = unsafe { seg.get_unchecked(ri + PF).ehash } as usize & group_mask;
                prefetch(unsafe { cptr.add(group) });
                prefetch(unsafe { tptr.add(group) }.cast());
            }
            let r = unsafe { seg.get_unchecked(ri) };
            let h = r.ehash;
            let tag = (h >> 25) as u8;
            let me = ((si << 9) | ri) as u32;
            let mut group = h as usize & group_mask;

            'probe: loop {

                // Aligned group indices and cap>=16 prove the whole load lies
                // within the initialized control allocation, including wrap.
                let (mut matching, empty) = unsafe { masks(cptr.add(group), tag) };
                while matching != 0 {
                    let lane = matching.trailing_zeros() as usize;
                    matching &= matching - 1;
                    let i = group + lane;
                    let slot = unsafe { *tptr.add(i) };

                    if !slot.tag_eq(h) { continue; }

                    let oi = slot.idx();
                    let old = rec(oi);
                    if old.eid != r.eid { continue; }

                    if ingest_less(r.ingest, old.ingest, base, oob) {
                        loser[(oi >> 6) as usize] |= 1 << (oi & 63);
                        unsafe { *tptr.add(i) = S::make(h, me); }
                    } else {
                        loser[(me >> 6) as usize] |= 1 << (me & 63);
                    }
                    break 'probe;
                }
                if empty != 0 {
                    let i = group + empty.trailing_zeros() as usize;
                    unsafe {
                        *tptr.add(i) = S::make(h, me);
                        *cptr.add(i) = tag;
                    }
                    break;
                }
                group = (group + 16) & group_mask;
            }
        }
    }

}









# [cfg (test)]
mod tests {
    use super::*;
    use crate::{arena::{Scatter, ShardSet}, key::{Key8, UserKey}, INGEST_NONE, INGEST_OOB_BASE, TS_INVALID};

    fn shard(rows: &[RawEv]) -> Segs<RawEv> {
        let set = ShardSet::new(1);
        let mut scatter = Scatter::new(&set);
        for &r in rows { scatter.push(0, r); }
        scatter.finish();
        set.take(0)
    }

    fn row(id: &[u8], ingest: u32, h: u32) -> RawEv {
        RawEv { eid: Key8::event_id(id), uid: UserKey::new(b"u"), ts: 10, price: 0, ingest, ehash: h }
    }

    fn check<S: Slot + Default + Clone>(rows: &[RawEv], table: &mut Vec<S>, controls: &mut Vec<u8>) {
        let segs = shard(rows);
        let cap = (rows.len() + rows.len() / 2).next_power_of_two().max(16);
        let mut actual = vec![0; segs.segs.len() * 8];
        let mut expected = actual.clone();
        let base = 1000;
        let oob = [-400, 20_000, 20_001];
        // Independent exact-key minimum over real segmented record indices.
        let mut winners: std::collections::HashMap<Key8, (u32, (u8, i64))> = std::collections::HashMap::new();
        for (si, seg) in segs.iter().enumerate() {
            for (ri, r) in seg.iter().enumerate() {
                let me = ((si << 9) | ri) as u32;
                let rank = if r.ingest == INGEST_NONE { (1, i64::MAX) }
                    else if r.ingest >= INGEST_OOB_BASE { (0, oob[(r.ingest - INGEST_OOB_BASE) as usize]) }
                    else { (0, base + r.ingest as i64) };
                let entry = winners.entry(r.eid).or_insert((me, rank));
                if entry.0 == me { continue; }
                let dead = if rank < entry.1 { let old = entry.0; *entry = (me, rank); old } else { me };
                expected[(dead >> 6) as usize] |= 1 << (dead & 63);
            }
        }
        mark_losers(&segs, table, controls, cap, &mut actual, base, &oob);
        assert_eq!(actual, expected);
        let mut baseline = vec![0; actual.len()];
        crate::mark_losers(&segs, &mut Vec::<S>::new(), cap, &mut baseline, base, &oob);
        assert_eq!(actual, baseline);
        segs.free();
    }

    #[test]
    fn controls_match_all_tags_and_group_edges() {
        let mut data = [EMPTY; 48];
        for offset in 0..=32 {
            for tag in 0..128 {
                for i in 0..16 { data[offset+i] = if i % 5 == 0 { EMPTY } else { (tag+i as u8) % 128 }; }
                let mut want_match = 0;
                let mut want_empty = 0;
                for i in 0..16 {
                    want_match |= ((data[offset+i] == tag) as u16) << i;
                    want_empty |= ((data[offset+i] == EMPTY) as u16) << i;
                }
                assert_eq!(unsafe { masks(data.as_ptr().add(offset), tag) }, (want_match, want_empty));
            }
        }
    }

    #[test]
    fn colliding_groups_wrap_and_reuse_growing_storage() {
        let mut t32 = Vec::<u32>::new(); let mut c32 = Vec::new();
        let mut t64 = Vec::<u64>::new(); let mut c64 = Vec::new();
        // Same full hash forces both fingerprint and exact-key collisions and
        // spills beyond multiple groups. 0xffff_fff0 starts at the final group.
        for n in [0, 1, 15, 17, 80, 1200, 2, 700] {
            let mut rows: Vec<_> = (0..n).map(|i| row(format!("event-{i}\0long").as_bytes(), 100, 0xffff_fff0)).collect();
            for i in (0..n).step_by(3) {
                let mut copy = rows[i]; copy.ingest = 50; copy.uid = UserKey::new(b"changed-person");
                if i % 2 == 0 { copy.ts = TS_INVALID; }
                rows.push(copy);
            }
            check(&rows, &mut t32, &mut c32);
            check(&rows, &mut t64, &mut c64);
        }
    }

    #[test]
    fn ingestion_ties_oob_invalid_and_missing_winners() {
        let ids: &[&[u8]] = &[b"a", b"a\0", b"a\r", b"a-long-identifier", "é".as_bytes(),
            b"e", b"ea", b"ea\0", b"ea\0\0", b"e1234567", b"e12345678", b"e123456789"];
        let mut rows = Vec::new();
        for &id in ids {
            for (i, ingest) in [INGEST_NONE, INGEST_OOB_BASE+2, 5, INGEST_OOB_BASE+1,
                INGEST_OOB_BASE, INGEST_OOB_BASE, 0].into_iter().enumerate() {
                let mut r = row(id, ingest, 0); r.ts = if i == 4 { TS_INVALID } else { 10 };
                r.uid = UserKey::new(if i % 2 == 0 { b"a" } else { b"z" }); rows.push(r);
            }
        }
        check(&rows, &mut Vec::<u32>::new(), &mut Vec::new());
        check(&rows, &mut Vec::<u64>::new(), &mut Vec::new());
        // Large-shard slot indices retain all 32 bits; the ordinary u32 path
        // separately caps segmented indices below its 24-bit payload limit.
        for index in [0, (1 << 24) - 2, 1 << 24, (1 << 31) - 1] {
            let slot = <u64 as Slot>::make(0xfedc_ba98, index);
            assert_eq!(slot.idx(), index); assert!(slot.tag_eq(0xfedc_ba98));
        }
    }
}
