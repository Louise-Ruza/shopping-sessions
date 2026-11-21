//! Sessionize dense histories from per-bin extrema. The diameter of a half-open
//! bin is less than the gap, so all its events belong to one session. Between
//! occupied bins the only relevant gap is next.minimum minus previous.maximum.
use super::{Cl24, Ev8, Sess, UserKey, Worker, DAY_MS, GAP_MS};

const WIDTH: u32 = GAP_MS as u32;
const BINS: usize = (u32::MAX / WIDTH) as usize + 1;

#[derive(Clone, Copy)]
struct Bin { min: u32, max: u32, count: u32, session: u16 }
const EMPTY: Bin = Bin { min: u32::MAX, max: 0, count: 0, session: 0 };

#[derive(Clone, Copy, Default)]
struct Day { sessions: u32, events: u32, revenue: u64 }

#[derive(Default)]
pub(super) struct Scratch {
    bins: Vec<Bin>,
    purchases: Vec<(u16, u32)>,
    days: Vec<Day>,
    categories: Vec<u64>,
}

impl Scratch {
    pub fn try_person(&mut self, w: &mut Worker<'_>, person: UserKey, events: &[Ev8],
        clicks: &mut [Cl24], ctmp: &mut Vec<Cl24>) -> bool {
        // Sparse histories already have an efficient comparison sort. For wide
        // catalogs preserve the existing memory bound and exact side-table path.
        if events.len() < 512 || w.ctx.wide { return false; }
        self.bins.resize(BINS, EMPTY);
        let (mut first, mut last) = (BINS, 0usize);
        self.purchases.clear();
        for &event in events {
            let ts = event.ts();
            let index = (ts / WIDTH) as usize;
            first = first.min(index);
            last = last.max(index);
            let bin = &mut self.bins[index];
            bin.count += 1;
            bin.min = bin.min.min(ts);
            bin.max = bin.max.max(ts);
            if event.pc() != 0 { self.purchases.push((index as u16, event.pc())); }
        }
        self.days.clear();
        self.days.resize(w.ctx.ndays, Day::default());
        let cats = w.ctx.cat_names.len();
        self.categories.resize(w.ctx.ndays * cats, 0);
        let mut previous_max = None;
        w.sess.clear();
        for bin in &mut self.bins[first..=last] {
            if bin.count == 0 { continue; }
            if previous_max.is_none_or(|prev| bin.min - prev > WIDTH) {
                let day = ((w.ctx.base + bin.min as i64).div_euclid(DAY_MS) - w.ctx.day0) as usize;
                w.sess.push(Sess { start: bin.min as i64, rev: 0, day: day as u32, purch: false });
                self.days[day].sessions += 1;
            }
            let session = w.sess.len() - 1;
            // At most one session can start per bin, and BINS fits in u16.
            bin.session = session as u16;
            self.days[w.sess[session].day as usize].events += bin.count;
            previous_max = Some(bin.max);
        }
        for &(bin, pc) in &self.purchases {
            let session = &mut w.sess[self.bins[bin as usize].session as usize];
            let day = session.day as usize;
            let price = (pc >> 8) as u64;
            let category = (pc & 255) as usize;
            session.rev += price;
            session.purch = true;
            self.days[day].revenue += price;
            self.categories[day * cats + category] += price;
        }
        self.bins[first..=last].fill(EMPTY);
        for (day, totals) in self.days.iter().enumerate() {
            if totals.events == 0 { continue; }
            // Reuse the row writer's exact category tie-breaker and clearing.
            for (cat, value) in self.categories[day * cats..(day + 1) * cats].iter_mut().enumerate() {
                if *value == 0 { continue; }
                w.cat_rev[cat] = std::mem::take(value);
                w.touched.push(cat as u32);
            }
            w.flush_row(person, day as u32, totals.sessions, totals.events, totals.revenue);
        }
        w.attribute(clicks, ctmp, events.len());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClickKey, Ctx, DateCache, KeyMap, SEVEN_DAYS_MS};

    #[test]
    fn aggregates_match_a_full_sort_across_gaps_days_and_reused_scratch() {
        // Offset the bin origin from UTC midnight. Session starts determine
        // the reporting day, including for purchases after a midnight crossing.
        let base = 17 * 60_000;
        let ws = SEVEN_DAYS_MS + base;
        let we = ws + 9 * DAY_MS;
        let ctx = Ctx {
            ws, we, base, day0: ws / DAY_MS, ndays: 10, dates: DateCache::new(ws, we),
            products: KeyMap::new(0), unknown_cat: 0, cat_names: vec![b"a", b"b"], wide: false,
            camps: KeyMap::new(0), camp_ids: vec![b"organic", b"ad"], camp_channels: vec![b"o", b"c"],
        };
        let person = UserKey::new(b"u12345678");
        let (mut exact, mut bins) = (Worker::new(&ctx, &[]), Worker::new(&ctx, &[]));
        let mut scratch = Scratch::default();
        for shift in [0, 4 * DAY_MS, DAY_MS] {
            let midnight = SEVEN_DAYS_MS + DAY_MS - base + shift;
            let ts = [midnight - WIDTH as i64, midnight, midnight + WIDTH as i64 + 1,
                midnight + DAY_MS + 1, midnight + DAY_MS + WIDTH as i64,
                midnight + DAY_MS + 2 * WIDTH as i64];
            let mut rows: Vec<_> = (0..1031).map(|i| {
                let pc = if i % 3 == 0 { (100 << 8) | (i % 2) as u32 } else { 0 };
                Ev8::new(ts[(i * 5) % ts.len()] as u32, pc)
            }).collect();
            let clicks: Vec<_> = [midnight, midnight - SEVEN_DAYS_MS, midnight - 1].iter()
                .map(|&t| Cl24 { cid: ClickKey::new(b"k8"), ts: t as u32, camp: 1 }).collect();
            assert!(scratch.try_person(&mut bins, person, &rows, &mut clicks.clone(), &mut Vec::new()));
            rows.sort_unstable();
            exact.person(person, &rows, &mut clicks.clone(), &mut Vec::new());
            assert!(exact.cols == bins.cols);
            let sums = |w: &Worker<'_>| w.acc.iter().map(|c| (c.sessions, c.purchasing, c.assisted, c.revenue)).collect::<Vec<_>>();
            assert_eq!(sums(&exact), sums(&bins));
        }
    }
}
