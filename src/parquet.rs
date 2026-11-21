//! Minimal Parquet writer: one row group, one uncompressed PLAIN data page (V1)
//! per column. Supports required strings, optional strings, DATE and INT64.

use std::io::{IoSlice, Write};
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Str,
    OptStr,
    Date,
    I32,
    I64,
}

pub struct Column {
    name: &'static str,
    kind: Kind,
    values: Vec<u8>,
    defs: Vec<u8>,
    n: usize,
}

impl Column {
    pub fn new(name: &'static str, kind: Kind, cap_rows: usize) -> Column {
        let per = match kind {
            Kind::Str => 14,
            Kind::OptStr => 2,
            Kind::Date | Kind::I32 => 4,
            Kind::I64 => 8,
        };
        let values = Vec::with_capacity(cap_rows * per + 16);
        #[cfg(all(target_os = "linux", not(test)))]
        if values.capacity() >= (1 << 20) {
            let ptr = values.as_ptr() as usize;
            let a = (ptr + 4095) & !4095;
            let e = (ptr + values.capacity()) & !4095;
            if e > a {
                unsafe { libc::madvise(a as *mut libc::c_void, e - a, libc::MADV_HUGEPAGE) };
            }
        }
        Column {
            name,
            kind,
            values,
            defs: if kind == Kind::OptStr { Vec::with_capacity(cap_rows / 8 + 16) } else { Vec::new() },
            n: 0,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Advise the kernel to drop backing pages of this column after writing to disk
    /// (moves teardown off the serial process-exit path).
    pub fn release_pages(&self) {
        #[cfg(all(target_os = "linux", not(test)))]
        {
            for (ptr, cap) in [
                (self.values.as_ptr() as usize, self.values.capacity()),
                (self.defs.as_ptr() as usize, self.defs.capacity()),
            ] {
                if cap >= 131_072 {
                    let a = (ptr + 4095) & !4095;
                    let e = (ptr + cap) & !4095;
                    if e > a {
                        unsafe { libc::madvise(a as *mut libc::c_void, e - a, libc::MADV_DONTNEED) };
                    }
                }
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn grow_val(&mut self, extra: usize) {
        self.values.reserve(extra.max(4096));
    }

    #[cold]
    #[inline(never)]
    fn grow_def(&mut self) {
        self.defs.reserve(4096);
    }

    #[inline(always)]
    fn push_def(&mut self, present: bool) {
        let byte_idx = self.n >> 3;
        let bit = self.n & 7;
        if bit == 0 {
            if byte_idx == self.defs.capacity() {
                self.grow_def();
            }
            unsafe {
                *self.defs.as_mut_ptr().add(byte_idx) = present as u8;
                self.defs.set_len(byte_idx + 1);
            }
        } else if present {
            unsafe { *self.defs.get_unchecked_mut(byte_idx) |= 1 << bit };
        }
    }

    #[inline(always)]
    fn ensure_val(&mut self, extra: usize) {
        if self.values.capacity() - self.values.len() < extra {
            self.grow_val(extra);
        }
    }

    /// Fast path for inline `Key8::as_slice` slices backed by a 16-byte stack buffer.
    #[inline(always)]
    pub unsafe fn push_str_inline16(&mut self, s: &[u8]) {
        debug_assert!(self.kind == Kind::Str && s.len() <= 16);
        let slen = s.len();
        self.ensure_val(20);
        let len = self.values.len();
        let dst = self.values.as_mut_ptr().add(len);
        (dst as *mut u32).write_unaligned((slen as u32).to_le());
        (dst.add(4) as *mut u128).write_unaligned((s.as_ptr() as *const u128).read_unaligned());
        self.values.set_len(len + 4 + slen);
        self.n += 1;
    }

    #[inline(always)]
    pub fn push_null(&mut self) {
        debug_assert!(self.kind == Kind::OptStr);
        self.push_def(false);
        self.n += 1;
    }

    #[inline(always)]
    pub fn push_str(&mut self, s: &[u8]) {
        debug_assert!(self.kind == Kind::Str);
        let slen = s.len();
        self.ensure_val(4 + slen);
        unsafe {
            let len = self.values.len();
            let dst = self.values.as_mut_ptr().add(len);
            (dst as *mut u32).write_unaligned((slen as u32).to_le());
            std::ptr::copy_nonoverlapping(s.as_ptr(), dst.add(4), slen);
            self.values.set_len(len + 4 + slen);
        }
        self.n += 1;
    }

    #[inline(always)]
    pub fn push_opt_str(&mut self, s: Option<&[u8]>) {
        debug_assert!(self.kind == Kind::OptStr);
        self.push_def(s.is_some());
        if let Some(s) = s {
            let slen = s.len();
            self.ensure_val(4 + slen);
            unsafe {
                let len = self.values.len();
                let dst = self.values.as_mut_ptr().add(len);
                (dst as *mut u32).write_unaligned((slen as u32).to_le());
                std::ptr::copy_nonoverlapping(s.as_ptr(), dst.add(4), slen);
                self.values.set_len(len + 4 + slen);
            }
        }
        self.n += 1;
    }

    #[inline(always)]
    pub fn push_date(&mut self, d: i32) {
        self.ensure_val(4);
        unsafe {
            let len = self.values.len();
            (self.values.as_mut_ptr().add(len) as *mut i32).write_unaligned(d.to_le());
            self.values.set_len(len + 4);
        }
        self.n += 1;
    }

    #[inline(always)]
    pub fn push_i32(&mut self, v: i32) {
        self.ensure_val(4);
        unsafe {
            let len = self.values.len();
            (self.values.as_mut_ptr().add(len) as *mut i32).write_unaligned(v.to_le());
            self.values.set_len(len + 4);
        }
        self.n += 1;
    }

    #[inline(always)]
    pub fn push_i64(&mut self, v: i64) {
        self.ensure_val(8);
        unsafe {
            let len = self.values.len();
            (self.values.as_mut_ptr().add(len) as *mut i64).write_unaligned(v.to_le());
            self.values.set_len(len + 8);
        }
        self.n += 1;
    }
}

struct Thrift {
    buf: Vec<u8>,
    last: i16,
    stack: Vec<i16>,
}

impl Thrift {
    fn new() -> Thrift {
        Thrift { buf: Vec::with_capacity(1024), last: 0, stack: Vec::new() }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }
    fn header(&mut self, id: i16, ty: u8) {
        let d = id - self.last;
        if d > 0 && d <= 15 {
            self.buf.push(((d as u8) << 4) | ty);
        } else {
            self.buf.push(ty);
            self.varint(((id << 1) ^ (id >> 15)) as u16 as u64);
        }
        self.last = id;
    }
    fn i32(&mut self, id: i16, v: i32) {
        self.header(id, 5);
        self.varint(((v << 1) ^ (v >> 31)) as u32 as u64);
    }
    fn i64(&mut self, id: i16, v: i64) {
        self.header(id, 6);
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }
    fn bin(&mut self, id: i16, b: &[u8]) {
        self.header(id, 8);
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn begin_struct(&mut self, id: i16) {
        self.header(id, 12);
        self.push_scope();
    }
    fn push_scope(&mut self) {
        self.stack.push(self.last);
        self.last = 0;
    }
    fn end_struct(&mut self) {
        self.buf.push(0);
        self.last = self.stack.pop().unwrap();
    }
    fn list(&mut self, id: i16, elem: u8, n: usize) {
        self.header(id, 9);
        self.list_header(elem, n);
    }
    fn list_header(&mut self, elem: u8, n: usize) {
        if n < 15 {
            self.buf.push(((n as u8) << 4) | elem);
        } else {
            self.buf.push(0xf0 | elem);
            self.varint(n as u64);
        }
    }
}

fn physical(kind: Kind) -> i32 {
    match kind {
        Kind::Str | Kind::OptStr => 6,
        Kind::Date | Kind::I32 => 1,
        Kind::I64 => 2,
    }
}

fn write_all_vectored(f: &mut std::fs::File, slices: &mut [&[u8]]) -> std::io::Result<()> {
    let mut cur = slices;
    while !cur.is_empty() {
        let mut iov = [IoSlice::new(&[]); 32];
        let count = cur.len().min(32);
        for i in 0..count {
            iov[i] = IoSlice::new(cur[i]);
        }
        let mut n = match f.write_vectored(&iov[..count]) {
            Ok(0) => return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "write_vectored returned 0")),
            Ok(n) => n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        while !cur.is_empty() && n >= cur[0].len() {
            n -= cur[0].len();
            cur = &mut cur[1..];
        }
        if !cur.is_empty() && n > 0 {
            cur[0] = &cur[0][n..];
        }
    }
    Ok(())
}

/// Writes a Parquet file holding `cols` (all with the same row count) in a single
/// vectored `writev` syscall without copying column or definition-level buffers.
pub fn write_file(path: &Path, cols: &[Column]) -> std::io::Result<()> {
    let nrows = cols.first().map(|c| c.n).unwrap_or(0);
    let mut f = std::fs::File::create(path)?;
    let mut off = 4i64;
    let mut chunks: Vec<(i64, i64)> = Vec::with_capacity(cols.len());
    let mut hdrs: Vec<Vec<u8>> = Vec::with_capacity(cols.len());
    if nrows > 0 {
        for c in cols {
            assert_eq!(c.n, nrows);
            let mut lv_hdr = [0u8; 16];
            let mut lv_hdr_len = 0usize;
            let page_len;
            if c.kind == Kind::OptStr {
                let groups = c.n.div_ceil(8);
                debug_assert_eq!(c.defs.len(), groups);
                let mut h = ((groups as u64) << 1) | 1;
                let mut pos = 4usize;
                while h >= 0x80 {
                    lv_hdr[pos] = (h as u8) | 0x80;
                    h >>= 7;
                    pos += 1;
                }
                lv_hdr[pos] = h as u8;
                pos += 1;
                let lv_len = (pos - 4 + c.defs.len()) as u32;
                lv_hdr[0..4].copy_from_slice(&lv_len.to_le_bytes());
                lv_hdr_len = pos;
                page_len = pos + c.defs.len() + c.values.len();
            } else {
                page_len = c.values.len();
            }
            assert!(page_len < i32::MAX as usize, "parquet page too large");
            let mut t = Thrift::new();
            t.i32(1, 0); // DATA_PAGE
            t.i32(2, page_len as i32);
            t.i32(3, page_len as i32);
            t.begin_struct(5);
            t.i32(1, c.n as i32);
            t.i32(2, 0); // PLAIN
            t.i32(3, 3); // RLE
            t.i32(4, 3); // RLE
            t.end_struct();
            t.buf.push(0);
            let thdr_len = t.buf.len();
            if lv_hdr_len > 0 {
                t.buf.extend_from_slice(&lv_hdr[..lv_hdr_len]);
            }
            let chunk_size = (thdr_len + page_len) as i64;
            hdrs.push(t.buf);
            chunks.push((off, chunk_size));
            off += chunk_size;
        }
    }

    let mut t = Thrift::new();
    t.i32(1, 1);
    t.list(2, 12, cols.len() + 1);
    // root
    t.push_scope();
    t.bin(4, b"schema");
    t.i32(5, cols.len() as i32);
    t.end_struct();
    for c in cols {
        t.push_scope();
        t.i32(1, physical(c.kind));
        t.i32(3, if c.kind == Kind::OptStr { 1 } else { 0 });
        t.bin(4, c.name.as_bytes());
        match c.kind {
            Kind::Str | Kind::OptStr => {
                t.i32(6, 0); // UTF8
                t.begin_struct(10);
                t.begin_struct(1); // STRING
                t.end_struct();
                t.end_struct();
            }
            Kind::Date => {
                t.i32(6, 6); // DATE
                t.begin_struct(10);
                t.begin_struct(6); // DATE
                t.end_struct();
                t.end_struct();
            }
            Kind::I32 | Kind::I64 => {}
        }
        t.end_struct();
    }
    t.i64(3, nrows as i64);
    if nrows > 0 {
        t.list(4, 12, 1);
        t.push_scope();
        t.list(1, 12, cols.len());
        let mut total = 0i64;
        for (c, &(off, size)) in cols.iter().zip(&chunks) {
            total += size;
            t.push_scope();
            t.i64(2, off);
            t.begin_struct(3);
            t.i32(1, physical(c.kind));
            t.list(2, 5, 2);
            t.varint(0); // PLAIN (zigzag 0)
            t.varint(6); // RLE (zigzag 3)
            t.list(3, 8, 1);
            t.varint(c.name.len() as u64);
            t.buf.extend_from_slice(c.name.as_bytes());
            t.i32(4, 0); // UNCOMPRESSED
            t.i64(5, c.n as i64);
            t.i64(6, size);
            t.i64(7, size);
            t.i64(9, off);
            t.end_struct();
            t.end_struct();
        }
        t.i64(2, total);
        t.i64(3, nrows as i64);
        t.end_struct();
    } else {
        t.list(4, 12, 0);
    }
    t.bin(6, b"shopping-sessions");
    t.buf.push(0);
    let flen = (t.buf.len() as u32).to_le_bytes();
    t.buf.extend_from_slice(&flen);
    t.buf.extend_from_slice(b"PAR1");

    let mut slices: Vec<&[u8]> = Vec::with_capacity(cols.len() * 3 + 2);
    slices.push(b"PAR1");
    if nrows > 0 {
        for (c, hdr) in cols.iter().zip(&hdrs) {
            slices.push(hdr.as_slice());
            if c.kind == Kind::OptStr && !c.defs.is_empty() {
                slices.push(c.defs.as_slice());
            }
            if !c.values.is_empty() {
                slices.push(c.values.as_slice());
            }
        }
    }
    slices.push(t.buf.as_slice());
    write_all_vectored(&mut f, &mut slices)
}
