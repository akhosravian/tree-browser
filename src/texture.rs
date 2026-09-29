//! DDS, KTX and KTX2 textures: the top mip of the first layer / slice, as RGBA. Cubemaps
//! unfold into a horizontal cross.
//! HDR formats are clamped to [0, 1]; sRGB is shown as stored.
use std::io::Read;
use std::path::Path;

use half::f16;
use image::{DynamicImage, RgbaImage};
use texture2ddecoder as t2d;

pub const EXT: &[&str] = &["dds", "ktx", "ktx2"];

const KTX1: &[u8] = b"\xabKTX 11\xbb\r\n\x1a\n";
const KTX2: &[u8] = b"\xabKTX 20\xbb\r\n\x1a\n";

type Blocks = fn(&[u8], usize, usize, &mut [u32]) -> Result<(), &'static str>;

enum Fmt {
    /// 4x4 blocks of `bytes` each.
    Block(Blocks, usize),
    Astc(usize, usize),
    /// Uncompressed: bytes per pixel, pixel -> RGBA.
    Raw(usize, fn(&[u8]) -> [u8; 4]),
    /// Legacy DDS bitmasks (R, G, B, A); `lum` copies R into G and B.
    Masks { bytes: usize, m: [u32; 4], lum: bool },
}

struct Tex<'a> {
    w: usize,
    h: usize,
    fmt: Fmt,
    data: &'a [u8],
    /// KTX1 pads uncompressed rows to 4 bytes.
    row_align: usize,
    /// 6 for a cubemap, else 1; `stride` bytes apart in `data`.
    faces: usize,
    stride: usize,
}

impl Fmt {
    fn size(&self, w: usize, h: usize) -> usize {
        match *self {
            Fmt::Block(_, bytes) => w.div_ceil(4) * h.div_ceil(4) * bytes,
            Fmt::Astc(bw, bh) => w.div_ceil(bw) * h.div_ceil(bh) * 16,
            Fmt::Raw(bytes, _) | Fmt::Masks { bytes, .. } => w * h * bytes,
        }
    }
}

fn u32at(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
}

fn u64at(b: &[u8], o: usize) -> usize {
    b.get(o..o + 8).map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap()) as usize)
}

fn tail(b: &[u8], o: usize) -> Result<&[u8], String> {
    b.get(o..).ok_or_else(|| "truncated".into())
}

pub fn decode(path: &Path) -> Result<DynamicImage, String> {
    let b = std::fs::read(path).map_err(|e| e.to_string())?;
    let mut owned = Vec::new();
    let tex = if b.starts_with(b"DDS ") {
        dds(&b)?
    } else if b.starts_with(KTX1) {
        ktx1(&b)?
    } else if b.starts_with(KTX2) {
        // vkFormat 0 marks Basis Universal (ETC1S or UASTC).
        if u32at(&b, 12) == 0 {
            return basis(&b);
        }
        ktx2(&b, &mut owned)?
    } else {
        return Err("not a DDS or KTX file".into());
    };
    unfold((0..tex.faces).map(|i| tex.face(i)).collect::<Result<_, _>>()?)
}

/// Cubemap faces (+X -X +Y -Y +Z -Z) as a horizontal cross; a single image as is.
fn unfold(mut faces: Vec<RgbaImage>) -> Result<DynamicImage, String> {
    if faces.len() != 6 {
        return faces.pop().map(DynamicImage::ImageRgba8).ok_or_else(|| "no image".into());
    }
    let (w, h) = faces[0].dimensions();
    let mut out = RgbaImage::new(w * 4, h * 3);
    for (f, (c, r)) in faces.iter().zip([(2, 1), (0, 1), (1, 0), (1, 2), (1, 1), (3, 1)]) {
        image::imageops::replace(&mut out, f, (c * w) as i64, (r * h) as i64);
    }
    Ok(DynamicImage::ImageRgba8(out))
}

fn basis(b: &[u8]) -> Result<DynamicImage, String> {
    let t = basisu::Transcoder::new(b).map_err(|e| format!("basis: {e:?}"))?;
    let (w, h) = t.base_dimensions();
    let faces = (0..t.face_count())
        .map(|f| {
            let px = t
                .transcode_image(0, 0, f, basisu::TargetFormat::Rgba32, basisu::DecodeFlags::NONE)
                .map_err(|e| format!("basis: {e:?}"))?;
            RgbaImage::from_raw(w, h, px).ok_or_else(|| "basis: short output".to_string())
        })
        .collect::<Result<_, _>>()?;
    unfold(faces)
}

impl Tex<'_> {
    fn face(&self, i: usize) -> Result<RgbaImage, String> {
        let (w, h) = (self.w, self.h);
        let data = self.data.get(i * self.stride..).ok_or("truncated")?;
        if w == 0 || h == 0 || w > 16384 || h > 16384 {
            return Err(format!("unsupported size {w}×{h}"));
        }
        let mut out = Vec::with_capacity(w * h * 4);
        let px = |p: &[u8], f: &dyn Fn(&[u8]) -> [u8; 4], bytes: usize, out: &mut Vec<u8>| -> Result<(), String> {
            let stride = (w * bytes).next_multiple_of(self.row_align);
            if p.len() < stride * (h - 1) + w * bytes {
                return Err("truncated".into());
            }
            for y in 0..h {
                for x in 0..w {
                    let o = y * stride + x * bytes;
                    out.extend_from_slice(&f(&p[o..o + bytes]));
                }
            }
            Ok(())
        };
        match &self.fmt {
            Fmt::Raw(bytes, f) => px(data, f, *bytes, &mut out)?,
            Fmt::Masks { bytes, m, lum } => {
                let f = |p: &[u8]| {
                    let mut v = [0u8; 4];
                    v[..p.len()].copy_from_slice(p);
                    let v = u32::from_le_bytes(v);
                    let ch = |m: u32, none: u8| {
                        if m == 0 {
                            return none;
                        }
                        let max = (m >> m.trailing_zeros()) as u64;
                        ((((v & m) >> m.trailing_zeros()) as u64 * 255 + max / 2) / max) as u8
                    };
                    let r = ch(m[0], 0);
                    let (g, b) = if *lum { (r, r) } else { (ch(m[1], 0), ch(m[2], 0)) };
                    [r, g, b, ch(m[3], 255)]
                };
                px(data, &f, *bytes, &mut out)?
            }
            Fmt::Block(..) | Fmt::Astc(..) => {
                let mut buf = vec![0u32; w * h];
                match self.fmt {
                    Fmt::Block(f, _) => f(data, w, h, &mut buf),
                    Fmt::Astc(bw, bh) => t2d::decode_astc(data, w, h, bw, bh, &mut buf),
                    _ => unreachable!(),
                }?;
                // The decoders write BGRA.
                for p in buf {
                    let [b, g, r, a] = p.to_le_bytes();
                    out.extend_from_slice(&[r, g, b, a]);
                }
            }
        }
        Ok(RgbaImage::from_raw(w as u32, h as u32, out).unwrap())
    }
}

fn unorm(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn rgba8(p: &[u8]) -> [u8; 4] {
    [p[0], p[1], p[2], p[3]]
}
fn bgra8(p: &[u8]) -> [u8; 4] {
    [p[2], p[1], p[0], p[3]]
}
fn rgb8(p: &[u8]) -> [u8; 4] {
    [p[0], p[1], p[2], 255]
}
fn r8(p: &[u8]) -> [u8; 4] {
    [p[0], p[0], p[0], 255]
}
fn rg8(p: &[u8]) -> [u8; 4] {
    [p[0], p[1], 0, 255]
}
fn rgba16f(p: &[u8]) -> [u8; 4] {
    std::array::from_fn(|i| unorm(f16::from_le_bytes([p[2 * i], p[2 * i + 1]]).to_f32()))
}
fn rgba32f(p: &[u8]) -> [u8; 4] {
    std::array::from_fn(|i| unorm(f32::from_le_bytes(p[4 * i..4 * i + 4].try_into().unwrap())))
}

fn bc6u(d: &[u8], w: usize, h: usize, o: &mut [u32]) -> Result<(), &'static str> {
    t2d::decode_bc6(d, w, h, o, false)
}
fn bc6s(d: &[u8], w: usize, h: usize, o: &mut [u32]) -> Result<(), &'static str> {
    t2d::decode_bc6(d, w, h, o, true)
}

/// ATC color with BC2-style explicit 4-bit alpha in front of each block.
fn atce(d: &[u8], w: usize, h: usize, o: &mut [u32]) -> Result<(), &'static str> {
    let bw = w.div_ceil(4);
    let n = bw * h.div_ceil(4);
    let d = d.get(..n * 16).ok_or("Not enough data to decode image!")?;
    let color: Vec<u8> = d.as_chunks::<16>().0.iter().flat_map(|b| b[8..].iter().copied()).collect();
    t2d::decode_atc_rgb4(&color, w, h, o)?;
    for y in 0..h {
        for x in 0..w {
            let (blk, i) = ((y / 4) * bw + x / 4, (y % 4) * 4 + x % 4);
            let a = (d[blk * 16 + i / 2] >> ((i & 1) * 4) & 0xf) as u32 * 17;
            let p = &mut o[y * w + x];
            *p = *p & 0x00ff_ffff | a << 24;
        }
    }
    Ok(())
}

/// LATC2: BC5 with luminance in the first channel and alpha in the second.
fn latc2(d: &[u8], w: usize, h: usize, o: &mut [u32]) -> Result<(), &'static str> {
    t2d::decode_bc5(d, w, h, o)?;
    for p in &mut o[..w * h] {
        let [_, a, l, _] = p.to_le_bytes();
        *p = u32::from_le_bytes([l, l, l, a]);
    }
    Ok(())
}

/// ASTC block sizes in the order both GL and Vulkan enumerate them.
const ASTC: [(usize, usize); 14] =
    [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6), (10, 8), (10, 10), (12, 10), (12, 12)];

fn dds(b: &[u8]) -> Result<Tex<'_>, String> {
    if b.len() < 128 {
        return Err("truncated".into());
    }
    let (h, w) = (u32at(b, 12) as usize, u32at(b, 16) as usize);
    let (flags, fourcc) = (u32at(b, 80), &b[84..88]);
    let mut start = 128;
    let mut cube = u32at(b, 112) & 0xfe00 == 0xfe00;
    let fmt = if flags & 0x4 != 0 {
        match fourcc {
            b"DXT1" => Fmt::Block(t2d::decode_bc1, 8),
            b"DXT2" | b"DXT3" => Fmt::Block(t2d::decode_bc2, 16),
            b"DXT4" | b"DXT5" => Fmt::Block(t2d::decode_bc3, 16),
            b"ATI1" | b"BC4U" => Fmt::Block(t2d::decode_bc4, 8),
            b"ATI2" | b"BC5U" => Fmt::Block(t2d::decode_bc5, 16),
            b"ATC " => Fmt::Block(t2d::decode_atc_rgb4, 8),
            b"ATCI" => Fmt::Block(t2d::decode_atc_rgba8, 16),
            b"ATCA" | b"ATCE" => Fmt::Block(atce, 16),
            // bimg's own ASTC codes; ':' is the digit after '9'.
            b"AS44" => Fmt::Astc(4, 4),
            b"AS55" => Fmt::Astc(5, 5),
            b"AS66" => Fmt::Astc(6, 6),
            b"AS85" => Fmt::Astc(8, 5),
            b"AS86" => Fmt::Astc(8, 6),
            b"AS:5" => Fmt::Astc(10, 5),
            b"DX10" => {
                start = 148;
                cube = u32at(b, 136) & 0x4 != 0;
                dxgi(u32at(b, 128))?
            }
            // D3DFMT codes stored as numbers.
            [113, 0, 0, 0] => Fmt::Raw(8, rgba16f),
            [116, 0, 0, 0] => Fmt::Raw(16, rgba32f),
            _ => return Err(format!("unsupported DDS format {:?}", String::from_utf8_lossy(fourcc))),
        }
    } else {
        let bits = u32at(b, 88) as usize;
        if !matches!(bits, 8 | 16 | 24 | 32) {
            return Err(format!("unsupported DDS bit count {bits}"));
        }
        let a = if flags & 0x3 != 0 { u32at(b, 104) } else { 0 };
        Fmt::Masks { bytes: bits / 8, m: [u32at(b, 92), u32at(b, 96), u32at(b, 100), a], lum: flags & 0x20000 != 0 }
    };
    // Each face carries its whole mip chain.
    let mips = if u32at(b, 8) & 0x20000 != 0 { u32at(b, 28).clamp(1, 32) } else { 1 };
    let stride = (0..mips).map(|k| fmt.size((w >> k).max(1), (h >> k).max(1))).sum();
    Ok(Tex { w, h, fmt, data: tail(b, start)?, row_align: 1, faces: if cube { 6 } else { 1 }, stride })
}

fn dxgi(f: u32) -> Result<Fmt, String> {
    Ok(match f {
        2 => Fmt::Raw(16, rgba32f),
        10 => Fmt::Raw(8, rgba16f),
        28 | 29 => Fmt::Raw(4, rgba8),
        49 => Fmt::Raw(2, rg8),
        61 => Fmt::Raw(1, r8),
        87 | 91 => Fmt::Raw(4, bgra8),
        70..=72 => Fmt::Block(t2d::decode_bc1, 8),
        73..=75 => Fmt::Block(t2d::decode_bc2, 16),
        76..=78 => Fmt::Block(t2d::decode_bc3, 16),
        79 | 80 => Fmt::Block(t2d::decode_bc4, 8),
        82 | 83 => Fmt::Block(t2d::decode_bc5, 16),
        95 => Fmt::Block(bc6u, 16),
        96 => Fmt::Block(bc6s, 16),
        97..=99 => Fmt::Block(t2d::decode_bc7, 16),
        _ => return Err(format!("unsupported DXGI format {f}")),
    })
}

fn ktx1(b: &[u8]) -> Result<Tex<'_>, String> {
    if b.len() < 68 {
        return Err("truncated".into());
    }
    if u32at(b, 12) != 0x0403_0201 {
        return Err("big-endian KTX isn't supported".into());
    }
    let gl = u32at(b, 28);
    let fmt = match gl {
        0x83f0 | 0x8c4c => Fmt::Block(t2d::decode_bc1, 8),
        0x83f1 | 0x8c4d => Fmt::Block(t2d::decode_bc1a, 8),
        0x83f2 | 0x8c4e => Fmt::Block(t2d::decode_bc2, 16),
        0x83f3 | 0x8c4f => Fmt::Block(t2d::decode_bc3, 16),
        0x8dbb => Fmt::Block(t2d::decode_bc4, 8),
        0x8dbd => Fmt::Block(t2d::decode_bc5, 16),
        0x8c72 => Fmt::Block(latc2, 16),
        0x8e8c | 0x8e8d => Fmt::Block(t2d::decode_bc7, 16),
        0x8e8e => Fmt::Block(bc6s, 16),
        0x8e8f => Fmt::Block(bc6u, 16),
        0x8d64 => Fmt::Block(t2d::decode_etc1, 8),
        0x9270 => Fmt::Block(t2d::decode_eacr, 8),
        0x9272 => Fmt::Block(t2d::decode_eacrg, 16),
        0x9274 | 0x9275 => Fmt::Block(t2d::decode_etc2_rgb, 8),
        0x9276 | 0x9277 => Fmt::Block(t2d::decode_etc2_rgba1, 8),
        0x9278 | 0x9279 => Fmt::Block(t2d::decode_etc2_rgba8, 16),
        0x93b0..=0x93bd => {
            let (bw, bh) = ASTC[(gl - 0x93b0) as usize];
            Fmt::Astc(bw, bh)
        }
        0x93d0..=0x93dd => {
            let (bw, bh) = ASTC[(gl - 0x93d0) as usize];
            Fmt::Astc(bw, bh)
        }
        0x8058 | 0x8c43 => Fmt::Raw(4, rgba8),
        0x8051 | 0x8c41 => Fmt::Raw(3, rgb8),
        0x8229 => Fmt::Raw(1, r8),
        0x822b => Fmt::Raw(2, rg8),
        0x881a => Fmt::Raw(8, rgba16f),
        0x8814 => Fmt::Raw(16, rgba32f),
        _ => return Err(format!("unsupported KTX format 0x{gl:04x}")),
    };
    // Header, key/value data, then level 0's imageSize: one face, or the whole
    // level for arrays.
    let start = 64 + u32at(b, 60) as usize + 4;
    let (size, layers) = (u32at(b, start - 4) as usize, u32at(b, 48) as usize);
    let cube = u32at(b, 52) == 6;
    let stride = if layers > 0 { size / (layers * if cube { 6 } else { 1 }) } else { size.next_multiple_of(4) };
    let (w, h) = (u32at(b, 36) as usize, (u32at(b, 40) as usize).max(1));
    Ok(Tex { w, h, fmt, data: tail(b, start)?, row_align: 4, faces: if cube { 6 } else { 1 }, stride })
}

fn ktx2<'a>(b: &'a [u8], owned: &'a mut Vec<u8>) -> Result<Tex<'a>, String> {
    if b.len() < 104 {
        return Err("truncated".into());
    }
    let (vk, scheme) = (u32at(b, 12), u32at(b, 44));
    let fmt = match vk {
        9 | 15 => Fmt::Raw(1, r8),
        16 | 22 => Fmt::Raw(2, rg8),
        23 | 29 => Fmt::Raw(3, rgb8),
        37 | 43 => Fmt::Raw(4, rgba8),
        44 | 50 => Fmt::Raw(4, bgra8),
        97 => Fmt::Raw(8, rgba16f),
        109 => Fmt::Raw(16, rgba32f),
        131 | 132 => Fmt::Block(t2d::decode_bc1, 8),
        133 | 134 => Fmt::Block(t2d::decode_bc1a, 8),
        135 | 136 => Fmt::Block(t2d::decode_bc2, 16),
        137 | 138 => Fmt::Block(t2d::decode_bc3, 16),
        139 => Fmt::Block(t2d::decode_bc4, 8),
        141 => Fmt::Block(t2d::decode_bc5, 16),
        143 => Fmt::Block(bc6u, 16),
        144 => Fmt::Block(bc6s, 16),
        145 | 146 => Fmt::Block(t2d::decode_bc7, 16),
        147 | 148 => Fmt::Block(t2d::decode_etc2_rgb, 8),
        149 | 150 => Fmt::Block(t2d::decode_etc2_rgba1, 8),
        151 | 152 => Fmt::Block(t2d::decode_etc2_rgba8, 16),
        153 => Fmt::Block(t2d::decode_eacr, 8),
        155 => Fmt::Block(t2d::decode_eacrg, 16),
        157..=184 => {
            let (bw, bh) = ASTC[(vk - 157) as usize / 2];
            Fmt::Astc(bw, bh)
        }
        _ => return Err(format!("unsupported KTX2 vkFormat {vk}")),
    };
    // Level 0 is the first entry of the level index.
    let (off, len) = (u64at(b, 80), u64at(b, 88));
    let level = b.get(off..off.saturating_add(len)).ok_or("truncated")?;
    let data = match scheme {
        0 => level,
        2 => {
            ruzstd::streaming_decoder::StreamingDecoder::new(level)
                .map_err(|e| format!("zstd: {e}"))?
                .read_to_end(owned)
                .map_err(|e| format!("zstd: {e}"))?;
            owned.as_slice()
        }
        _ => return Err(format!("unsupported KTX2 supercompression {scheme}")),
    };
    // Level data is layer-major, then face.
    let (faces, layers) = (u32at(b, 36).max(1) as usize, u32at(b, 32).max(1) as usize);
    let stride = data.len() / (faces * layers);
    let (w, h) = (u32at(b, 20) as usize, (u32at(b, 24) as usize).max(1));
    Ok(Tex { w, h, fmt, data, row_align: 1, faces: if faces == 6 { 6 } else { 1 }, stride })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(img: &DynamicImage, x: u32, y: u32) -> [u8; 4] {
        img.to_rgba8().get_pixel(x, y).0
    }

    fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("tb-tex-{}-{name}", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn dds_bc1_solid_red() {
        let mut b = vec![0u8; 128];
        b[..4].copy_from_slice(b"DDS ");
        b[12..16].copy_from_slice(&4u32.to_le_bytes());
        b[16..20].copy_from_slice(&4u32.to_le_bytes());
        b[80..84].copy_from_slice(&4u32.to_le_bytes());
        b[84..88].copy_from_slice(b"DXT1");
        // Both endpoints pure red in RGB565, all indices 0.
        b.extend_from_slice(&[0x00, 0xf8, 0x00, 0xf8, 0, 0, 0, 0]);
        let img = decode(&write("bc1.dds", &b)).unwrap();
        assert_eq!((img.width(), img.height()), (4, 4));
        assert_eq!(px(&img, 3, 3), [255, 0, 0, 255]);
    }

    #[test]
    fn dds_bgra_masks() {
        let mut b = vec![0u8; 128];
        b[..4].copy_from_slice(b"DDS ");
        b[12..16].copy_from_slice(&1u32.to_le_bytes());
        b[16..20].copy_from_slice(&2u32.to_le_bytes());
        b[80..84].copy_from_slice(&0x41u32.to_le_bytes());
        b[88..92].copy_from_slice(&32u32.to_le_bytes());
        for (o, m) in [(92, 0x00ff_0000u32), (96, 0xff00), (100, 0xff), (104, 0xff00_0000)] {
            b[o..o + 4].copy_from_slice(&m.to_le_bytes());
        }
        b.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let img = decode(&write("bgra.dds", &b)).unwrap();
        assert_eq!(px(&img, 0, 0), [3, 2, 1, 4]);
        assert_eq!(px(&img, 1, 0), [7, 6, 5, 8]);
    }

    #[test]
    fn ktx1_rgb8_rows_padded_to_4() {
        let mut b = vec![0u8; 64];
        b[..12].copy_from_slice(KTX1);
        b[12..16].copy_from_slice(&0x0403_0201u32.to_le_bytes());
        b[28..32].copy_from_slice(&0x8051u32.to_le_bytes());
        b[36..40].copy_from_slice(&1u32.to_le_bytes());
        b[40..44].copy_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&[10, 20, 30, 0, 40, 50, 60, 0]);
        let img = decode(&write("rgb.ktx", &b)).unwrap();
        assert_eq!(px(&img, 0, 1), [40, 50, 60, 255]);
    }

    #[test]
    fn ktx1_cubemap_unfolds_to_a_cross() {
        let mut b = vec![0u8; 64];
        b[..12].copy_from_slice(KTX1);
        b[12..16].copy_from_slice(&0x0403_0201u32.to_le_bytes());
        b[28..32].copy_from_slice(&0x8058u32.to_le_bytes());
        b[36..40].copy_from_slice(&1u32.to_le_bytes());
        b[40..44].copy_from_slice(&1u32.to_le_bytes());
        b[52..56].copy_from_slice(&6u32.to_le_bytes());
        b.extend_from_slice(&4u32.to_le_bytes());
        for f in 0..6u8 {
            b.extend_from_slice(&[f * 40, 0, 0, 255]);
        }
        let img = decode(&write("cube.ktx", &b)).unwrap();
        assert_eq!((img.width(), img.height()), (4, 3));
        // +X, -X, +Y, -Y, +Z, -Z
        for (f, (x, y)) in [(2, 1), (0, 1), (1, 0), (1, 2), (1, 1), (3, 1)].into_iter().enumerate() {
            assert_eq!(px(&img, x, y), [f as u8 * 40, 0, 0, 255]);
        }
        assert_eq!(px(&img, 0, 0)[3], 0);
    }

    #[test]
    fn ktx2_bad_basis_is_an_error() {
        let mut b = vec![0u8; 104];
        b[..12].copy_from_slice(KTX2);
        b[44..48].copy_from_slice(&1u32.to_le_bytes());
        assert!(decode(&write("basis.ktx2", &b)).unwrap_err().starts_with("basis"));
    }
}
