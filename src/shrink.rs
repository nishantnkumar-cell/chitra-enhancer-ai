//! Shrinks an upscaling model's result to half size inside the model, so a 4x
//! model asked for a 2x picture does the shrinking on the graphics card and
//! only a quarter of the data has to come back.
//!
//! The filter is the Lanczos-3 the program otherwise applies afterwards,
//! written as two convolutions added to the end of the ONNX graph (after
//! clipping to black..white). An ONNX file
//! is a protobuf message; only the few fields involved are read here, and
//! everything else is copied byte for byte. Anything unexpected means "leave the
//! model alone".

/// ONNX's numbers for the element types used.
const FLOAT: u64 = 1;
const INT64: u64 = 7;
const FLOAT16: u64 = 10;

/// The oldest operator set in which `Pad` takes its pads as an input.
const OLDEST_OPSET: u64 = 11;

/// Taps of the filter: it reaches 3 samples of the smaller picture each way.
const TAPS: usize = 12;

#[derive(Clone, Copy)]
enum Value<'a> {
    Number(u64),
    Bytes(&'a [u8]),
    Fixed,
}

#[derive(Clone, Copy)]
struct Field<'a> {
    number: u64,
    value: Value<'a>,
    /// The field as it stands in the file, key included.
    whole: &'a [u8],
}

fn read_varint(data: &[u8], at: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *data.get(*at)?;
        *at += 1;
        v |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Some(v);
        }
    }
    None
}

/// The fields of one message, or None if it is not well formed.
fn fields(data: &[u8]) -> Option<Vec<Field<'_>>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < data.len() {
        let start = at;
        let key = read_varint(data, &mut at)?;
        let take = |at: &mut usize, n: usize| {
            let end = at.checked_add(n)?;
            let s = data.get(*at..end)?;
            *at = end;
            Some(s)
        };
        let value = match key & 7 {
            0 => Value::Number(read_varint(data, &mut at)?),
            1 => take(&mut at, 8).map(|_| Value::Fixed)?,
            2 => {
                let n = usize::try_from(read_varint(data, &mut at)?).ok()?;
                Value::Bytes(take(&mut at, n)?)
            }
            5 => take(&mut at, 4).map(|_| Value::Fixed)?,
            _ => return None,
        };
        out.push(Field {
            number: key >> 3,
            value,
            whole: &data[start..at],
        });
    }
    Some(out)
}

fn bytes_of<'a>(list: &[Field<'a>], number: u64) -> Option<&'a [u8]> {
    list.iter().find_map(|f| match f.value {
        Value::Bytes(b) if f.number == number => Some(b),
        _ => None,
    })
}

fn number_of(list: &[Field<'_>], number: u64) -> Option<u64> {
    list.iter().find_map(|f| match f.value {
        Value::Number(n) if f.number == number => Some(n),
        _ => None,
    })
}

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn put_number(out: &mut Vec<u8>, field: u64, v: u64) {
    varint(out, field << 3);
    varint(out, v);
}

fn put_bytes(out: &mut Vec<u8>, field: u64, v: &[u8]) {
    varint(out, field << 3 | 2);
    varint(out, v.len() as u64);
    out.extend_from_slice(v);
}

fn message(build: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut out = Vec::new();
    build(&mut out);
    out
}

/// An attribute holding one string.
fn text(name: &str, value: &str) -> Vec<u8> {
    message(|a| {
        put_bytes(a, 1, name.as_bytes());
        put_bytes(a, 4, value.as_bytes());
        put_number(a, 20, 3);
    })
}

fn one(name: &str, value: u64) -> Vec<u8> {
    message(|a| {
        put_bytes(a, 1, name.as_bytes());
        put_number(a, 3, value);
        put_number(a, 20, 2);
    })
}

fn many(name: &str, values: &[u64]) -> Vec<u8> {
    message(|a| {
        put_bytes(a, 1, name.as_bytes());
        for &v in values {
            put_number(a, 8, v);
        }
        put_number(a, 20, 7);
    })
}

fn node(op: &str, inputs: &[&str], output: &str, attributes: &[Vec<u8>]) -> Vec<u8> {
    message(|n| {
        for i in inputs {
            put_bytes(n, 1, i.as_bytes());
        }
        put_bytes(n, 2, output.as_bytes());
        put_bytes(n, 3, output.as_bytes());
        put_bytes(n, 4, op.as_bytes());
        for a in attributes {
            put_bytes(n, 5, a);
        }
    })
}

fn tensor(name: &str, kind: u64, dims: &[u64], raw: &[u8]) -> Vec<u8> {
    message(|t| {
        for &d in dims {
            put_number(t, 1, d);
        }
        put_number(t, 2, kind);
        put_bytes(t, 8, name.as_bytes());
        put_bytes(t, 9, raw);
    })
}

/// Weights for halving: output sample `i` sits at input position `2i + 0.5`
/// and the filter is stretched to twice its width.
fn taps() -> [f32; TAPS] {
    let lanczos = |x: f64| {
        if x == 0.0 {
            return 1.0;
        }
        if x.abs() >= 3.0 {
            return 0.0;
        }
        let p = std::f64::consts::PI * x;
        3.0 * p.sin() * (p / 3.0).sin() / (p * p)
    };
    let raw: [f64; TAPS] = std::array::from_fn(|j| lanczos((j as f64 - 5.5) / 2.0));
    let sum: f64 = raw.iter().sum();
    raw.map(|w| (w / sum) as f32)
}

/// IEEE half precision, rounded to nearest. The weights are all well inside
/// its normal range or small enough to be nothing.
fn half_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    let rounded = ((exponent as u32) << 10) + (((bits & 0x7f_ffff) + 0x1000) >> 13);
    sign | rounded as u16
}

/// The three filter weights for one direction, once for each colour.
fn kernel(name: &str, kind: u64, dims: [u64; 4]) -> Vec<u8> {
    let mut raw = Vec::new();
    for _ in 0..3 {
        for w in taps() {
            if kind == FLOAT16 {
                raw.extend(half_bits(w).to_le_bytes());
            } else {
                raw.extend(w.to_le_bytes());
            }
        }
    }
    tensor(name, kind, &dims, &raw)
}

/// `model` with its one picture output shrunk to half the width and height, or
/// None if it is not a model this can safely be done to.
pub fn halve_output(model: &[u8]) -> Option<Vec<u8>> {
    let top = fields(model)?;
    // The default operator set (an empty domain) must be new enough.
    let opset = top
        .iter()
        .filter(|f| f.number == 8)
        .find_map(|f| match f.value {
            Value::Bytes(b) => {
                let set = fields(b)?;
                let domain = bytes_of(&set, 1).unwrap_or_default();
                (domain.is_empty() || domain == b"ai.onnx")
                    .then(|| number_of(&set, 2))
                    .flatten()
            }
            _ => None,
        })?;
    if opset < OLDEST_OPSET {
        return None;
    }
    let graph = fields(bytes_of(&top, 7)?)?;
    let mut outputs = graph.iter().filter(|f| f.number == 12);
    let (only, None) = (outputs.next()?, outputs.next()) else {
        return None;
    };
    let Value::Bytes(info) = only.value else {
        return None;
    };
    let info = fields(info)?;
    let old = std::str::from_utf8(bytes_of(&info, 1)?).ok()?;
    let tensor_type = fields(bytes_of(&fields(bytes_of(&info, 2)?)?, 1)?)?;
    let kind = number_of(&tensor_type, 1)?;
    if kind != FLOAT && kind != FLOAT16 {
        return None;
    }

    let name = |s: &str| format!("chitra_shrink_{s}");
    let (pads, across, down) = (name("pads"), name("across"), name("down"));
    let (padded, narrow, result) = (name("padded"), name("narrow"), name("out"));
    let (least, most, clipped) = (name("least"), name("most"), name("clipped"));
    let convolve = |input: &str, weights: &str, output: &str, stride: [u64; 2], size: [u64; 2]| {
        node(
            "Conv",
            &[input, weights],
            output,
            &[
                one("group", 3),
                many("strides", &stride),
                many("kernel_shape", &size),
            ],
        )
    };
    let nodes = [
        // A model may overshoot black and white; cut that off first, as the
        // program does when it shrinks the picture itself.
        node("Clip", &[old, &least, &most], &clipped, &[]),
        // The edge samples repeat outwards, 5 on every side.
        node("Pad", &[&clipped, &pads], &padded, &[text("mode", "edge")]),
        convolve(&padded, &across, &narrow, [1, 2], [1, TAPS as u64]),
        convolve(&narrow, &down, &result, [2, 1], [TAPS as u64, 1]),
    ];
    let pad_values: Vec<u8> = [0i64, 0, 5, 5, 0, 0, 5, 5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let limit = |name: &str, v: f32| {
        let raw = if kind == FLOAT16 {
            half_bits(v).to_le_bytes().to_vec()
        } else {
            v.to_le_bytes().to_vec()
        };
        tensor(name, kind, &[], &raw)
    };
    let weights = [
        limit(&least, 0.0),
        limit(&most, 1.0),
        tensor(&pads, INT64, &[8], &pad_values),
        kernel(&across, kind, [3, 1, 1, TAPS as u64]),
        kernel(&down, kind, [3, 1, TAPS as u64, 1]),
    ];
    let new_output = message(|v| {
        let shape = message(|s| {
            for fixed in [1u64, 3] {
                put_bytes(s, 1, &message(|d| put_number(d, 1, fixed)));
            }
            for free in ["chitra_h", "chitra_w"] {
                put_bytes(s, 1, &message(|d| put_bytes(d, 2, free.as_bytes())));
            }
        });
        let kind_and_shape = message(|t| {
            put_number(t, 1, kind);
            put_bytes(t, 2, &shape);
        });
        put_bytes(v, 1, result.as_bytes());
        put_bytes(v, 2, &message(|t| put_bytes(t, 1, &kind_and_shape)));
    });

    let new_graph = message(|g| {
        // New steps go after the old ones, which keeps them in order.
        for f in graph.iter().filter(|f| f.number != 12) {
            g.extend_from_slice(f.whole);
        }
        for n in &nodes {
            put_bytes(g, 1, n);
        }
        for w in &weights {
            put_bytes(g, 5, w);
        }
        put_bytes(g, 12, &new_output);
    });
    Some(message(|m| {
        for f in &top {
            if f.number == 7 {
                put_bytes(m, 7, &new_graph);
            } else {
                m.extend_from_slice(f.whole);
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model with one output of element type `kind`.
    fn model(opset: u64, kind: u64, outputs: usize) -> Vec<u8> {
        let info = message(|v| {
            let t = message(|t| put_number(t, 1, kind));
            put_bytes(v, 1, b"picture");
            put_bytes(v, 2, &message(|m| put_bytes(m, 1, &t)));
        });
        let graph = message(|g| {
            put_bytes(g, 1, &node("Identity", &["in"], "picture", &[]));
            put_bytes(g, 2, b"g");
            for _ in 0..outputs {
                put_bytes(g, 12, &info);
            }
        });
        message(|m| {
            put_number(m, 1, 8);
            put_bytes(m, 7, &graph);
            put_bytes(m, 8, &message(|s| put_number(s, 2, opset)));
        })
    }

    #[test]
    fn the_shrinking_steps_are_added_and_the_output_is_replaced() {
        for kind in [FLOAT, FLOAT16] {
            let before = model(17, kind, 1);
            let after = halve_output(&before).unwrap();
            let top = fields(&after).unwrap();
            assert_eq!(number_of(&top, 1), Some(8), "other fields are kept");
            let graph = fields(bytes_of(&top, 7).unwrap()).unwrap();
            let count = |n| graph.iter().filter(|f| f.number == n).count();
            assert_eq!((count(1), count(5), count(12)), (5, 5, 1));
            let out = fields(bytes_of(&graph, 12).unwrap()).unwrap();
            assert_eq!(bytes_of(&out, 1), Some(&b"chitra_shrink_out"[..]));
        }
    }

    #[test]
    fn anything_unexpected_leaves_the_model_alone() {
        assert!(
            halve_output(&model(10, FLOAT, 1)).is_none(),
            "old operators"
        );
        assert!(
            halve_output(&model(17, INT64, 1)).is_none(),
            "not a picture"
        );
        assert!(halve_output(&model(17, FLOAT, 2)).is_none(), "two outputs");
        assert!(halve_output(&model(17, FLOAT, 0)).is_none(), "no output");
        assert!(halve_output(b"not a model").is_none());
        assert!(halve_output(&[]).is_none());
        // Cut short anywhere, it must give up rather than fail.
        let whole = model(17, FLOAT, 1);
        for cut in 0..whole.len() {
            let _ = halve_output(&whole[..cut]);
        }
        // A length far beyond the file.
        assert!(halve_output(&[0x3a, 0xff, 0xff, 0xff, 0xff, 0x0f]).is_none());
    }

    #[test]
    fn the_filter_keeps_brightness_and_is_symmetrical() {
        let t = taps();
        assert!((t.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for i in 0..TAPS / 2 {
            assert!((t[i] - t[TAPS - 1 - i]).abs() < 1e-7);
        }
        assert!(t[5] > 0.4 && t[5] == t[6]);
    }

    #[test]
    fn half_precision_is_rounded_correctly() {
        for (v, bits) in [
            (0.0f32, 0x0000u16),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (0.1, 0x2e66),
            (1.0e-9, 0x0000),
            (1.0e9, 0x7c00),
        ] {
            assert_eq!(half_bits(v), bits, "{v}");
        }
    }
}
