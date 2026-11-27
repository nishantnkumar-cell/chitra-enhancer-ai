//! What the service lets in: which computers may connect, which key they must
//! show, and which files it will look at. Nothing here touches the network.

use std::net::IpAddr;

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// The shortest key the service will start with.
pub const SHORTEST_KEY: usize = 24;

/// This computer, or one on a Tailscale network (the addresses Tailscale hands
/// out: 100.64.0.0/10 and fd7a:115c:a1e0::/48). The service listens on and
/// answers nothing else.
pub fn allowed_address(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(a) => {
            let [first, second, ..] = a.octets();
            a.is_loopback() || (first == 100 && (64..128).contains(&second))
        }
        IpAddr::V6(a) => match a.to_ipv4_mapped() {
            Some(inner) => allowed_address(IpAddr::V4(inner)),
            None => a.is_loopback() || a.segments()[..3] == [0xfd7a, 0x115c, 0xa1e0],
        },
    }
}

/// The key everyone must send, kept only as its digest.
#[derive(Clone)]
pub struct Key([u8; 32]);

impl Key {
    pub fn new(key: &str) -> Result<Key, String> {
        if key.len() < SHORTEST_KEY || !key.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(format!(
                "the key must be at least {SHORTEST_KEY} letters, digits or signs without spaces (make one with `chitra-enhance new-key`)"
            ));
        }
        Ok(Key(Sha256::digest(key.as_bytes()).into()))
    }

    /// Whether an `Authorization` header carries this key. Takes the same time
    /// however much of a wrong key matches.
    pub fn opens(&self, header: Option<&str>) -> bool {
        let Some(given) = header.and_then(|h| h.strip_prefix("Bearer ")) else {
            return false;
        };
        let given: [u8; 32] = Sha256::digest(given.as_bytes()).into();
        self.0.ct_eq(&given).into()
    }
}

/// A new random key: 32 bytes as 64 hexadecimal digits.
pub fn new_key() -> anyhow::Result<String> {
    Ok(random_hex::<32>()?)
}

/// `N` random bytes in hexadecimal.
pub fn random_hex<const N: usize>() -> Result<String, std::io::Error> {
    let mut bytes = [0u8; N];
    getrandom::fill(&mut bytes).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// A job number as `random_hex::<16>` makes them. Nothing else is looked up or
/// removed.
pub fn is_job_id(s: &str) -> bool {
    s.len() == 32 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The kind of picture `start` begins, by its first bytes.
pub fn photo_kind(start: &[u8]) -> Option<&'static str> {
    match start {
        [0xff, 0xd8, 0xff, ..] => Some("jpg"),
        [0x89, b'P', b'N', b'G', ..] => Some("png"),
        [b'I', b'I', 42, 0, ..] | [b'M', b'M', 0, 42, ..] => Some("tif"),
        _ if start.len() >= 12 && &start[..4] == b"RIFF" && &start[8..12] == b"WEBP" => {
            Some("webp")
        }
        _ => None,
    }
}

/// The kind of video `start` begins. Only real containers are let through:
/// ffmpeg would also open play lists and scripts, which can name other files on
/// this computer.
pub fn video_kind(start: &[u8]) -> Option<&'static str> {
    match start {
        [0x1a, 0x45, 0xdf, 0xa3, ..] => Some("mkv"),
        [0, 0, 1, 0xba, ..] | [0, 0, 1, 0xb3, ..] => Some("mpg"),
        _ if start.len() >= 12 && &start[4..8] == b"ftyp" => Some("mp4"),
        _ if start.len() >= 12 && &start[..4] == b"RIFF" && &start[8..12] == b"AVI " => Some("avi"),
        // Transport stream: a packet starts every 188 bytes.
        _ if start.len() > 376 && [0, 188, 376].iter().all(|&i| start[i] == 0x47) => Some("ts"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_this_computer_and_tailscale_addresses_are_allowed() {
        for yes in [
            "127.0.0.1",
            "::1",
            "100.64.0.1",
            "100.101.102.103",
            "100.127.255.254",
            "fd7a:115c:a1e0::1",
            "::ffff:100.100.1.1",
        ] {
            assert!(allowed_address(yes.parse().unwrap()), "{yes}");
        }
        for no in [
            "0.0.0.0",
            "::",
            "192.168.1.10",
            "10.0.0.5",
            "100.63.255.255",
            "100.128.0.0",
            "8.8.8.8",
            "fd7a:115c:a1e1::1",
            "fe80::1",
            "::ffff:192.168.1.10",
        ] {
            assert!(!allowed_address(no.parse().unwrap()), "{no}");
        }
    }

    #[test]
    fn the_right_key_opens_and_nothing_else_does() {
        let k = new_key().unwrap();
        assert_eq!(k.len(), 64);
        assert_ne!(k, new_key().unwrap());
        let key = Key::new(&k).unwrap();
        assert!(key.opens(Some(&format!("Bearer {k}"))));
        for wrong in [
            None,
            Some(String::new()),
            Some(k.clone()),
            Some("Bearer ".into()),
            Some(format!("Bearer {}", &k[..63])),
            Some(format!("Bearer {k}x")),
            Some(format!("bearer {k}")),
        ] {
            assert!(!key.opens(wrong.as_deref()), "{wrong:?}");
        }
        for weak in [
            "",
            "short",
            "twenty three characters",
            "has a space in it, long enough",
        ] {
            assert!(Key::new(weak).is_err(), "{weak}");
        }
    }

    #[test]
    fn job_numbers_cannot_name_other_things() {
        assert!(is_job_id(&random_hex::<16>().unwrap()));
        for no in [
            "",
            "..",
            "../../etc",
            "0123456789ABCDEF0123456789ABCDEF",
            "0123456789abcdef0123456789abcde",
            "0123456789abcdef0123456789abcdeg",
        ] {
            assert!(!is_job_id(no), "{no}");
        }
    }

    #[test]
    fn files_are_known_by_their_first_bytes() {
        assert_eq!(photo_kind(&[0xff, 0xd8, 0xff, 0xe0]), Some("jpg"));
        assert_eq!(photo_kind(b"\x89PNG\r\n\x1a\n"), Some("png"));
        assert_eq!(photo_kind(b"RIFF\0\0\0\0WEBPVP8 "), Some("webp"));
        assert_eq!(photo_kind(b"MM\0*\0\0\0\x08"), Some("tif"));
        assert_eq!(video_kind(b"\0\0\0\x20ftypisom"), Some("mp4"));
        assert_eq!(video_kind(&[0x1a, 0x45, 0xdf, 0xa3]), Some("mkv"));
        assert_eq!(video_kind(&[0, 0, 1, 0xba, 0x44]), Some("mpg"));
        assert_eq!(video_kind(b"RIFF\0\0\0\0AVI LIST"), Some("avi"));
        let mut ts = vec![0u8; 400];
        (ts[0], ts[188], ts[376]) = (0x47, 0x47, 0x47);
        assert_eq!(video_kind(&ts), Some("ts"));
        // Files that make ffmpeg read other files, and plain junk.
        for no in [
            &b"#EXTM3U\n#EXTINF:1,\nfile:///etc/passwd\n"[..],
            b"ffconcat version 1.0\nfile /etc/passwd\n",
            b"<?xml version=\"1.0\"?><MPD></MPD>",
            b"[playlist]\nFile1=x",
            b"hello",
            b"",
        ] {
            assert_eq!(video_kind(no), None, "{no:?}");
            assert_eq!(photo_kind(no), None);
        }
    }
}
