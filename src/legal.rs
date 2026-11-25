//! The notices an interactive program under the AGPL shows its users.

/// Who holds the copyright.
pub const COPYRIGHT: &str = "Copyright (C) 2026 the Chitra Enhancer AI authors";
/// Where the source of this program is.
pub const SOURCE: &str = env!("CARGO_PKG_REPOSITORY");
/// The terms, in short.
pub const TERMS: &str = "Free software: you may use, share and change it under the GNU \
Affero General Public License, version 3 or later. If you pass on a changed version, or \
let others use one over a network, you must offer them its source. It comes with NO \
WARRANTY, to the extent the law allows.";
/// What else is inside, and where its notices are.
pub const OTHERS: &str = "It is built with other open-source libraries and fonts; their \
licences are in THIRD_PARTY_LICENCES.txt in each download. The AI models are downloaded \
separately under their authors' licences (MODEL_LICENCES.md). FFmpeg and ONNX Runtime are \
separate programs that you install.";

/// The whole notice, for the command line.
pub fn notice() -> String {
    format!(
        "Chitra Enhancer AI {version}\n{COPYRIGHT}\n\n{TERMS}\n\n\
         Licence: {SOURCE}/blob/main/LICENSE\n\
         Source:  {SOURCE} (this version: tag v{version})\n\n{OTHERS}",
        version = env!("CARGO_PKG_VERSION"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_notice_says_what_the_licence_asks_for() {
        let n = notice();
        for part in ["Copyright", "Affero", "NO WARRANTY", "https://github.com/"] {
            assert!(n.contains(part), "{part} is missing from:\n{n}");
        }
        assert_eq!(env!("CARGO_PKG_LICENSE"), "AGPL-3.0-or-later");
    }
}
