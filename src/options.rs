//! Three ways to restore one video. Each is tried on a short sample on this
//! computer, so the user can look at the results, see how long the whole video
//! would take, and choose. The program cannot judge which looks best; it only
//! marks the one its measurements of the video point to.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::analyze::{Available, Suggestion};

/// One way of restoring the video.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// Short name, used in file names.
    pub key: String,
    pub title: String,
    /// What it does, in plain words.
    pub what: String,
    /// The one the video's measurements point to.
    pub recommended: bool,
    pub settings: Suggestion,
}

/// A choice after it was tried.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Offer {
    pub choice: Choice,
    /// Seconds the whole video would take, and frames per second; None if the
    /// try failed.
    pub speed: Option<(f64, f64)>,
    /// The short sample.
    pub clip: PathBuf,
    /// A still: the original on the left, this result on the right.
    pub picture: PathBuf,
    /// Why it could not be tried.
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub offers: Vec<Offer>,
    /// The middle of the same frame: original, then the three results.
    pub together: Option<PathBuf>,
}

/// The simple filter that suits this much noise (0..255 scale).
fn filter_for(noise: f32) -> &'static str {
    if noise < 2.5 {
        "light"
    } else if noise < 6.0 {
        "medium"
    } else {
        "strong"
    }
}

/// Three different choices for a video whose analysis suggested `base`.
pub fn choices(base: &Suggestion, noise: f32, have: Available) -> [Choice; 3] {
    let sigma = (noise * 1.6).clamp(3.0, 60.0).round();
    let usual = filter_for(noise);
    let enlarge = base.scale > 1 && have.upscaler;
    let make = |key: &str, title: &str, ai: bool, five: bool, filter: &str| {
        let grain = match (five, filter) {
            (true, _) if noise < 6.0 => 2,
            (true, _) => 3,
            (false, "light") => 0,
            _ => 2,
        };
        let enlarging = match (base.scale, ai) {
            (1, _) => "Same size".to_string(),
            (n, true) => format!("Enlarged {n}x by the AI upscaler"),
            (n, false) => format!("Enlarged {n}x without AI"),
        };
        let cleaning = if five {
            "cleaned by the five-frame AI denoiser".to_string()
        } else {
            format!("{filter} simple cleaning")
        };
        Choice {
            key: key.into(),
            title: title.into(),
            what: format!("{enlarging}, {cleaning}."),
            recommended: false,
            settings: Suggestion {
                ai_upscaler: ai,
                five_frame: five,
                denoise: if five { "off" } else { filter }.into(),
                sigma: five.then_some(sigma),
                grain,
                ..base.clone()
            },
        }
    };
    let (mut three, best) = match (enlarge, have.denoiser) {
        (true, true) => (
            [
                make("thorough", "Most thorough", true, true, usual),
                make("balanced", "Balanced", true, false, usual),
                make("quick", "Quick", false, false, usual),
            ],
            if base.five_frame { 0 } else { 1 },
        ),
        (true, false) => {
            let other = match usual {
                "light" => "medium",
                "medium" => "strong",
                _ => "medium",
            };
            (
                [
                    make("balanced", "Balanced", true, false, usual),
                    make("other-cleaning", "Different cleaning", true, false, other),
                    make("quick", "Quick", false, false, usual),
                ],
                0,
            )
        }
        (false, true) => (
            [
                make("thorough", "Most thorough", false, true, usual),
                make("gentle", "Gentle", false, false, "light"),
                make(
                    "firm",
                    "Firmer",
                    false,
                    false,
                    if usual == "strong" {
                        "strong"
                    } else {
                        "medium"
                    },
                ),
            ],
            if base.five_frame { 0 } else { 1 },
        ),
        (false, false) => (
            [
                make("gentle", "Gentle", false, false, "light"),
                make("medium", "Medium", false, false, "medium"),
                make("firm", "Firm", false, false, "strong"),
            ],
            match usual {
                "light" => 0,
                "medium" => 1,
                _ => 2,
            },
        ),
    };
    three[best].recommended = true;
    three
}

/// "about 45 s", "about 6 min", "about 1 h 20 min".
pub fn roughly(secs: f64) -> String {
    let secs = Duration::try_from_secs_f64(secs.max(0.0))
        .unwrap_or_default()
        .as_secs();
    match secs {
        0..90 => format!("about {secs} s"),
        90..3600 => format!("about {} min", (secs + 30) / 60),
        _ => {
            let mins = (secs + 30) / 60;
            format!("about {} h {:02} min", mins / 60, mins % 60)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(scale: usize, five: bool) -> Suggestion {
        Suggestion {
            deinterlace: true,
            double_rate: false,
            denoise: "light".into(),
            five_frame: five,
            sigma: None,
            scale,
            ai_upscaler: scale > 1,
            grain: 0,
            crf: 18,
            crop: Some([10, 6, 0, 0]),
            colour_align: Some((2, 3)),
            dehalo: 40,
        }
    }

    #[test]
    fn there_are_always_three_different_choices_and_one_is_recommended() {
        for scale in [1, 2] {
            for noise in [1.0, 4.0, 12.0] {
                for upscaler in [false, true] {
                    for denoiser in [false, true] {
                        let have = Available { upscaler, denoiser };
                        let b = base(scale, denoiser && noise >= 2.5);
                        let c = choices(&b, noise, have);
                        let ctx = format!("{scale} {noise} {upscaler} {denoiser}");
                        assert_ne!(c[0].settings, c[1].settings, "{ctx}");
                        assert_ne!(c[0].settings, c[2].settings, "{ctx}");
                        assert_ne!(c[1].settings, c[2].settings, "{ctx}");
                        assert_ne!(c[0].key, c[1].key, "{ctx}");
                        assert_ne!(c[1].key, c[2].key, "{ctx}");
                        assert_eq!(c.iter().filter(|c| c.recommended).count(), 1, "{ctx}");
                        for c in &c {
                            // Never a model that is not there.
                            assert!(upscaler || !c.settings.ai_upscaler, "{ctx}");
                            assert!(denoiser || !c.settings.five_frame, "{ctx}");
                            assert!(scale > 1 || !c.settings.ai_upscaler, "{ctx}");
                            assert_eq!(c.settings.five_frame, c.settings.sigma.is_some(), "{ctx}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_repairs_the_analysis_found_are_in_every_choice() {
        let have = Available {
            upscaler: true,
            denoiser: true,
        };
        for c in choices(&base(2, false), 1.4, have) {
            assert_eq!(c.settings.crop, Some([10, 6, 0, 0]));
            assert_eq!(c.settings.colour_align, Some((2, 3)));
            assert_eq!(c.settings.dehalo, 40);
        }
    }

    #[test]
    fn a_quiet_tape_recommends_the_gentler_choice_and_a_noisy_one_the_thorough() {
        let have = Available {
            upscaler: true,
            denoiser: true,
        };
        let quiet = choices(&base(2, false), 1.4, have);
        assert!(quiet[1].recommended && quiet[1].settings.denoise == "light");
        let noisy = choices(&base(2, true), 9.0, have);
        assert!(noisy[0].recommended && noisy[0].settings.sigma == Some(14.0));
    }

    #[test]
    fn times_are_said_roughly() {
        assert_eq!(roughly(45.4), "about 45 s");
        assert_eq!(roughly(366.0), "about 6 min");
        assert_eq!(roughly(4790.0), "about 1 h 20 min");
        assert_eq!(roughly(f64::NAN), "about 0 s");
    }
}
