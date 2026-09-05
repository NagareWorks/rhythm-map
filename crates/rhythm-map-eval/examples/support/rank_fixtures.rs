//! Authored inputs only; labels never enter the clock scorer.
pub const N: usize = 1152;
pub const BASES: [&str; 14] = [
    "constant_intact",
    "constant_weak_alternating",
    "half_speed_intact",
    "double_speed_intact",
    "double_speed_weak_alternating",
    "non_octave_intact",
    "constant_all_weak",
    "constant_erased_beats",
    "constant_erased_beats_and_bars",
    "flat",
    "fixed_seed_noise",
    "flat_middle",
    "unavailable_gap",
    "all_unavailable",
];

#[derive(Clone)]
pub struct Fixture {
    pub name: String,
    pub authored_clock: &'static str,
    pub beat: Vec<f32>,
    pub bar: Vec<f32>,
    pub available: Vec<bool>,
}

pub fn base(name: &str) -> Fixture {
    let (period, clock) = match name {
        "half_speed_intact" => (48, "half"),
        "double_speed_intact" | "double_speed_weak_alternating" => (12, "double"),
        "non_octave_intact" => (32, "non_octave"),
        _ => (24, "constant"),
    };
    let mut f = Fixture {
        name: name.into(),
        authored_clock: clock,
        beat: vec![-8.; N],
        bar: vec![-8.; N],
        available: vec![true; N],
    };
    for (part, p) in [24, period, 24].into_iter().enumerate() {
        for (i, t) in (part * 384 + 4..(part + 1) * 384).step_by(p).enumerate() {
            let value = if name == "constant_all_weak" { -2. } else { 8. };
            f.beat[t - 1..=t + 1].fill(value);
            if i.is_multiple_of(4) {
                f.bar[t - 1..=t + 1].fill(value);
            }
        }
    }
    if name.ends_with("alternating") {
        for t in (388 + period..768).step_by(period * 2) {
            f.beat[t - 1..=t + 1].fill(-2.);
        }
    }
    if name.starts_with("constant_erased") {
        for t in (412..768).step_by(48) {
            f.beat[t - 1..=t + 1].fill(-8.);
        }
        if name.ends_with("and_bars") {
            for t in (484..768).step_by(192) {
                f.bar[t - 1..=t + 1].fill(-8.);
            }
        }
    }
    match name {
        "flat" => {
            f.beat.fill(-8.);
            f.bar.fill(-8.);
        }
        "flat_middle" => {
            f.beat[480..672].fill(-8.);
            f.bar[480..672].fill(-8.);
        }
        "unavailable_gap" => f.available[480..672].fill(false),
        "all_unavailable" => f.available.fill(false),
        "fixed_seed_noise" => {
            let mut seed = 0x1357_2468_u32;
            for values in [&mut f.beat, &mut f.bar] {
                for value in values {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *value = -8. + f32::from(u8::try_from(seed >> 24).unwrap()) / 255. * 6.;
                }
            }
        }
        _ => {}
    }
    f
}

pub fn all() -> Vec<Fixture> {
    let mut cases: Vec<_> = BASES.iter().map(|name| base(name)).collect();
    for name in [
        "constant_intact",
        "half_speed_intact",
        "double_speed_weak_alternating",
    ] {
        for intervention in ["tiny_contrast", "middle_offset"] {
            let mut f = base(name);
            f.name = format!("{name}_{intervention}");
            for values in [&mut f.beat, &mut f.bar] {
                match intervention {
                    "tiny_contrast" => values.iter_mut().for_each(|v| *v = -8. + (*v + 8.) / 4096.),
                    // One fixed stationarity challenge, not a fitted offset sweep.
                    _ => values[384..768].iter_mut().for_each(|v| *v -= 16.),
                }
            }
            cases.push(f);
        }
    }
    cases
}
