//! Non-finite `f64` policy: NaN / ±∞ are encoded as explicit string sentinels, never silently
//! turned into `null`. Exercised through `KernelMeasurements` (all-optional f64 fields), which uses
//! the `#[serde(with = "crate::non_finite::opt")]` adapter like every other `Option<f64>` in the
//! schema.

use cls_schema::KernelMeasurements;

fn measurements(mean: Option<f64>, median: Option<f64>, p99: Option<f64>) -> KernelMeasurements {
    KernelMeasurements {
        iterations: None,
        mean_us: mean,
        median_us: median,
        p99_us: p99,
        source: None,
    }
}

#[test]
fn sentinels_deserialize_to_non_finite() {
    let m: KernelMeasurements =
        serde_json::from_str(r#"{"mean_us":"NaN","median_us":"Infinity","p99_us":"-Infinity"}"#)
            .unwrap();
    assert!(m.mean_us.unwrap().is_nan());
    assert_eq!(m.median_us.unwrap(), f64::INFINITY);
    assert_eq!(m.p99_us.unwrap(), f64::NEG_INFINITY);
}

#[test]
fn non_finite_serializes_to_sentinels_not_null() {
    let out = serde_json::to_string(&measurements(
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
    ))
    .unwrap();
    assert!(out.contains(r#""mean_us":"NaN""#), "{out}");
    assert!(out.contains(r#""median_us":"Infinity""#), "{out}");
    assert!(out.contains(r#""p99_us":"-Infinity""#), "{out}");
    assert!(
        !out.contains("null"),
        "non-finite must not become null: {out}"
    );
}

#[test]
fn nan_round_trips_losslessly() {
    let out = serde_json::to_string(&measurements(Some(f64::NAN), None, None)).unwrap();
    let back: KernelMeasurements = serde_json::from_str(&out).unwrap();
    assert!(back.mean_us.unwrap().is_nan()); // NaN survived; not flattened to null/None
}

#[test]
fn finite_values_stay_plain_numbers() {
    let out = serde_json::to_string(&measurements(Some(12.5), None, Some(40.0))).unwrap();
    assert!(out.contains(r#""mean_us":12.5"#), "{out}"); // a JSON number, not a "12.5" string
    let back: KernelMeasurements = serde_json::from_str(&out).unwrap();
    assert_eq!(back.mean_us, Some(12.5));
    assert_eq!(back.p99_us, Some(40.0));
}

#[test]
fn none_is_skipped_not_null() {
    let out = serde_json::to_string(&measurements(None, None, None)).unwrap();
    assert!(
        !out.contains("mean_us"),
        "None must be skipped, not null: {out}"
    );
}

#[test]
fn an_unknown_string_is_a_hard_error() {
    // A non-sentinel string is a real error, not a silently dropped value.
    let err = serde_json::from_str::<KernelMeasurements>(r#"{"mean_us":"bogus"}"#).unwrap_err();
    assert!(err.to_string().contains("bogus"), "{err}");
}
