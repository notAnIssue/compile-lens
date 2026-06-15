//! Explicit string-sentinel encoding for non-finite `f64` in the schema.
//!
//! JSON has no NaN or Infinity literal, and serde_json silently serializes a non-finite `f64` to
//! `null`. For an `Option<f64>` that makes a non-finite value indistinguishable from "not
//! computed" — a real loss for a diagnostics schema, where a NaN `max_abs_diff` means "the compiled
//! model produced NaN at this layer", exactly the signal the divergence tool exists to surface.
//!
//! So non-finite values are encoded *explicitly* as the JSON strings `"NaN"`, `"Infinity"`, and
//! `"-Infinity"`, round-tripping losslessly. Finite values stay JSON numbers; `None` stays absent.
//! The policy is applied uniformly to every `Option<f64>` field via
//! `#[serde(with = "crate::non_finite::opt")]`, so the contract is one rule with no per-field
//! exceptions to get wrong. See the schema ADR on the non-finite policy.

/// serde adapter for `Option<f64>` fields: `#[serde(with = "crate::non_finite::opt")]`.
pub mod opt {
    use serde::de::{self, Deserializer};
    use serde::{Deserialize, Serializer};

    /// Finite → JSON number; NaN / +∞ / −∞ → string sentinel; `None` → `null` (and then skipped by
    /// the field's `skip_serializing_if`, so a minimal artifact stays minimal).
    pub fn serialize<S>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            None => serializer.serialize_none(),
            Some(x) if x.is_finite() => serializer.serialize_f64(*x),
            Some(x) if x.is_nan() => serializer.serialize_str("NaN"),
            Some(x) if x.is_sign_positive() => serializer.serialize_str("Infinity"),
            Some(_) => serializer.serialize_str("-Infinity"),
        }
    }

    /// JSON number → finite; the three sentinels → NaN / ±∞; `null` or absent → `None`. Any other
    /// string is a hard error rather than a silently dropped value.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // An untagged enum lets a present value be either a JSON number or a sentinel string;
        // wrapping it in `Option` lets `null`/absent map to `None`.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Num(f64),
            Str(String),
        }

        match Option::<Repr>::deserialize(deserializer)? {
            None => Ok(None),
            Some(Repr::Num(x)) => Ok(Some(x)),
            Some(Repr::Str(s)) => match s.as_str() {
                "NaN" => Ok(Some(f64::NAN)),
                "Infinity" => Ok(Some(f64::INFINITY)),
                "-Infinity" => Ok(Some(f64::NEG_INFINITY)),
                other => Err(de::Error::custom(format!(
                    "invalid f64 value {other:?} \
                     (expected a number, \"NaN\", \"Infinity\", or \"-Infinity\")"
                ))),
            },
        }
    }
}
