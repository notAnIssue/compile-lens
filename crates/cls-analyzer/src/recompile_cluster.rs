//! Guard-expression clustering for Tool 1 — the "describe → attribute" core (ADR-032).
//!
//! Each [`Recompilation`] carries the *primary* failed guard (the collector already picked the
//! failure against the frame's base compile). This module turns those guards into
//! [`GuardCategory`] clusters that **attribute** a recompile to a concrete dynamic axis rather
//! than merely bucketing by a coarse label:
//!
//! - the structured Mode-A guard text `tensor 'x' size mismatch at index 0` parses into a
//!   category (`size` / `dtype` / `stride`), a canonical template, and an **axis** (`x[0]`);
//! - recompiles are grouped by `(category, axis)`, so "dim 0 keeps going dynamic" is one
//!   cluster distinct from "dim 1 keeps going dynamic" — each becomes its own actionable axis;
//! - per cluster we keep the count and the distinct `previous -> new` value transitions.
//!
//! An unrecognized guard (e.g. a symbolic `s77 <= 1024` from Mode B, or a `self.x == 32` guard)
//! falls into the `other` category, canonicalized by replacing integer literals with `<int>` so
//! structurally-identical guards still group. A recompile with no guard expression is skipped.
//!
//! Why a hand parser and not a regex engine: the guard text is a fixed shape and the toolkit is
//! deliberately dependency-light (ADR-015); a few `str` ops are clearer than pulling in `regex`.

use cls_schema::Recompilation;
use indexmap::IndexMap;

use crate::recompile::{GuardCategory, ValueTransition};

/// The structured form extracted from one guard expression.
struct ParsedGuard {
    category: String,
    canonical: String,
    axis: Option<String>,
}

const CATEGORIES: [&str; 3] = ["size", "dtype", "stride"];

/// Parse a guard expression into `(category, canonical_template, axis)`.
///
/// Recognizes the Mode-A mismatch text `tensor '<name>' <cat> mismatch [at index <n>]`; anything
/// else becomes the `other` category with integer literals canonicalized to `<int>`.
fn parse_guard(expr: &str) -> ParsedGuard {
    if let Some(parsed) = parse_tensor_mismatch(expr) {
        return parsed;
    }
    ParsedGuard {
        category: "other".to_string(),
        canonical: canonicalize_literals(expr),
        axis: None,
    }
}

fn parse_tensor_mismatch(expr: &str) -> Option<ParsedGuard> {
    let rest = expr.strip_prefix("tensor '")?;
    let quote_end = rest.find('\'')?;
    let name = &rest[..quote_end];
    let tail = rest[quote_end + 1..].trim_start(); // e.g. "size mismatch at index 0"

    for category in CATEGORIES {
        let prefix = format!("{category} mismatch");
        let Some(after) = tail.strip_prefix(&prefix) else {
            continue;
        };
        let after = after.trim_start();
        if let Some(index_str) = after.strip_prefix("at index ") {
            let index: String = index_str.chars().take_while(char::is_ascii_digit).collect();
            if !index.is_empty() {
                return Some(ParsedGuard {
                    category: category.to_string(),
                    canonical: format!("tensor '<t>' {category} mismatch at index <n>"),
                    axis: Some(format!("{name}[{index}]")),
                });
            }
        }
        return Some(ParsedGuard {
            category: category.to_string(),
            canonical: format!("tensor '<t>' {category} mismatch"),
            axis: Some(name.to_string()),
        });
    }
    None
}

/// Replace each maximal run of ASCII digits with `<int>` so guards that differ only in their
/// literal values (`self.batch_size == 32` vs `== 24`) group together.
fn canonicalize_literals(expr: &str) -> String {
    let mut out = String::with_capacity(expr.len());
    let mut chars = expr.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            while chars.peek().is_some_and(char::is_ascii_digit) {
                chars.next();
            }
            out.push_str("<int>");
        } else {
            out.push(c);
        }
    }
    out
}

/// Mutable accumulator while grouping; converted to [`GuardCategory`] at the end.
struct Cluster {
    category: String,
    canonical: String,
    axis: Option<String>,
    count: u64,
    values: Vec<ValueTransition>,
}

/// Cluster a session's recompiles into guard categories.
///
/// Grouped by `(category, axis)` (falling back to the canonical template for `other` guards).
/// Output order is deterministic: by descending count, then canonical template, then axis — so
/// the same input always yields byte-identical findings (D7 determinism).
pub fn cluster(recompilations: &[Recompilation]) -> Vec<GuardCategory> {
    let mut groups: IndexMap<String, Cluster> = IndexMap::new();

    for recompile in recompilations {
        let Some(guard) = &recompile.failed_guard else {
            continue; // no guard to attribute — skip (still counted in total_recompilations)
        };
        let Some(expr) = guard.expression.as_deref() else {
            continue;
        };
        if expr.is_empty() {
            continue;
        }

        let parsed = parse_guard(expr);
        let key = match &parsed.axis {
            Some(axis) => format!("{}\u{1f}{axis}", parsed.category),
            None => format!("{}\u{1f}{}", parsed.category, parsed.canonical),
        };
        let cluster = groups.entry(key).or_insert_with(|| Cluster {
            category: parsed.category.clone(),
            canonical: parsed.canonical.clone(),
            axis: parsed.axis.clone(),
            count: 0,
            values: Vec::new(),
        });
        cluster.count += 1;

        let transition = ValueTransition {
            previous: guard.previous_value.clone(),
            new: guard.new_value.clone(),
        };
        if (transition.previous.is_some() || transition.new.is_some())
            && !cluster.values.contains(&transition)
        {
            cluster.values.push(transition);
        }
    }

    let mut out: Vec<GuardCategory> = groups
        .into_values()
        .map(|c| GuardCategory {
            category: c.category,
            count: c.count,
            canonical_guard: c.canonical,
            axis: c.axis,
            observed_values: c.values,
        })
        .collect();

    out.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.canonical_guard.cmp(&b.canonical_guard))
            .then_with(|| a.axis.cmp(&b.axis))
    });
    out
}
