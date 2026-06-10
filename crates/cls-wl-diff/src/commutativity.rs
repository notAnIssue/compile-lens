//! Which operations are commutative in their operands (ADR-025).
//!
//! The WL-signature diff (Tool 2a) must know, for each node, whether reordering its operands
//! changes the result. For a commutative op the diff sorts the operand signatures so that
//! `add(a, b)` and `add(b, a)` hash the same (a reorder is a no-op and must stay silent); for a
//! non-commutative op it preserves operand order so that `sub(a, b)` and `sub(b, a)` hash
//! differently (a reorder is a real change and must be reported). Misjudging this is a silent
//! correctness bug, so the policy is deliberately small, hardcoded, and conservative.
//!
//! **Conservative default: an op is non-commutative unless it is explicitly on the list.** This
//! makes the failure modes asymmetric, which is the point. Wrongly *omitting* a commutative op
//! makes a pure operand reorder look like a modification — noisy, but safe and visible. Wrongly
//! *including* a non-commutative op would sort its operands and hide a real change — a silent
//! miss. So the list contains only operations whose commutativity is unambiguous, and anything
//! unknown falls through to "non-commutative", which can only ever over-report.
//!
//! The set lives here in Rust rather than being queried from PyTorch because the diff is a pure
//! function with no torch dependency (it runs over `.cls.json` artifacts long after capture), and
//! because aten exposes no reliable per-op commutativity attribute to query anyway — the list
//! would be hand-maintained either way. See ADR-025 for the full rationale.

/// The hardcoded set of commutative operation names, **sorted** so [`CommutativitySet::is_commutative`]
/// can binary-search it without allocating. Bare op names (no `aten.` namespace, no overload
/// suffix); [`op_name`] normalizes an incoming `op_type` to this form before lookup.
///
/// Keep this sorted — `test_set_is_sorted` enforces it, because an out-of-order entry would make
/// the binary search silently miss and turn a commutative op non-commutative.
const COMMUTATIVE_OPS: &[&str] = &[
    "add",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "eq",
    "logical_and",
    "logical_or",
    "maximum",
    "minimum",
    "mul",
    "ne",
];

/// Namespace segments stripped from an `op_type` before the operation name is read. An FX node's
/// op_type can arrive as `add`, `aten.add`, or `torch.ops.aten.add.Tensor`; these are the leading
/// segments that are *not* the operation itself.
const NAMESPACES: &[&str] = &["torch", "ops", "aten", "prims", "_operator"];

/// Extract the bare operation name from an `op_type` string.
///
/// Splits on `.` and returns the first segment that is neither empty nor a known namespace token,
/// which is the operation name; any trailing overload tag (such as `.Tensor` or `.default`) is
/// ignored. Returns `None` only when there is no such segment (an empty or all-namespace string).
///
/// - `aten.add` and `torch.ops.aten.add.Tensor` and `add` all yield `add`.
/// - `aten.sub` yields `sub`; `placeholder` yields `placeholder`.
/// - The in-place `add_` yields `add_`, which is intentionally *not* `add`, so it stays
///   non-commutative under the conservative default.
fn op_name(op_type: &str) -> Option<&str> {
    op_type
        .split('.')
        .find(|seg| !seg.is_empty() && !NAMESPACES.contains(seg))
}

/// The commutativity policy threaded into the WL-signature computation.
///
/// It is a thin handle over a sorted list of commutative op names. The diff takes it by reference
/// so the policy is injected rather than hardcoded into the algorithm — [`CommutativitySet::standard`]
/// is the production set, and tests can build narrower ones to exercise the lookup in isolation.
#[derive(Debug, Clone, Copy)]
pub struct CommutativitySet {
    ops: &'static [&'static str],
}

impl CommutativitySet {
    /// The production commutativity policy: the hardcoded conservative set (ADR-025).
    pub const fn standard() -> Self {
        Self {
            ops: COMMUTATIVE_OPS,
        }
    }

    /// Whether `op_type`'s operation is commutative in its operands.
    ///
    /// Normalizes `op_type` to a bare op name (dropping namespace and overload), then looks it up.
    /// Anything not found — including unrecognized ops, `placeholder`, `output`, and malformed
    /// strings — is treated as **non-commutative** (the conservative default).
    pub fn is_commutative(&self, op_type: &str) -> bool {
        op_name(op_type).is_some_and(|name| self.ops.binary_search(&name).is_ok())
    }

    /// The commutative op names this policy recognizes. Exposed for diagnostics and tests.
    pub fn ops(&self) -> &'static [&'static str] {
        self.ops
    }
}

impl Default for CommutativitySet {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The set must stay sorted or the binary search silently misses (turning a commutative op
    /// non-commutative — exactly the silent bug this module exists to prevent).
    #[test]
    fn test_set_is_sorted() {
        let mut sorted = COMMUTATIVE_OPS.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            COMMUTATIVE_OPS,
            sorted.as_slice(),
            "COMMUTATIVE_OPS must be sorted"
        );
    }

    /// At least ten operations are recognized as commutative, across `aten.` and bare forms.
    #[test]
    fn test_commutative_ops_recognized() {
        let set = CommutativitySet::standard();
        let commutative = [
            "aten.add",
            "aten.mul",
            "aten.bitwise_and",
            "aten.bitwise_or",
            "aten.bitwise_xor",
            "aten.logical_and",
            "aten.logical_or",
            "aten.minimum",
            "aten.maximum",
            "aten.eq",
            "ne", // bare form also resolves
        ];
        assert!(commutative.len() >= 10);
        for op in commutative {
            assert!(set.is_commutative(op), "{op} should be commutative");
        }
    }

    /// At least ten operations fall through to non-commutative, including the operand-order
    /// danger ops the diff must never silence.
    #[test]
    fn test_non_commutative_ops_default_safe() {
        let set = CommutativitySet::standard();
        let non_commutative = [
            "aten.sub",
            "aten.div",
            "aten.pow",
            "aten.matmul",
            "aten.mm",
            "aten.lt",
            "aten.gt",
            "aten.remainder",
            "aten.index_select",
            "aten.gather",
            "aten.add_", // in-place is not the same op name as add
        ];
        assert!(non_commutative.len() >= 10);
        for op in non_commutative {
            assert!(!set.is_commutative(op), "{op} should be non-commutative");
        }
    }

    /// Namespace prefixes and overload suffixes are normalized away before lookup.
    #[test]
    fn test_op_name_normalization() {
        assert_eq!(op_name("aten.add"), Some("add"));
        assert_eq!(op_name("torch.ops.aten.add.Tensor"), Some("add"));
        assert_eq!(op_name("add.Scalar"), Some("add"));
        assert_eq!(op_name("placeholder"), Some("placeholder"));
        assert_eq!(op_name(""), None);

        let set = CommutativitySet::standard();
        // The overload-suffixed and fully-qualified forms still resolve to commutative.
        assert!(set.is_commutative("torch.ops.aten.add.Tensor"));
        // An all-namespace string has no op name, so it is non-commutative (never panics).
        assert!(!set.is_commutative("aten"));
    }

    /// Unknown / never-seen ops are non-commutative, never silently commutative.
    #[test]
    fn test_unknown_op_is_conservative() {
        let set = CommutativitySet::standard();
        assert!(!set.is_commutative("aten.some_future_op_we_have_never_heard_of"));
        assert!(!set.is_commutative("output"));
    }
}
