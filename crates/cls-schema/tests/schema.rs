//! Integration tests for `cls-schema`.
//!
//! These exercise the crate through its public API exactly as a downstream analyzer would,
//! and they deserialize the *canonical* example artifacts shipped in `schema/examples/` so
//! the Rust bindings and the JSON Schema can never silently drift apart.

use cls_schema::{ClsArtifact, RedactionPolicy, Session, SCHEMA_VERSION};

/// Canonical minimal artifact (only the required fields). Source of truth = JSON Schema
/// examples; included at compile time so a schema edit that breaks the bindings fails the
/// build.
const MINIMAL: &str = include_str!("../../../schema/examples/minimal.cls.json");
/// Canonical fully-populated artifact (every field exercised at least once).
const FULL: &str = include_str!("../../../schema/examples/full.cls.json");

fn parse(json: &str) -> ClsArtifact {
    serde_json::from_str(json).expect("example artifact should deserialize")
}

// ── test_minimal_example_deserialize ────────────────────────────────────────────────────
#[test]
fn test_minimal_example_deserialize() {
    let artifact = parse(MINIMAL);

    assert_eq!(artifact.schema_version, SCHEMA_VERSION);
    assert_eq!(artifact.session.id, "00000000-0000-4000-8000-000000000000");
    assert_eq!(artifact.session.torch_version, "2.6.0");
    assert_eq!(
        artifact.session.redaction_policy,
        RedactionPolicy::DefaultStrict
    );

    // Everything optional is absent, and the record arrays default to empty (not null).
    assert!(artifact.session.compile_config.is_none());
    assert!(artifact.session.env_snapshot.is_none());
    assert!(artifact.session.triton_version.is_none());
    assert!(artifact.graph_breaks.is_empty());
    assert!(artifact.kernels.is_empty());
    assert!(artifact.session.extra.is_empty());
}

// ── test_full_example_deserialize ───────────────────────────────────────────────────────
#[test]
fn test_full_example_deserialize() {
    let artifact = parse(FULL);

    let session = &artifact.session;
    assert_eq!(session.gpu_name.as_deref(), Some("NVIDIA H100 80GB HBM3"));
    assert_eq!(session.duration_ms, Some(45_230));
    assert_eq!(session.world_size, Some(1));
    assert_eq!(
        session
            .collector_versions
            .get("recompile")
            .map(String::as_str),
        Some("0.5.0")
    );

    // Nested [T, null] union field: present and explicitly null.
    let compile_config = session
        .compile_config
        .as_ref()
        .expect("compile_config present");
    assert_eq!(compile_config.fullgraph, Some(false));
    assert_eq!(compile_config.dynamic, None);
    assert_eq!(compile_config.backend.as_deref(), Some("inductor"));

    let env = session.env_snapshot.as_ref().expect("env_snapshot present");
    assert_eq!(env.random_seed, Some(42));
    assert_eq!(
        env.relevant_env_vars.get("TORCH_LOGS").map(String::as_str),
        Some("+recompiles")
    );

    // One record from each top-level array is present.
    assert_eq!(artifact.graph_breaks.len(), 1);
    assert_eq!(artifact.recompilations.len(), 1);
    assert_eq!(artifact.compiled_graphs[0].kernel_ids, vec!["k_001"]);
    assert_eq!(artifact.compile_phases.len(), 2);
    assert_eq!(artifact.iterations[0].cache_hit, Some(true));
    assert_eq!(
        artifact.lint_findings[0].pattern_category,
        "in_place_op_on_alias"
    );
    assert_eq!(
        artifact.roofline_predictions[0].bound_type.as_deref(),
        Some("memory")
    );

    // number-typed feature arrives as f64 even though it's written as an integer literal.
    let flops = artifact.kernels[0]
        .features
        .as_ref()
        .unwrap()
        .flops
        .unwrap();
    assert_eq!(flops, 1.2e10);
}

// ── test_kernel_features_num_regs_present ───────────────────────────────────────────────
#[test]
fn test_kernel_features_num_regs_present() {
    let artifact = parse(FULL);
    let features = artifact.kernels[0]
        .features
        .as_ref()
        .expect("kernel features present");
    // Register pressure proxy must survive deserialization (Tool 5's 4th correction).
    assert_eq!(features.num_regs, Some(168));
    assert_eq!(features.n_spills, Some(0));
}

// ── test_fx_nodes_parse_with_ordered_inputs ─────────────────────────────────────────────
#[test]
fn test_fx_nodes_parse_with_ordered_inputs() {
    let artifact = parse(FULL);
    let nodes = &artifact.compiled_graphs[0].nodes;
    assert_eq!(nodes.len(), 4);

    // A required-only node: no inputs, no attrs.
    assert_eq!(nodes[0].id, "n0");
    assert_eq!(nodes[0].op_type, "placeholder");
    assert!(nodes[0].inputs.is_empty());
    assert!(nodes[0].attrs.is_empty());

    // The sub node carries inputs in load-bearing order (n0 before n1) plus an attr.
    let sub = &nodes[2];
    assert_eq!(sub.op_type, "aten.sub");
    assert_eq!(sub.inputs, vec!["n0", "n1"]);
    assert_eq!(sub.attrs.get("alpha").and_then(|v| v.as_i64()), Some(1));

    // Operand order is preserved as a sequence, not normalized to a set: reversing it is a
    // genuinely different node, which is exactly what the WL diff relies on.
    assert_ne!(sub.inputs, vec!["n1", "n0"]);
}

// ── test_fx_nodes_absent_when_not_captured ──────────────────────────────────────────────
#[test]
fn test_fx_nodes_absent_when_not_captured() {
    // A graph written before node capture (the minimal artifact has none) parses fine: the
    // field is optional and defaults to empty rather than erroring.
    let artifact = parse(MINIMAL);
    assert!(artifact.compiled_graphs.is_empty());

    // And an empty `nodes` is omitted on re-serialize (skip_serializing_if), so a Tool 1
    // artifact does not grow a spurious `"nodes":[]`.
    let graph = cls_schema::CompiledGraph {
        graph_id: "g".into(),
        compiled_function_id: None,
        fx_graph_path: None,
        inductor_ir_path: None,
        guard_list: Vec::new(),
        kernel_ids: Vec::new(),
        compile_phases_summary: Default::default(),
        nodes: Vec::new(),
    };
    let json = serde_json::to_string(&graph).unwrap();
    assert!(!json.contains("nodes"));
}

// ── test_session_round_trip_serde ───────────────────────────────────────────────────────
#[test]
fn test_session_round_trip_serde() {
    let original = parse(FULL);
    let json = serde_json::to_string(&original).expect("serialize");
    let reparsed: ClsArtifact = serde_json::from_str(&json).expect("re-deserialize");
    // Structural equality across a full serialize -> deserialize cycle.
    assert_eq!(original, reparsed);

    // Minimal must also round-trip without acquiring extra fields.
    let minimal = parse(MINIMAL);
    let minimal_json = serde_json::to_string(&minimal).expect("serialize minimal");
    assert_eq!(
        minimal,
        serde_json::from_str(&minimal_json).expect("re-deserialize minimal")
    );
}

// ── test_redaction_policy_enum_serialize ────────────────────────────────────────────────
#[test]
fn test_redaction_policy_enum_serialize() {
    let cases = [
        (RedactionPolicy::DefaultStrict, "\"default-strict\""),
        (RedactionPolicy::Internal, "\"internal\""),
        (RedactionPolicy::Confidential, "\"confidential\""),
        (RedactionPolicy::PublicSafe, "\"public-safe\""),
    ];
    for (policy, wire) in cases {
        assert_eq!(serde_json::to_string(&policy).unwrap(), wire);
        assert_eq!(
            serde_json::from_str::<RedactionPolicy>(wire).unwrap(),
            policy
        );
    }
    // Fail closed: an unknown policy is a hard error, never a silent permissive default.
    assert!(serde_json::from_str::<RedactionPolicy>("\"wide-open\"").is_err());
}

// ── test_unknown_field_handled ──────────────────────────────────────────────────────────
#[test]
fn test_unknown_field_handled() {
    // Inject unknown keys at the top level, the session level, and inside a nested record.
    let mut value: serde_json::Value = serde_json::from_str(FULL).unwrap();
    value["future_top_level_array"] = serde_json::json!([{ "x": 1 }]);
    value["session"]["gpu_arch"] = serde_json::json!("hopper");
    value["graph_breaks"][0]["future_break_field"] = serde_json::json!("anything");
    let json = serde_json::to_string(&value).unwrap();

    let artifact: ClsArtifact = serde_json::from_str(&json).expect("unknown keys must not error");

    // Growth points capture unknown keys so they survive a round trip (D6).
    assert!(artifact.extra.contains_key("future_top_level_array"));
    assert_eq!(
        artifact
            .session
            .extra
            .get("gpu_arch")
            .and_then(|v| v.as_str()),
        Some("hopper")
    );

    // Deeper records accept the unknown key (no error) but do not preserve it.
    let reserialized = serde_json::to_string(&artifact).unwrap();
    assert!(reserialized.contains("future_top_level_array"));
    assert!(reserialized.contains("gpu_arch"));
    assert!(!reserialized.contains("future_break_field"));
}

// ── extra: constructor + missing-required-field behavior ────────────────────────────────
#[test]
fn test_new_stamps_current_version() {
    let session = parse(MINIMAL).session;
    let artifact = ClsArtifact::new(session);
    assert_eq!(artifact.schema_version, SCHEMA_VERSION);
    assert!(artifact.kernels.is_empty());
}

#[test]
fn test_missing_required_field_fails() {
    // Session without the required `redaction_policy` must fail to deserialize.
    let bad = r#"{ "schema_version": "0.5.0",
                   "session": { "id": "x", "timestamp": "t", "torch_version": "2.6.0" } }"#;
    assert!(serde_json::from_str::<ClsArtifact>(bad).is_err());

    // A bare Session missing `torch_version` likewise fails.
    let bad_session = r#"{ "id": "x", "timestamp": "t", "redaction_policy": "internal" }"#;
    assert!(serde_json::from_str::<Session>(bad_session).is_err());
}
