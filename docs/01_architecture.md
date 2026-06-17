# Architecture

compile-lens is a two-language toolkit joined by one on-disk contract. **Python captures**
(it is the only side that touches `torch.compile`); **Rust analyzes and renders**. The two never
share memory — they hand off a versioned `.cls.json` file, so each side can evolve independently and
every run is reproducible from its artifact.

## Data flow

```mermaid
flowchart TB
    subgraph py["Python — capture (source of truth)"]
        sess["cl.session() context manager"]
        coll["collectors — torch.compile tracing"]
    end
    art[".cls.json<br/>versioned schema contract"]
    subgraph rust["Rust — analyze & render"]
        schema["cls-schema · cls-schema-migrate<br/>parse + version-gate"]
        subgraph analyzers["analyzers"]
            an["cls-analyzer<br/>recompile · cache-stability<br/>divergence · lint · fusion"]
            diff["cls-wl-diff<br/>Tool 2a graph diff"]
            roof["cls-roofline<br/>Tool 5 roofline"]
        end
        report["cls-report → self-contained HTML"]
        scrub["cls-scrub → share-safe artifact / report"]
    end
    cli["cl — CLI orchestrator"]

    sess --> coll --> art
    art --> schema
    schema --> analyzers
    analyzers --> report
    report --> scrub
    cli -.drives.-> schema
    cli -.drives.-> report
    cli -.drives.-> scrub
```

The control flow crosses the language boundary as a subprocess + file, never FFI (ADR-006): the
Python `cl.session()` front-end writes the artifact, and the `cl` binary reads it back to analyze or
render. `collect` is the one CLI subcommand left as a typed stub, because collection is driven from
Python.

## Schema (the `.cls.json` contract)

The artifact is a single `session` object plus parallel, normalized record arrays — records
cross-reference each other by `*_id` rather than nesting (ADR-021). This keeps each analysis result
independently addressable and the document flat.

```mermaid
erDiagram
    ClsArtifact ||--|| Session : has
    ClsArtifact ||--o{ Recompilation : recompilations
    ClsArtifact ||--o{ CompiledGraph : compiled_graphs
    ClsArtifact ||--o{ LintFinding : lint_findings
    ClsArtifact ||--o{ Divergence : divergences
    ClsArtifact ||--o{ Kernel : kernels
    ClsArtifact ||--o{ RooflinePrediction : roofline_predictions
    ClsArtifact ||--o{ FusionOpportunity : fusion_opportunities
    CompiledGraph ||--o{ FxNode : nodes
    CompiledGraph }o--o{ Kernel : kernel_ids
    Session ||--|| RedactionPolicy : redaction_policy
```

A minor schema bump requires a `cls-schema-migrate` function; the same input always serializes to
byte-identical output (D10), which is what lets the redaction corpus and the round-trip suite gate
on exact bytes.

## Roadmap

```mermaid
flowchart LR
    A["✅ Toolkit + Hero<br/>Tools 1–6 · cl.session() · cl scrub"]
    B["🚧 v0.5.0 release<br/>tag · PyPI · screencast"]
    C["📋 Agentic — Phase 8<br/>sandboxed MCP server"]
    D["📋 v1.0.0<br/>public API freeze"]
    A --> B --> C --> D
```

The `cls-context-builder` and `cls-mcp-server` crates are intentional empty stubs reserved for the
Phase 8 agentic layer; they are not part of the current toolkit.
