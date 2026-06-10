"""``CompileArtifactCollector`` — capture a ``torch.compile`` graph into a ``.cls.json``.

Tool 2a (compile-diff) needs the node-level FX structure of a compiled graph so the Rust
analyzer can diff two compiles. This collector hooks ``torch.compile`` to capture the
**aten-normalized** FX graph and serializes it into the inline ``compiled_graphs[].nodes[]``
contract (ADR-024): each node carries its ``id``, ``op_type``, ordered ``inputs`` (upstream
node ids), and scalar ``attrs``.

Why the aten graph rather than the dynamo graph: the dynamo-level graph's targets are Python
builtins (``<built-in method sub>``), which are version-noisy and not canonical. Routing the
capture through ``aot_autograd`` yields canonical aten ops (``aten.sub.Tensor``), which is what
the diff's commutativity normalization and signatures expect.

**Scope (single-iteration).** This captures the graph structure of one compile. Per-iteration
runtime behaviour (cache hits, recompilation, output signatures) is the multi-iteration capture
that builds on this. Inductor IR and Triton-kernel capture are deliberately out of scope here:
they require an actual Inductor/GPU compile, whereas the node-level structure the diff consumes
is exactly the aten FX graph captured on CPU.

``torch`` is an optional dependency (the user brings their own), so it is imported lazily inside
the capture path; constructing the collector and finalizing an artifact never import torch.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from compile_lens._schema import (
    SCHEMA_VERSION,
    ClsArtifact,
    CompiledGraph,
    FxNode,
    RedactionPolicy,
    Session,
    to_json,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_log = structlog.get_logger("compile_lens.collectors.artifact")


# ── FX graph -> nodes[] serialization ──────────────────────────────────────────────────
# These run only while torch is present (called from the capture backend), so torch / torch.fx
# are imported lazily by the caller and the Node type is duck-typed here via ``op``/``args``.


def _format_target(target: Any) -> str:
    """The op_type string for a ``call_function`` target.

    An aten ``OpOverload`` stringifies to a clean canonical name (``aten.sub.Tensor``). A Python
    builtin stringifies to an unstable repr (``<built-in function add>``); for those we fall back
    to ``__name__`` (``add``) so the op_type stays stable and namespace-normalizable downstream.
    """
    text = str(target)
    if text.startswith("<"):
        return str(getattr(target, "__name__", text))
    return text


def _op_type(node: Any) -> str:
    """Map an FX node to its op_type. Computed ops use their target; the structural markers
    (placeholder / output / get_attr) use the node's ``op`` kind."""
    if node.op in ("call_function", "call_method", "call_module"):
        return _format_target(node.target)
    return str(node.op)


def _iter_node_refs(value: Any, node_cls: type) -> list[Any]:
    """Flatten an args/kwargs value into the FX nodes it references, in order. Operand order is
    preserved (it is load-bearing — ``sub(a, b)`` differs from ``sub(b, a)``), and nested
    tuples/lists (such as an ``output`` node's ``((x,),)``) are flattened depth-first."""
    refs: list[Any] = []
    if isinstance(value, node_cls):
        refs.append(value)
    elif isinstance(value, tuple | list):
        for item in value:
            refs.extend(_iter_node_refs(item, node_cls))
    return refs


def _is_scalar(value: Any) -> bool:
    """Whether a value is a simple constant worth recording as an attr (for modify-detection).
    Bools are included (they are valid constants); containers and tensors are not."""
    return isinstance(value, bool | int | float | str) or value is None


def _serialize_node(node: Any, node_cls: type) -> FxNode:
    """Serialize one FX node to the contract: ordered input ids from the node references in its
    args then kwargs, and scalar constants captured as attrs keyed by position / kwarg name.

    Empty ``inputs`` / ``attrs`` are left unset so the serializer omits them, matching the
    skip-empty convention the schema bindings use (ADR-024)."""
    inputs = [ref.name for ref in _iter_node_refs(node.args, node_cls)]
    inputs += [ref.name for ref in _iter_node_refs(tuple(node.kwargs.values()), node_cls)]

    attrs: dict[str, Any] = {}
    for i, arg in enumerate(node.args):
        if _is_scalar(arg):
            attrs[f"arg{i}"] = arg
    for key, val in node.kwargs.items():
        if _is_scalar(val):
            attrs[key] = val

    fields: dict[str, Any] = {"id": str(node.name), "op_type": _op_type(node)}
    if inputs:
        fields["inputs"] = inputs
    if attrs:
        fields["attrs"] = attrs
    return FxNode(**fields)


class CompileArtifactCollector:
    """Captures compiled FX graphs and writes a schema-valid ``.cls.json``.

    Mirrors :class:`compile_lens.collectors.recompile.RecompileCollector`'s session + redaction
    handling. The capture surface is :meth:`capture` (compile + run a function once) or
    :meth:`backend` (a ``torch.compile`` backend to wire in manually); both feed
    :meth:`add_graph_module`, and :meth:`finalize` writes the artifact.

    Args:
        output_path: where :meth:`finalize` writes the ``.cls.json``.
        redaction_policy: classification for the session (D11), coerced through
            :class:`RedactionPolicy` so an unknown value fails closed.
        session_id / timestamp / torch_version: session metadata; each defaults to a runtime
            value when ``None`` (fresh UUID4, current UTC time, installed torch version), and
            tests pass them explicitly for deterministic output.
    """

    def __init__(
        self,
        output_path: Path | str,
        *,
        redaction_policy: RedactionPolicy | str = RedactionPolicy.DEFAULT_STRICT,
        session_id: str | None = None,
        timestamp: str | None = None,
        torch_version: str | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.redaction_policy = RedactionPolicy(redaction_policy)  # fail-closed (D11)
        self._session_id = session_id
        self._timestamp = timestamp
        self._torch_version = torch_version
        self._compiled_graphs: list[CompiledGraph] = []

    # ── capture ──────────────────────────────────────────────────────────────────────
    def add_graph_module(self, graph_module: Any) -> None:
        """Serialize one captured ``torch.fx.GraphModule`` into a :class:`CompiledGraph` and
        append it. The graph id is positional (``graph_0``, ``graph_1``, …) in capture order."""
        import torch  # noqa: PLC0415 — optional dependency, imported lazily

        graph_id = f"graph_{len(self._compiled_graphs)}"
        nodes = [_serialize_node(n, torch.fx.Node) for n in graph_module.graph.nodes]
        self._compiled_graphs.append(CompiledGraph(graph_id=graph_id, nodes=nodes))

    def backend(self) -> Callable[..., Any]:
        """A ``torch.compile`` backend that captures the aten-normalized forward graph.

        Routes through ``aot_autograd`` so the captured graph carries canonical aten ops rather
        than the dynamo-level Python builtins. The compiled callable runs the graph eagerly —
        capture, not acceleration, is the point."""
        from torch._dynamo.backends.common import aot_autograd  # noqa: PLC0415

        def _forward_compiler(aten_gm: Any, _example_inputs: Any) -> Any:
            self.add_graph_module(aten_gm)
            return aten_gm.forward

        # aot_autograd is untyped (returns Any); bind to the declared type so the public
        # signature stays precise rather than leaking Any to callers.
        compile_backend: Callable[..., Any] = aot_autograd(fw_compiler=_forward_compiler)
        return compile_backend

    def capture(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        """Compile ``fn`` with the capturing backend and run it once under ``no_grad`` so the
        graph is traced and serialized. ``fn`` may be a module or a plain callable; pass a
        function whose compilation has not been cached elsewhere so the backend is invoked."""
        import torch  # noqa: PLC0415

        compiled = torch.compile(fn, backend=self.backend())
        with torch.no_grad():
            compiled(*args, **kwargs)

    # ── serialization ──────────────────────────────────────────────────────────────────
    def _build_session(self) -> Session:
        def _detect_torch_version() -> str:
            try:
                import torch  # noqa: PLC0415

                return str(torch.__version__)
            except Exception:  # pragma: no cover — environment-dependent
                return "unknown"

        return Session(
            id=self._session_id or str(uuid.uuid4()),
            timestamp=self._timestamp or datetime.now(UTC).isoformat(),
            torch_version=self._torch_version or _detect_torch_version(),
            redaction_policy=self.redaction_policy,
        )

    def finalize(self) -> Path:
        """Assemble the captured graphs into a :class:`ClsArtifact` and write it.

        Under a strict policy any captured artifact paths are normalized at write time (D11);
        the node-level structure itself carries op types and ids, which are not secret-bearing.
        """
        from compile_lens.security import redactor  # noqa: PLC0415

        if redactor.is_strict(self.redaction_policy):
            repo = Path.cwd().name
            for graph in self._compiled_graphs:
                if graph.fx_graph_path is not None:
                    graph.fx_graph_path = redactor.normalize_path(graph.fx_graph_path, repo=repo)
                if graph.inductor_ir_path is not None:
                    graph.inductor_ir_path = redactor.normalize_path(
                        graph.inductor_ir_path, repo=repo
                    )

        artifact = ClsArtifact(
            schema_version=SCHEMA_VERSION,
            session=self._build_session(),
            compiled_graphs=self._compiled_graphs,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(to_json(artifact))
        _log.debug(
            "artifact_collector.finalize",
            output_path=str(self.output_path),
            compiled_graphs=len(self._compiled_graphs),
            redaction_policy=str(self.redaction_policy),
        )
        return self.output_path
