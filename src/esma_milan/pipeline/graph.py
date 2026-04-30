"""Stage 4: bipartite loan-property graph + connected components.

Mirrors r_reference/R/cross_collateralization_functions.R::
build_loan_collateral_graph. Produces three frames keyed on a
deterministic `collateral_group_id`:

    loan_groups        (calc_loan_id, collateral_group_id)
    collateral_groups  (calc_property_id, collateral_group_id)
    edges_with_group   (loan_exposure_id, collateral_id, collateral_group_id)

Group IDs are 1-indexed integers assigned in a fully deterministic
order:

    1. Build the underlying undirected bipartite graph (loan nodes ↔
       property nodes, edges from one row of the loans-properties
       inner join).
    2. Compute connected components via networkx.
    3. Within each component, sort nodes lexically.
    4. Sort components by their lex-smallest node.
    5. Assign group IDs 1..N in that sorted order.

Step 3 + 4 + 5 mean the output is a pure function of the input edge set
- no dependence on networkx iteration order, hash randomisation, or
input row order. R's `igraph::components()` produces different integer
labels (R sees nodes in a different order), so the parity harness
canonicalises both sides via sorted-min-loan-id-per-group before
comparing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import networkx as nx
import polars as pl
import structlog

log = structlog.get_logger(__name__)

# Internal node-type tags. Loan IDs and property IDs share string
# namespace risk in principle (a defective input could duplicate "X1"
# across both sides); tagging the nodes ("loan", id) / ("property", id)
# makes the bipartite structure explicit and protects against that.
_LOAN: str = "loan"
_PROPERTY: str = "property"


@dataclass(frozen=True)
class GraphResult:
    """Output of Stage 4.

    All three frames carry a deterministic `collateral_group_id` (Int64,
    1-indexed). `loan_groups` and `collateral_groups` are sorted by
    their key column ascending, so identical inputs always produce
    identical row order.
    """

    loan_groups: pl.DataFrame
    collateral_groups: pl.DataFrame
    edges_with_group: pl.DataFrame


def build_loan_collateral_graph(
    loans: pl.DataFrame,
    properties: pl.DataFrame,
) -> GraphResult:
    """Build the loan-collateral bipartite graph and assign group IDs."""
    if "calc_loan_id" not in loans.columns:
        raise ValueError(
            "build_loan_collateral_graph: loans is missing column 'calc_loan_id'"
        )
    for col in ("underlying_exposure_identifier", "calc_property_id"):
        if col not in properties.columns:
            raise ValueError(
                f"build_loan_collateral_graph: properties is missing column {col!r}"
            )

    # --- Edge list: inner-join loans.calc_loan_id onto
    # --- properties.underlying_exposure_identifier; keep distinct pairs.
    edges = (
        loans.select("calc_loan_id")
        .join(
            properties.select(
                "underlying_exposure_identifier", "calc_property_id"
            ),
            left_on="calc_loan_id",
            right_on="underlying_exposure_identifier",
            how="inner",
        )
        .rename(
            {
                "calc_loan_id": "loan_exposure_id",
                "calc_property_id": "collateral_id",
            }
        )
        .unique(subset=["loan_exposure_id", "collateral_id"], keep="first")
        .drop_nulls(["loan_exposure_id", "collateral_id"])
    )

    edge_pairs: list[tuple[str, str]] = list(
        zip(
            edges["loan_exposure_id"].to_list(),
            edges["collateral_id"].to_list(),
            strict=True,
        )
    )

    # --- Build the graph and compute components.
    graph = nx.Graph()
    loan_nodes_seen: set[str] = set()
    property_nodes_seen: set[str] = set()
    for loan_id, prop_id in edge_pairs:
        graph.add_edge((_LOAN, loan_id), (_PROPERTY, prop_id))
        loan_nodes_seen.add(loan_id)
        property_nodes_seen.add(prop_id)

    components_raw: Iterator[set[tuple[str, str]]] = nx.connected_components(graph)
    canonical = _canonicalise_components(components_raw)

    # --- Map every node to its 1-indexed group label.
    node_to_gid: dict[tuple[str, str], int] = {}
    for gid, nodes in enumerate(canonical, start=1):
        for n in nodes:
            node_to_gid[n] = gid

    # --- Build the three output frames.
    loan_groups = _build_node_frame(
        nodes=sorted(loan_nodes_seen),
        node_kind=_LOAN,
        id_column="calc_loan_id",
        node_to_gid=node_to_gid,
    )
    collateral_groups = _build_node_frame(
        nodes=sorted(property_nodes_seen),
        node_kind=_PROPERTY,
        id_column="calc_property_id",
        node_to_gid=node_to_gid,
    )

    edges_with_group = (
        edges.with_columns(
            pl.Series(
                "collateral_group_id",
                [node_to_gid[(_LOAN, lid)] for lid in edges["loan_exposure_id"].to_list()],
                dtype=pl.Int64,
            )
        )
        .sort(by=["collateral_group_id", "loan_exposure_id", "collateral_id"])
    )

    log.info(
        "build_loan_collateral_graph",
        n_groups=len(canonical),
        n_loan_nodes=len(loan_nodes_seen),
        n_property_nodes=len(property_nodes_seen),
        n_edges=len(edge_pairs),
    )

    return GraphResult(
        loan_groups=loan_groups,
        collateral_groups=collateral_groups,
        edges_with_group=edges_with_group,
    )


# Stage 4 driver alias - mirrors the run_stageN naming of other stages.
# Kept thin because the build helper already does everything Stage 4
# needs to do.
def run_stage4(loans: pl.DataFrame, properties: pl.DataFrame) -> GraphResult:
    """Stage 4 driver. Wraps build_loan_collateral_graph for naming
    consistency with run_stage1..run_stage3."""
    return build_loan_collateral_graph(loans, properties)


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------


def _canonicalise_components(
    components: Iterable[set[tuple[str, str]]],
) -> list[list[tuple[str, str]]]:
    """Turn networkx's set-of-sets output into a deterministically-ordered
    list-of-lists.

    Within each component, nodes are sorted lex (tuples sort by type
    first, then ID, so all loan nodes precede all property nodes). The
    list of components itself is sorted by each component's lex-smallest
    node, so the output ordering is a pure function of the input edge
    set with no dependence on networkx iteration order or hash seed.
    """
    sorted_components = [sorted(c) for c in components]
    sorted_components.sort(key=lambda nodes: nodes[0])
    return sorted_components


def _build_node_frame(
    *,
    nodes: list[str],
    node_kind: str,
    id_column: str,
    node_to_gid: dict[tuple[str, str], int],
) -> pl.DataFrame:
    """Build a (id_column, collateral_group_id) frame for one node-kind."""
    return pl.DataFrame(
        {
            id_column: nodes,
            "collateral_group_id": [node_to_gid[(node_kind, n)] for n in nodes],
        },
        schema={id_column: pl.String, "collateral_group_id": pl.Int64},
    )
