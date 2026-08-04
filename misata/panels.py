"""Ragged per-entity panels: one child row per period of a parent's own span.

Some tables have no row count to declare. "One row per subscription per day it
was active" is not a number — it is the sum over parents of a duration, and it
moves whenever the parents do. Writing `rows: 1500000` states a consequence
rather than the rule, and gets it wrong as soon as anything upstream changes.

The result is also *ragged*: every parent contributes a different number of rows.
A daily meter over accounts that joined and left at different times is the
ordinary case, not an exotic one, and it is the shape most window functions are
written against.

This module turns a `RowsPerParent` declaration and a parent frame into the
child's index: which parent each row belongs to, and which date it carries.
Everything else about the child generates normally against that row count.

The expansion is deliberately not random. Steps are contiguous, ordered per
parent, and complete — that is the whole point of a panel, and the reason a
`row_number() over (partition by ... order by ...)` over the result means
something.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

# pandas offset aliases per grain. Anchored on the period start so a monthly
# panel lands on the first of the month rather than drifting by the day of the
# parent's start date, which is what makes two parents' months line up.
_GRAIN_FREQ = {
    "day": "D",
    "week": "W-MON",
    "month": "MS",
    "quarter": "QS",
    "year": "YS",
}


def _floor_to_grain(series: pd.Series, grain: str) -> pd.Series:
    """Snap timestamps to the start of their period.

    Without this a monthly panel starting on the 17th produces the 17th of every
    month, and two parents that started in the same month never agree on a
    period label — so nothing downstream can group by it.
    """
    if grain == "day":
        return series.dt.normalize()
    if grain == "week":
        return series.dt.normalize() - pd.to_timedelta(series.dt.weekday, unit="D")
    if grain == "month":
        return series.values.astype("datetime64[M]").astype("datetime64[ns]")
    if grain == "quarter":
        months = series.values.astype("datetime64[M]").astype(int)
        return (months - (months % 3)).astype("datetime64[M]").astype("datetime64[ns]")
    if grain == "year":
        return series.values.astype("datetime64[Y]").astype("datetime64[ns]")
    raise ValueError(f"unknown grain {grain!r}")


def _period_counts(starts: pd.Series, ends: pd.Series, grain: str) -> np.ndarray:
    """How many periods each parent spans, inclusive of both ends."""
    if grain == "day":
        steps = (ends - starts).dt.days
    elif grain == "week":
        steps = (ends - starts).dt.days // 7
    else:
        months = (
            (ends.dt.year - starts.dt.year) * 12 + (ends.dt.month - starts.dt.month)
        )
        steps = months if grain == "month" else (
            months // 3 if grain == "quarter" else (ends.dt.year - starts.dt.year)
        )
    counts = steps.to_numpy(dtype="float64")
    counts = np.where(np.isnan(counts), -1.0, counts) + 1.0
    return np.clip(counts, 0, None).astype("int64")


def plan(
    parent: pd.DataFrame,
    parent_key: str,
    spec: Any,
    table_name: str = "",
) -> Optional[pd.DataFrame]:
    """Build the child's index from the parent frame.

    Returns a frame with two columns — the parent key and the period date —
    with exactly one row per parent per period. Returns None when the
    declaration cannot be honoured, having warned; the caller then falls back to
    the declared row count rather than generating nothing.
    """
    for col in (spec.from_column, spec.to_column):
        if col and col not in parent.columns:
            warnings.warn(
                f"rows_per_parent on '{table_name}' names {col!r}, which is not a "
                f"column on '{parent_key}''s table (has: "
                f"{', '.join(map(str, parent.columns[:12]))}). Falling back to the "
                f"declared row count.",
                UserWarning,
            )
            return None
    if parent_key not in parent.columns:
        warnings.warn(
            f"rows_per_parent on '{table_name}' cannot find parent key "
            f"{parent_key!r}. Falling back to the declared row count.",
            UserWarning,
        )
        return None

    keys = parent[parent_key]
    starts = pd.to_datetime(parent[spec.from_column], errors="coerce")

    if spec.to_column:
        ends = pd.to_datetime(parent[spec.to_column], errors="coerce")
    else:
        ends = pd.Series(pd.NaT, index=parent.index)

    # An open-ended period is the normal case for anything still live, so it is
    # closed at the declared boundary rather than dropped. Without a boundary
    # there is genuinely no span, and those rows contribute nothing.
    if spec.default_to is not None:
        ends = ends.fillna(pd.Timestamp(spec.default_to))

    starts = _floor_to_grain(pd.Series(starts).dt.normalize(), spec.grain)
    ends = _floor_to_grain(pd.Series(ends).dt.normalize(), spec.grain)
    starts = pd.Series(pd.to_datetime(starts), index=parent.index)
    ends = pd.Series(pd.to_datetime(ends), index=parent.index)

    counts = _period_counts(starts, ends, spec.grain)

    over = counts > spec.max_periods
    if over.any():
        warnings.warn(
            f"rows_per_parent on '{table_name}': {int(over.sum())} parent(s) span "
            f"more than max_periods={spec.max_periods:,} at grain '{spec.grain}' and "
            f"were truncated to it. Their {spec.from_column}/{spec.to_column} are "
            f"probably further apart than intended.",
            UserWarning,
        )
        counts = np.minimum(counts, spec.max_periods)

    total = int(counts.sum())
    if total == 0:
        warnings.warn(
            f"rows_per_parent on '{table_name}' produced no rows: no parent has a "
            f"{spec.to_column or 'default_to'} at or after its {spec.from_column}. "
            f"Falling back to the declared row count.",
            UserWarning,
        )
        return None

    # Repeat each parent by its period count, then walk the offset within each
    # parent's own run. np.arange over the total minus the run start gives the
    # per-parent step index without a Python loop.
    repeated_keys = np.repeat(keys.to_numpy(), counts)
    repeated_starts = np.repeat(starts.to_numpy(), counts)
    run_starts = np.repeat(np.cumsum(counts) - counts, counts)
    step = np.arange(total, dtype="int64") - run_starts

    dates = _advance(repeated_starts, step, spec.grain)

    return pd.DataFrame({parent_key: repeated_keys, spec.date_column: dates})


def _advance(starts: np.ndarray, step: np.ndarray, grain: str) -> np.ndarray:
    """starts + step periods, vectorised."""
    if grain == "day":
        return starts + step.astype("timedelta64[D]")
    if grain == "week":
        return starts + (step * 7).astype("timedelta64[D]")
    months = starts.astype("datetime64[M]").astype("int64")
    if grain == "month":
        return (months + step).astype("datetime64[M]").astype("datetime64[ns]")
    if grain == "quarter":
        return (months + step * 3).astype("datetime64[M]").astype("datetime64[ns]")
    years = starts.astype("datetime64[Y]").astype("int64")
    return (years + step).astype("datetime64[Y]").astype("datetime64[ns]")


def relationship_for(config: Any, table_name: str) -> Optional[Tuple[Any, Any]]:
    """The (relationship, spec) that sizes `table_name`, if one does.

    Two declarations sizing the same child is a contradiction rather than a
    merge, so it raises instead of picking one.
    """
    found = [
        rel for rel in (getattr(config, "relationships", None) or [])
        if rel.child_table == table_name and getattr(rel, "rows_per_parent", None)
    ]
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            f"Table '{table_name}' has {len(found)} rows_per_parent declarations "
            f"(parents: {', '.join(r.parent_table for r in found)}). Its row count "
            f"cannot be two different durations; keep one."
        )
    return found[0], found[0].rows_per_parent


def panel_tables(config: Any) -> set:
    """Child tables whose row count comes from a parent's span."""
    return {
        rel.child_table
        for rel in (getattr(config, "relationships", None) or [])
        if getattr(rel, "rows_per_parent", None)
    }
