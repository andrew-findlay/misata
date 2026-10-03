"""Synthetic tells: the statistical fingerprints that make data look generated.

``misata audit`` catches contradictions: an order shipped before it was
placed, an age that disagrees with a birth date. Data can be free of every
contradiction and still read as fake to anyone who has looked at a production
table, because its *shapes* are wrong. Amounts spread evenly from min to max.
Every customer has about five orders. Every category gets the same share.
Timestamps pile up at midnight, or are spread flat across the hours and the
days of the week. ``Kevin Brown`` has the email ``rollinslisa@hotmail.com``.
Nothing is ever null.

Each of those is a *tell*, and each can be detected without any real data to
compare against, because real operational data has known regularities:

- money is right-skewed with a long tail, and transaction amounts spanning a
  few orders of magnitude follow Benford's first-digit law;
- popularity is concentrated: a few customers place many orders, a few
  products take a large share of sales (high Gini on FK fan-out);
- categorical columns are rarely perfectly balanced;
- human activity has a weekly and a daily rhythm, and it does not happen at
  exactly 00:00:00;
- emails are derived from names, and a few providers dominate;
- some optional values are missing.

:func:`realism_report` runs every check that applies to the tables it is given
and returns a scored :class:`RealismReport`. It works on any tabular data,
whoever generated it: a Faker script, SDV, Mockaroo, or Misata itself.

What a clean report means, and what it does not: each check is a heuristic
with a stated threshold, chosen to flag shapes real data almost never has.
Passing means "none of the known tells is present", not "indistinguishable
from production". Some tells are legitimate in context (a randomized A/B
assignment *is* balanced; a nightly batch job *does* run at midnight), so most
findings are warnings, and a check can be skipped with ``skip=[...]``.

Usage::

    import misata
    tables = misata.generate("An ecommerce store with 2k customers and orders")
    report = misata.realism_report(tables)
    print(report.summary())
    report.passed          # False if any check failed outright
    report.score           # 0.0 - 1.0, pass=1, warn=0.5, fail=0
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PASS, WARN, FAIL = "pass", "warn", "fail"
_STATUS_SCORE = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TellCheck:
    """One check applied to one table/column."""

    check: str
    table: str
    column: Optional[str]
    status: str               # "pass" | "warn" | "fail"
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check, "table": self.table, "column": self.column,
            "status": self.status, "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class RealismReport:
    """Every applicable tell check, with an overall score."""

    checks: List[TellCheck] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.checks:
            return 1.0
        return float(np.mean([_STATUS_SCORE[c.status] for c in self.checks]))

    @property
    def passed(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)

    @property
    def tells(self) -> List[TellCheck]:
        """Checks that did not pass, failures first."""
        bad = [c for c in self.checks if c.status != PASS]
        return sorted(bad, key=lambda c: (c.status != FAIL, c.table, c.column or ""))

    def counts(self) -> Dict[str, int]:
        out = {PASS: 0, WARN: 0, FAIL: 0}
        for c in self.checks:
            out[c.status] += 1
        return out

    def summary(self) -> str:
        n = self.counts()
        lines = [
            f"Realism score: {self.score:.2f}  "
            f"({len(self.checks)} checks: {n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail)"
        ]
        for c in self.tells:
            where = f"{c.table}.{c.column}" if c.column else c.table
            lines.append(f"  [{c.status}] {c.check:<22} {where}: {c.message}")
        if not self.tells:
            lines.append("  No known synthetic tells found.")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "passed": self.passed,
            "counts": self.counts(),
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Column classification (by name, then by content)
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(
    r"(?:^|_)(amount|amt|total|subtotal|revenue|spend|spent|balance|payment|"
    r"price|cost|fee|value|salary|income|mrr|arr|gmv|sales)(?:_|$)", re.I)
# Benford holds for transaction-like totals, not for posted prices (charm
# pricing) or salaries (banded), so it gets the narrower list.
_BENFORD_RE = re.compile(
    r"(?:^|_)(amount|amt|total|subtotal|revenue|spend|spent|balance|payment|"
    r"gmv|sales|value)(?:_|$)", re.I)
_DATE_NAME_RE = re.compile(r"(_at$|_on$|date|time|timestamp|^ts$|_ts$)", re.I)
_NOT_ACTIVITY_RE = re.compile(
    r"(birth|dob|expir|valid_|effective|due|deadline|start_date|end_date|"
    r"scheduled|hire|founded|graduat|period|month|year|week)", re.I)
_ID_RE = re.compile(r"(^id$|_id$|^uuid$|_uuid$|_key$|^pk$|_code$|_number$|_no$)", re.I)
# Columns where an even split is often the design, not a tell.
_BALANCED_BY_DESIGN_RE = re.compile(
    r"(gender|sex|variant|ab_|_ab$|arm|treatment|group|cohort|bucket|quarter|"
    r"month|weekday|day_of_week|dow|shift|split|fold|seat|side)", re.I)
_EMAIL_RE = re.compile(r"(^|_)e?mail($|_)|email", re.I)
_PLACEHOLDER_RE = re.compile(
    r"(?:@example\.(?:com|org|net)\b|\bjohn doe\b|\bjane doe\b|\bacme\b|"
    r"lorem ipsum|^(?:foo|bar|baz|foobar)\d*$|\basdf|\bqwerty\b|^test\d*$|"
    r"^sample\d*$|^dummy|^xxx+$|^placeholder|^todo$|^tbd$)", re.I)
_DIVERSE_NAME_RE = re.compile(
    r"^(full_name|name|customer_name|user_name|product_name|company|"
    r"company_name|title|product|item_name)$", re.I)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _as_datetime(s: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    elif s.dtype == object or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(50)
        if sample.empty or not sample.str.match(r"^\d{4}-\d{2}-\d{2}").all():
            return None
        out = pd.to_datetime(s, errors="coerce")
    else:
        return None
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dropna()


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def _chi2_uniform_p(counts: np.ndarray, expected: Optional[np.ndarray] = None) -> float:
    from scipy.stats import chisquare
    counts = np.asarray(counts, dtype=float)
    if expected is None:
        expected = np.full_like(counts, counts.sum() / len(counts))
    else:
        expected = np.asarray(expected, dtype=float)
        expected = expected * counts.sum() / expected.sum()
    return float(chisquare(counts, expected).pvalue)


def _ascii(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


# ---------------------------------------------------------------------------
# Column checks
# ---------------------------------------------------------------------------

def _check_amount_shape(table: str, col: str, s: pd.Series) -> Optional[TellCheck]:
    v = s.dropna().to_numpy(dtype=float)
    v = v[v > 0]
    if len(v) < 200 or np.unique(v).size < 20:
        return None
    from scipy.stats import kstest, skew
    lo, hi = float(v.min()), float(v.max())
    ks_uniform = float(kstest((v - lo) / (hi - lo), "uniform").statistic) if hi > lo else 1.0
    sk = float(skew(v))
    p50, p99 = np.percentile(v, [50, 99])
    tail = float(p99 / p50) if p50 > 0 else float("inf")
    ev = {"n": int(len(v)), "skew": round(sk, 3), "ks_vs_uniform": round(ks_uniform, 4),
          "p99_over_median": round(tail, 2)}
    if ks_uniform < 0.05:
        return TellCheck("amount_shape", table, col, FAIL,
                         f"values are spread evenly between {lo:g} and {hi:g}; real money "
                         f"is right-skewed with a long tail", ev)
    if sk < 0.5 or tail < 2.0:
        return TellCheck("amount_shape", table, col, WARN,
                         f"thin tail (skew {sk:.2f}, p99 is {tail:.1f}x the median); real "
                         f"amounts usually skew right with p99 several times the median", ev)
    return TellCheck("amount_shape", table, col, PASS, "right-skewed with a long tail", ev)


_BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def _check_benford(table: str, col: str, s: pd.Series) -> Optional[TellCheck]:
    v = s.dropna().to_numpy(dtype=float)
    v = v[v > 0]
    if len(v) < 500:
        return None
    p1, p99 = np.percentile(v, [1, 99])
    if p1 <= 0 or math.log10(p99 / p1) < 2:
        return None  # Benford needs a few orders of magnitude to apply
    first = (v / 10 ** np.floor(np.log10(v))).astype(int)
    first = first[(first >= 1) & (first <= 9)]
    observed = np.bincount(first, minlength=10)[1:] / len(first)
    mad = float(np.abs(observed - _BENFORD).mean())
    ev = {"n": int(len(first)), "mad": round(mad, 4),
          "first_digit_share": [round(x, 3) for x in observed]}
    # Nigrini's first-digit thresholds: <=0.012 close, <=0.015 acceptable.
    if mad > 0.015:
        return TellCheck("benford", table, col, WARN,
                         f"first digits do not follow Benford's law (MAD {mad:.3f} > 0.015); "
                         f"'1' leads {observed[0]:.0%} of values vs ~30% in real totals", ev)
    return TellCheck("benford", table, col, PASS, f"conforms to Benford (MAD {mad:.3f})", ev)


def _check_category_balance(table: str, col: str, s: pd.Series) -> Optional[TellCheck]:
    if _BALANCED_BY_DESIGN_RE.search(col) or _ID_RE.search(col):
        return None
    v = s.dropna()
    if len(v) < 300:
        return None
    counts = v.astype(str).value_counts().to_numpy()
    k = len(counts)
    if k < 3 or k > 30 or k / len(v) > 0.05:
        return None
    p = _chi2_uniform_p(counts)
    shares = counts / counts.sum()
    ev = {"n": int(len(v)), "categories": k, "chi2_p_vs_uniform": round(p, 4),
          "max_share": round(float(shares.max()), 3), "min_share": round(float(shares.min()), 3)}
    if p > 0.05:
        return TellCheck("category_balance", table, col, WARN,
                         f"all {k} categories get statistically equal shares; real category "
                         f"columns are almost never balanced", ev)
    return TellCheck("category_balance", table, col, PASS,
                     "categories are unevenly distributed", ev)


def _check_time_rhythm(table: str, col: str, raw: pd.Series) -> List[TellCheck]:
    if _NOT_ACTIVITY_RE.search(col):
        return []
    t = _as_datetime(raw)
    if t is None or len(t) < 500:
        return []
    out: List[TellCheck] = []
    span_days = (t.max() - t.min()).days
    if span_days >= 28:
        dow = t.dt.dayofweek.to_numpy()
        counts = np.bincount(dow, minlength=7)
        # expected: how many of each weekday the span actually contains
        days = pd.date_range(t.min().normalize(), t.max().normalize(), freq="D")
        expected = np.bincount(days.dayofweek, minlength=7)
        p = _chi2_uniform_p(counts, expected)
        ev = {"n": int(len(t)), "span_days": int(span_days), "chi2_p_vs_flat": round(p, 4),
              "weekday_counts_mon_to_sun": counts.tolist()}
        if p > 0.01:
            out.append(TellCheck("weekday_rhythm", table, col, WARN,
                                 "every day of the week is equally busy; real activity has a "
                                 "weekly rhythm (weekends differ from weekdays)", ev))
        else:
            out.append(TellCheck("weekday_rhythm", table, col, PASS, "has a weekly rhythm", ev))

    secs = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy()
    if (secs == 0).all():
        return out  # a date stored as a timestamp, not an activity time
    midnight = float((secs == 0).mean())
    hours = np.bincount(t.dt.hour.to_numpy(), minlength=24)
    peak = int(hours.argmax())
    ev = {"n": int(len(t)), "share_at_midnight": round(midnight, 3), "peak_hour": peak,
          "hour_counts": hours.tolist()}
    if midnight > 0.10:
        out.append(TellCheck("daily_rhythm", table, col, FAIL,
                             f"{midnight:.0%} of timestamps are exactly 00:00:00 while the rest "
                             f"carry a time of day; dates were mixed into a timestamp column", ev))
        return out
    p = _chi2_uniform_p(hours)
    ev["chi2_p_vs_flat"] = round(p, 4)
    if p > 0.01:
        out.append(TellCheck("daily_rhythm", table, col, WARN,
                             "every hour of the day is equally busy, 3am included; real "
                             "activity follows a daily cycle", ev))
    elif peak <= 5:
        out.append(TellCheck("daily_rhythm", table, col, WARN,
                             f"activity peaks at {peak:02d}:00; human activity peaks in the "
                             f"day or evening (fine for batch jobs, odd for people)", ev))
    else:
        out.append(TellCheck("daily_rhythm", table, col, PASS,
                             f"has a daily cycle peaking at {peak:02d}:00", ev))
    return out


def _check_placeholders(table: str, col: str, s: pd.Series) -> Optional[TellCheck]:
    v = s.dropna().astype(str)
    if len(v) < 50:
        return None
    hits = v.str.contains(_PLACEHOLDER_RE)
    share = float(hits.mean())
    if share == 0:
        return None
    ev = {"n": int(len(v)), "share": round(share, 4), "examples": v[hits].unique()[:3].tolist()}
    status = FAIL if share > 0.01 else WARN
    return TellCheck("placeholder_values", table, col, status,
                     f"{share:.1%} of values are placeholders (e.g. {ev['examples'][0]!r})", ev)


def _check_email_domains(table: str, col: str, s: pd.Series) -> Optional[TellCheck]:
    v = s.dropna().astype(str)
    v = v[v.str.contains("@")]
    if len(v) < 300:
        return None
    domains = v.str.split("@").str[-1].str.lower()
    counts = domains.value_counts().to_numpy()
    if len(counts) < 5 or len(counts) > 0.2 * len(v):
        return None  # corporate/unique domains: a different shape entirely
    top = float(counts[0] / counts.sum())
    p = _chi2_uniform_p(counts)
    ev = {"n": int(len(v)), "domains": int(len(counts)), "top_domain_share": round(top, 3),
          "chi2_p_vs_uniform": round(p, 4)}
    if p > 0.05:
        return TellCheck("email_domains", table, col, WARN,
                         f"{len(counts)} email providers each get an equal share; in real "
                         f"consumer data one or two providers dominate", ev)
    return TellCheck("email_domains", table, col, PASS, "email providers are concentrated", ev)


def _check_diversity(table: str, col: str, s: pd.Series) -> Optional[TellCheck]:
    if not _DIVERSE_NAME_RE.match(col):
        return None
    v = s.dropna().astype(str)
    if len(v) < 500:
        return None
    distinct = v.nunique()
    ratio = distinct / len(v)
    top = float(v.value_counts(normalize=True).iloc[0])
    ev = {"n": int(len(v)), "distinct": int(distinct), "distinct_ratio": round(ratio, 3),
          "top_value_share": round(top, 4)}
    if ratio < 0.2:
        return TellCheck("value_diversity", table, col, WARN,
                         f"only {distinct} distinct values across {len(v)} rows; the pool is "
                         f"small enough for a reader to notice repeats", ev)
    return TellCheck("value_diversity", table, col, PASS, "values are diverse", ev)


def _check_text_templates(table: str, col: str, s: pd.Series) -> Optional[TellCheck]:
    v = s.dropna().astype(str)
    if len(v) < 200 or v.str.len().mean() < 40 or not v.str.contains(" ").all():
        return None
    openers = v.str.lower().str.split().str[:3].str.join(" ")
    top_opener = float(openers.value_counts(normalize=True).iloc[0])
    dup = float(v.duplicated().mean())
    ev = {"n": int(len(v)), "top_opener_share": round(top_opener, 3),
          "top_opener": openers.value_counts().index[0], "duplicate_share": round(dup, 3)}
    if dup > 0.2 or top_opener > 0.10:
        reason = (f"{dup:.0%} of texts are exact duplicates" if dup > 0.2 else
                  f"{top_opener:.0%} of texts open with the same three words "
                  f"({ev['top_opener']!r})")
        return TellCheck("text_templates", table, col, WARN,
                         f"{reason}; free text reads as templated", ev)
    return TellCheck("text_templates", table, col, PASS, "text openings vary", ev)


# ---------------------------------------------------------------------------
# Table and cross-table checks
# ---------------------------------------------------------------------------

def _check_too_clean(table: str, df: pd.DataFrame) -> Optional[TellCheck]:
    cols = [c for c in df.columns if not _ID_RE.search(str(c))]
    if len(df) < 100 or len(cols) < 5:
        return None
    nulls = int(df[cols].isna().sum().sum())
    ev = {"rows": int(len(df)), "non_key_columns": len(cols), "null_cells": nulls}
    if nulls == 0:
        return TellCheck("too_clean", table, None, WARN,
                         f"no nulls in any of {len(cols)} non-key columns; real tables have "
                         f"optional fields that are sometimes empty", ev)
    return TellCheck("too_clean", table, None, PASS, "some optional values are missing", ev)


_NAME_COLS = ("first_name", "last_name", "full_name", "name", "customer_name", "user_name")


def _check_name_email(table: str, df: pd.DataFrame) -> Optional[TellCheck]:
    email = next((c for c in df.columns if _EMAIL_RE.search(str(c))), None)
    names = [c for c in _NAME_COLS if c in df.columns]
    if email is None or not names:
        return None
    sub = df[[email] + names].dropna()
    sub = sub[sub[email].astype(str).str.contains("@")]
    if len(sub) < 100:
        return None
    local = sub[email].astype(str).str.split("@").str[0].map(_ascii)
    tokens = sub[names].astype(str).agg(" ".join, axis=1).map(
        lambda s: [t for t in re.split(r"[^a-z]+", _ascii(s)) if len(t) >= 3])
    match = np.fromiter((any(t in loc for t in toks) for loc, toks in zip(local, tokens)),
                        dtype=bool, count=len(sub))
    share = float(match.mean())
    ev = {"n": int(len(sub)), "email_column": email, "name_columns": names,
          "share_matching": round(share, 3)}
    if share < 0.2:
        return TellCheck("name_email", table, email, FAIL,
                         f"only {share:.0%} of emails contain the person's name; the two "
                         f"columns were generated independently", ev)
    if share < 0.5:
        return TellCheck("name_email", table, email, WARN,
                         f"{share:.0%} of emails contain the person's name; most real "
                         f"addresses are built from it", ev)
    return TellCheck("name_email", table, email, PASS,
                     f"{share:.0%} of emails derive from the name", ev)


def _infer_relationships(tables: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, str, str]]:
    """(parent, parent_key, child, child_key) for `<entity>_id` columns that
    resolve into another table's key."""
    out = []
    for child, cdf in tables.items():
        for col in cdf.columns:
            m = re.match(r"^(.+)_id$", str(col))
            if not m:
                continue
            ent = m.group(1)
            stem = ent[:-1] if ent.endswith("y") else ent
            candidates = [ent, ent + "s", ent + "es", stem + "ies"]
            for parent in candidates:
                if parent == child or parent not in tables:
                    continue
                pdf = tables[parent]
                pkey = "id" if "id" in pdf.columns else (col if col in pdf.columns else None)
                if pkey is None:
                    continue
                vals = cdf[col].dropna()
                if len(vals) and vals.isin(set(pdf[pkey].dropna())).mean() >= 0.9:
                    out.append((parent, pkey, child, str(col)))
                    break
    return out


def _check_fanout(parent: str, pkey: str, child: str, ckey: str,
                  tables: Dict[str, pd.DataFrame]) -> Optional[TellCheck]:
    pdf, cdf = tables[parent], tables[child]
    keys = pdf[pkey].dropna()
    refs = cdf[ckey].dropna()
    if len(keys) < 20 or len(refs) < 200 or len(refs) < len(keys):
        return None  # 1:1 or sparse links have no fan-out shape to judge
    per_parent = refs.value_counts().reindex(keys.unique(), fill_value=0).to_numpy()
    g = _gini(per_parent)
    top_n = max(1, int(round(len(per_parent) * 0.1)))
    top_share = float(np.sort(per_parent)[::-1][:top_n].sum() / per_parent.sum())
    ev = {"parents": int(len(per_parent)), "children": int(len(refs)), "gini": round(g, 3),
          "top10pct_share": round(top_share, 3),
          "max_children": int(per_parent.max()), "median_children": float(np.median(per_parent))}
    msg_where = f"{child}.{ckey} -> {parent}.{pkey}"
    if g < 0.25:
        return TellCheck("fanout_skew", child, ckey, FAIL,
                         f"{msg_where}: every {parent[:-1] if parent.endswith('s') else parent} "
                         f"gets about the same number of {child} (Gini {g:.2f}); real "
                         f"popularity is concentrated, the top 10% usually hold 30%+", ev)
    if g < 0.4:
        return TellCheck("fanout_skew", child, ckey, WARN,
                         f"{msg_where}: fan-out is mildly concentrated (Gini {g:.2f}, top 10% "
                         f"hold {top_share:.0%}); real data is usually more lopsided", ev)
    return TellCheck("fanout_skew", child, ckey, PASS,
                     f"{msg_where}: concentrated (Gini {g:.2f}, top 10% hold {top_share:.0%})", ev)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_CHECKS = (
    "amount_shape", "benford", "category_balance", "weekday_rhythm", "daily_rhythm",
    "placeholder_values", "email_domains", "value_diversity", "text_templates",
    "too_clean", "name_email", "fanout_skew",
)


def _schema_relationships(schema: Any) -> Optional[List[Tuple[str, str, str, str]]]:
    rels = getattr(schema, "relationships", None)
    if not rels:
        return None
    return [(r.parent_table, r.parent_key, r.child_table, r.child_key) for r in rels]


def realism_report(
    tables: Dict[str, pd.DataFrame] | pd.DataFrame,
    schema: Any = None,
    *,
    skip: Iterable[str] = (),
) -> RealismReport:
    """Scan tables for the statistical tells of generated data.

    Args:
        tables: mapping of table name to DataFrame (a single DataFrame is
            treated as one table named ``"table"``).
        schema: optional :class:`~misata.schema.SchemaConfig`. Its declared
            relationships are used for the fan-out check; without it,
            ``<entity>_id`` columns are matched to tables by name.
        skip: check names to leave out (see :data:`ALL_CHECKS`), for tells
            that are legitimate in your data.

    Returns:
        A :class:`RealismReport`. Only checks with enough rows to be
        meaningful are run, so small fixtures produce short reports.
    """
    if isinstance(tables, pd.DataFrame):
        tables = {"table": tables}
    skipped = set(skip)
    unknown = skipped - set(ALL_CHECKS)
    if unknown:
        raise ValueError(f"unknown check(s) {sorted(unknown)}; choose from {list(ALL_CHECKS)}")

    checks: List[TellCheck] = []

    def add(c: Optional[TellCheck] | Sequence[TellCheck]) -> None:
        for item in ([] if c is None else [c] if isinstance(c, TellCheck) else c):
            if item.check not in skipped:
                checks.append(item)

    for table, df in tables.items():
        for col in df.columns:
            name, s = str(col), df[col]
            if _is_numeric(s) and not _ID_RE.search(name):
                if _MONEY_RE.search(name):
                    add(_check_amount_shape(table, name, s))
                if _BENFORD_RE.search(name):
                    add(_check_benford(table, name, s))
            if _DATE_NAME_RE.search(name) or pd.api.types.is_datetime64_any_dtype(s):
                add(_check_time_rhythm(table, name, s))
                continue
            textual = (s.dtype == object or pd.api.types.is_string_dtype(s)
                       or isinstance(s.dtype, pd.CategoricalDtype))
            if textual:
                add(_check_category_balance(table, name, s))
                add(_check_placeholders(table, name, s))
                add(_check_diversity(table, name, s))
                add(_check_text_templates(table, name, s))
                if _EMAIL_RE.search(name):
                    add(_check_email_domains(table, name, s))
        add(_check_too_clean(table, df))
        add(_check_name_email(table, df))

    rels = _schema_relationships(schema) or _infer_relationships(tables)
    for parent, pkey, child, ckey in rels:
        if (parent in tables and child in tables
                and pkey in tables[parent] and ckey in tables[child]):
            add(_check_fanout(parent, pkey, child, ckey, tables))

    return RealismReport(checks=checks)
