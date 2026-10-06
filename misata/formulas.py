"""
Formula engine for derived columns.

This module enables columns that are computed from other columns,
supporting expressions like:
- calories_burned = duration_minutes * @exercises.calories_per_minute
- total_price = quantity * @products.price
- discount_amount = total_price * 0.1
"""

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import ast
import operator

try:
    from simpleeval import simple_eval, NameNotDefined
except ImportError:
    simple_eval = None

    class NameNotDefined(NameError):
        """Fallback used when simpleeval is not installed."""

def _ts(x):
    return np.asarray(pd.to_datetime(np.asarray(x)), dtype="datetime64[ns]")


def _add_days(ts, days):
    """Timestamp plus a (possibly fractional, per-row) number of days."""
    ns = np.round(np.asarray(days, dtype="float64") * 86_400e9)
    ns = np.where(np.isfinite(ns), ns, 0).astype("int64")
    out = _ts(ts) + ns.astype("timedelta64[ns]")
    return np.where(np.isnan(np.asarray(days, dtype="float64")), np.datetime64("NaT", "ns"), out)


def _add_months(ts, months):
    """Timestamp plus whole months, keeping the day (clamped) and time."""
    from misata.panels import add_months
    return add_months(_ts(ts), np.broadcast_to(np.asarray(months, dtype="int64"), _ts(ts).shape))


def _month_start(ts):
    """First instant of each timestamp's month."""
    t = _ts(ts)
    return np.where(np.isnat(t), np.datetime64("NaT", "ns"),
                    t.astype("datetime64[M]").astype("datetime64[ns]"))


def _day_diff(start, end):
    """Calendar days from start to end -- midnights crossed, as SQL counts them."""
    s = _ts(start).astype("datetime64[D]")
    e = _ts(end).astype("datetime64[D]")
    out = (e - s).astype("int64").astype("float64")
    return np.where(np.isnat(s) | np.isnat(e), np.nan, out)


def _month_diff(start, end):
    """Calendar months from start to end -- month boundaries crossed."""
    s = _ts(start).astype("datetime64[M]")
    e = _ts(end).astype("datetime64[M]")
    out = (e - s).astype("int64").astype("float64")
    return np.where(np.isnat(s) | np.isnat(e), np.nan, out)


def _months_elapsed(start, end):
    """Whole months from start to end: anniversaries passed."""
    from misata.panels import _full_months
    s, e = _ts(start), _ts(end)
    return np.where(np.isnat(s) | np.isnat(e), np.nan,
                    _full_months(np.where(np.isnat(s), e, s), e).astype("float64"))


def _weekday(ts):
    """0 Monday ... 6 Sunday."""
    t = _ts(ts).astype("datetime64[D]")
    return np.where(np.isnat(t), -1, (t.astype("int64") + 3) % 7)


def _month_of(ts):
    t = _ts(ts)
    return np.where(np.isnat(t), 0, t.astype("datetime64[M]").astype("int64") % 12 + 1)


def _day_of(ts):
    t = _ts(ts)
    return np.where(np.isnat(t), 0, (t.astype("datetime64[D]") - t.astype("datetime64[M]")
                                      .astype("datetime64[D]")).astype("int64") + 1)


def _days_between(start, end):
    """Fractional days from start to end; NaN where either is null."""
    delta = (_ts(end) - _ts(start)).astype("timedelta64[ns]").astype("float64")
    out = delta / 86_400e9
    return np.where(np.isnat(_ts(end)) | np.isnat(_ts(start)), np.nan, out)


def _isnull(x):
    return pd.isna(np.asarray(x, dtype=object)) if np.asarray(x).dtype == object else pd.isna(np.asarray(x))


def _coalesce(*values):
    out = np.asarray(values[0]).copy()
    for v in values[1:]:
        out = np.where(_isnull(out), np.asarray(v), out)
    return out


def _timestamp(text):
    return np.datetime64(pd.Timestamp(text).to_datetime64(), "ns")


# Whitelist of safe functions
SAFE_FUNCTIONS = {
    'where': np.where,
    'abs': np.abs,
    'round': np.round,
    'ceil': np.ceil,
    'floor': np.floor,
    'min': np.minimum,
    'max': np.maximum,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'log': np.log,
    'exp': np.exp,
    'sqrt': np.sqrt,
    'random': np.random.random,
    'randint': np.random.randint,
    # Dates. numpy datetime arithmetic works on arrays already, but needs a
    # literal written as `.astype('timedelta64[D]')`, has no month that keeps
    # its day, and no null. These say what they mean.
    'add_days': _add_days,
    'add_months': _add_months,
    'days_between': _days_between,
    'month_start': _month_start,
    'day_diff': _day_diff,
    'month_diff': _month_diff,
    'months_elapsed': _months_elapsed,
    'weekday': _weekday,
    'month_of': _month_of,
    'day_of': _day_of,
    'timestamp': _timestamp,
    'isnull': _isnull,
    'notnull': lambda x: ~_isnull(x),
    'coalesce': _coalesce,
}

# Standard operators to bypass simpleeval's string length checks which fail on numpy arrays
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
    ast.In: lambda a, b: a in b,
}

class SafeNumpy:
    """Proxy for numpy to allow safe access to whitelisted functions."""
    def __getattr__(self, name):
        if name in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[name]
        raise NameNotDefined(name, f"Function 'np.{name}' is not allowed in formulas.")


def _require_simpleeval() -> None:
    """Raise a clear error when formula support is requested without the extra dependency."""
    if simple_eval is None:
        raise ImportError(
            "Formula support requires simpleeval. "
            "Install with: pip install 'misata[formulas]'"
        )

class FormulaEngine:
    """
    Evaluate column formulas using safe expressions.

    Supports:
    - Simple arithmetic: duration * 10
    - Column references: quantity * unit_price
    - Cross-table references: @exercises.calories_per_minute
    - Conditional expressions: np.where(status == 'active', 1, 0)
    """

    def __init__(self, tables: Dict[str, pd.DataFrame]):
        """
        Initialize with generated tables for cross-table lookups.

        Args:
            tables: Dict mapping table names to DataFrames
        """
        self.tables = tables

    def evaluate(
        self,
        df: pd.DataFrame,
        formula: str,
        fk_column: Optional[str] = None,
    ) -> np.ndarray:
        """
        Evaluate a formula on a DataFrame.

        Args:
            df: DataFrame to evaluate on
            formula: Expression string
            fk_column: Foreign key column name for cross-table lookups

        Returns:
            Array of computed values
        """
        fk_mappings: Optional[Dict[str, str]] = None
        table_refs = re.findall(r'@(\w+)\.(\w+)', formula)
        if fk_column and table_refs:
            unique_tables = {table_name for table_name, _ in table_refs}
            if len(unique_tables) == 1:
                fk_mappings = {next(iter(unique_tables)): fk_column}
        return self.evaluate_with_lookups(df, formula, fk_mappings=fk_mappings)

    def _prepare_formula_context(
        self,
        df: pd.DataFrame,
        formula: str,
        fk_mappings: Optional[Dict[str, str]] = None,
    ) -> tuple[str, Dict[str, np.ndarray]]:
        """Resolve cross-table refs and return names to inject into evaluation."""
        fk_mappings = fk_mappings or {}
        pattern = r'@(\w+)\.(\w+)'
        matches = re.findall(pattern, formula)

        result = formula
        lookup_names: Dict[str, np.ndarray] = {}

        for i, (table_name, col_name) in enumerate(matches):
            if table_name not in self.tables:
                raise ValueError(f"Table '{table_name}' not found")

            ref_table = self.tables[table_name]

            if col_name not in ref_table.columns:
                raise ValueError(f"Column '{col_name}' not found in table '{table_name}'")

            # Resolve the parent's primary key (the column the FK points at). Real schemas
            # use `employee_id`, `customer_id`, etc. — not a literal `id` — so we detect the
            # actual key instead of assuming. Preference: explicit mapping value, then
            # `<singular>_id` / `<table>_id`, then a lone `id`.
            singular = table_name[:-1] if table_name.endswith("s") else table_name
            parent_key = None
            for cand in (f"{singular}_id", f"{table_name}_id", "id"):
                if cand in ref_table.columns:
                    parent_key = cand
                    break
            if parent_key is None:
                raise ValueError(
                    f"Reference table '{table_name}' has no resolvable primary key "
                    f"(looked for {singular}_id, {table_name}_id, id)"
                )

            # Resolve the FK column on THIS table that references the parent.
            # Preference: explicit mapping (authoritative, from the relationships),
            # then conventional FK names (`employee_id`, `employees_id`), then a
            # named parent key only if it is not the generic "id".
            #
            # We must NEVER fall back to a literal "id": a child's own primary key
            # is named "id", and matching it would join the child to the parent on
            # the child's PK (timesheets.id -> employees.id), silently producing
            # wrong values for every row whose id has no matching parent.
            fk_col = fk_mappings.get(table_name)
            if fk_col is None or fk_col not in df.columns:
                candidates = [f"{singular}_id", f"{table_name}_id"]
                if parent_key and parent_key != "id":
                    candidates.insert(0, parent_key)
                fk_col = None
                for cand in candidates:
                    if cand in df.columns:
                        fk_col = cand
                        break
            if fk_col is None or fk_col not in df.columns:
                raise ValueError(
                    f"No FK column on this table references '{table_name}' "
                    f"(looked for {singular}_id, {table_name}_id)"
                )

            ref_col = ref_table.set_index(parent_key)[col_name]
            looked_up = df[fk_col].map(ref_col.to_dict())
            # A missing value is 0 only for a number. A null timestamp filled
            # with 0 is 1 January 1970, and a null label filled with 0 is the
            # integer zero: an unconverted account's converted_at, read by a
            # child, became the epoch.
            if pd.api.types.is_numeric_dtype(ref_col) and not pd.api.types.is_bool_dtype(ref_col):
                looked_up = looked_up.fillna(0)
            elif pd.api.types.is_datetime64_any_dtype(ref_col):
                looked_up = pd.to_datetime(looked_up)
            var_name = f'_ref_{i}'
            lookup_names[var_name] = looked_up.values
            result = result.replace(f'@{table_name}.{col_name}', var_name)

        return result, lookup_names

    def _resolve_cross_table_refs(
        self,
        df: pd.DataFrame,
        formula: str,
        fk_column: Optional[str] = None,
    ) -> str:
        """
        Replace @table.column references with actual looked-up values.

        Pattern: @tablename.columnname

        Args:
            df: Current DataFrame
            formula: Formula with potential cross-table refs
            fk_column: FK column to use for lookups

        Returns:
            Formula with refs replaced by _lookup_N variables
        """
        fk_mappings = None
        matches = re.findall(r'@(\w+)\.(\w+)', formula)
        if fk_column and matches:
            unique_tables = {table_name for table_name, _ in matches}
            if len(unique_tables) == 1:
                fk_mappings = {next(iter(unique_tables)): fk_column}

        result, _ = self._prepare_formula_context(df, formula, fk_mappings)
        return result

    def evaluate_with_lookups(
        self,
        df: pd.DataFrame,
        formula: str,
        fk_mappings: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """
        Evaluate formula with automatic cross-table lookups.

        Args:
            df: DataFrame to evaluate on
            formula: Expression with @table.column references
            fk_mappings: Optional dict mapping table name to FK column name
                         e.g., {"exercises": "exercise_id", "products": "product_id"}

        Returns:
            Array of computed values
        """
        _require_simpleeval()
        fk_mappings = fk_mappings or {}

        # Pattern to match @table.column
        result, lookup_names = self._prepare_formula_context(df, formula, fk_mappings)
        names = {
            'np': SafeNumpy(),
            'pd': pd,
            # A null timestamp, for `where(..., nat, ended_at)`.
            'nat': np.datetime64('NaT', 'ns'),
        }

        # Add columns to context
        for col in df.columns:
            names[col] = df[col].values
        names.update(lookup_names)

        # Evaluate safely
        try:
            return np.array(simple_eval(
                result,
                names=names,
                functions=SAFE_FUNCTIONS,
                operators=SAFE_OPERATORS
            ))
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula '{formula}': {e}")


class FormulaColumn:
    """
    Definition of a formula-based column.
    """

    def __init__(
        self,
        name: str,
        formula: str,
        result_type: str = "float",
        fk_mappings: Optional[Dict[str, str]] = None,
    ):
        """
        Define a formula column.

        Args:
            name: Column name
            formula: Expression (can include @table.column refs)
            result_type: Type of result (int, float, boolean)
            fk_mappings: Map table names to FK column names
        """
        self.name = name
        self.formula = formula
        self.result_type = result_type
        self.fk_mappings = fk_mappings or {}

    def evaluate(
        self,
        df: pd.DataFrame,
        tables: Dict[str, pd.DataFrame],
    ) -> np.ndarray:
        """
        Evaluate this formula column.

        Args:
            df: Current table DataFrame
            tables: All generated tables for cross-table lookups

        Returns:
            Array of computed values
        """
        engine = FormulaEngine(tables)
        result = engine.evaluate_with_lookups(df, self.formula, self.fk_mappings)

        # Cast to result type
        if self.result_type == "int":
            return result.astype(int)
        elif self.result_type == "float":
            return result.astype(float)
        elif self.result_type == "boolean":
            return result.astype(bool)

        return result


def apply_formula_columns(
    df: pd.DataFrame,
    formulas: List[FormulaColumn],
    tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Apply formula columns to a DataFrame.

    Args:
        df: DataFrame to add columns to
        formulas: List of formula column definitions
        tables: All tables for cross-table lookups

    Returns:
        DataFrame with formula columns added
    """
    result = df.copy()

    for formula_col in formulas:
        result[formula_col.name] = formula_col.evaluate(result, tables)

    return result
