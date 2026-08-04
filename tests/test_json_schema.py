"""JSON Schema sanity tests.

These guard the contract between misata's generated YAML and the
``misata.schema.json`` published for IDE autocomplete.  If a domain
schema is changed (new text_type, new column param, new domain enum,
etc.) without updating the JSON Schema, this file fails loudly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

import misata

jsonschema = pytest.importorskip("jsonschema")
from jsonschema import Draft202012Validator


# ---------------------------------------------------------------------------
# Static schema invariants
# ---------------------------------------------------------------------------


def test_json_schema_loads():
    schema = misata.json_schema()
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert "title" in schema
    assert "tables" in schema["properties"]


def test_json_schema_is_self_valid_against_meta_schema():
    schema = misata.json_schema()
    # Will raise if our schema is itself malformed
    Draft202012Validator.check_schema(schema)


def test_template_validates_clean():
    """The bundled misata.yaml template must always validate."""
    schema = misata.json_schema()
    template = yaml.safe_load(misata.MISATA_YAML_TEMPLATE)
    errors = list(Draft202012Validator(schema).iter_errors(template))
    assert errors == [], "\n".join(str(e) for e in errors)


def test_domain_enum_matches_story_parser():
    """Every domain the StoryParser detects must be listed in the JSON Schema."""
    from misata.story_parser import StoryParser

    schema = misata.json_schema()
    schema_domains = set(schema["properties"]["domain"]["enum"])
    parser_domains = set(StoryParser.DOMAIN_KEYWORDS.keys()) | {"generic"}

    missing = parser_domains - schema_domains
    assert not missing, (
        f"Domains exist in StoryParser but not in misata.schema.json: {missing}. "
        "Add them to schema/misata.schema.json domain enum."
    )


# ---------------------------------------------------------------------------
# Round-trip: every domain must produce a YAML that validates
# ---------------------------------------------------------------------------


_DOMAIN_STORIES = {
    "saas": "A SaaS company with 5k users and 20% churn",
    "ecommerce": "An ecommerce store with 10k orders",
    "fintech": "A fintech with payments and fraud detection",
    "healthcare": "A healthcare clinic with patients and doctors",
    "marketplace": "A freelance marketplace with sellers and buyers",
    "logistics": "A logistics fleet with drivers and shipments",
    "hr": "An HR system with employees and payroll",
    "social": "A social media app with influencers and reels",
    "realestate": "A real estate platform with property listings",
    "pharma": "A pharma research company with clinical trials",
    "fooddelivery": "A food delivery app with restaurants and couriers",
    "edtech": "An edtech platform with courses and quizzes",
    "gaming": "A gaming platform with players and achievements",
    "crm": "A CRM with contacts and deals pipeline",
    "crypto": "A crypto exchange with wallets and blockchain transactions",
    "insurance": "An insurance company with policies and claims",
    "travel": "A travel booking platform with hotels and flights",
    "streaming": "A Netflix streaming service with subscribers",
}


@pytest.mark.parametrize("domain,story", list(_DOMAIN_STORIES.items()))
def test_each_domain_yaml_validates(domain, story, tmp_path):
    """Generate a schema for every domain, save as YAML, validate against JSON Schema.

    This is the strong guarantee that misata.yaml files written by
    `misata init` for any domain will be accepted by IDEs that consume
    the published JSON Schema.
    """
    schema = misata.parse(story)
    path = tmp_path / f"{domain}.yaml"
    misata.save_yaml_schema(schema, path)

    yaml_doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    json_schema = misata.json_schema()

    errors = sorted(
        Draft202012Validator(json_schema).iter_errors(yaml_doc),
        key=lambda e: list(e.absolute_path),
    )
    assert errors == [], (
        f"Domain '{domain}' YAML failed JSON Schema validation:\n"
        + "\n".join(
            f"  - {list(e.absolute_path)}: {e.message[:200]}" for e in errors[:10]
        )
    )


class TestTableNestedDirectivesValidate:
    """The engine accepts __group_shares__ (and its siblings) nested inside
    the table dict it describes, the way __correlations__ is written -- the
    published JSON Schema used to reject that placement outright (every
    directive lived only under $defs/table's now-extended property list),
    which would have made `misata lint` fail a document the engine itself
    generates correctly."""

    def test_group_shares_nested_in_its_table_validates(self):
        schema = misata.json_schema()
        doc = {
            "name": "nested_test",
            "tables": {
                "sales": {
                    "rows": 900,
                    "columns": {
                        "category": {"type": "string", "enum": ["Electronics", "Home", "Apparel"]},
                        "revenue": {"type": "float", "min": 10, "max": 500},
                    },
                    "group_shares": [{
                        "table": "sales", "measure": "revenue", "group_column": "category",
                        "shares": {"Electronics": 0.5, "Home": 0.3, "Apparel": 0.2},
                    }],
                }
            },
        }
        errors = list(Draft202012Validator(schema).iter_errors(doc))
        assert errors == [], "\n".join(str(e) for e in errors)

    def test_lifecycles_nested_in_its_table_validates(self):
        schema = misata.json_schema()
        doc = {
            "name": "nested_test",
            "tables": {
                "orders": {
                    "rows": 500,
                    "columns": {
                        "status": {"type": "string"},
                        "placed_at": {"type": "date"},
                    },
                    "lifecycles": [{
                        "table": "orders", "name": "order_flow",
                        "state_column": "status", "start_column": "placed_at",
                        "states": [{"name": "placed"}, {"name": "shipped"}],
                        "transitions": [["placed", "shipped"]],
                        "initial": "placed",
                    }],
                }
            },
        }
        errors = list(Draft202012Validator(schema).iter_errors(doc))
        assert errors == [], "\n".join(str(e) for e in errors)


# ---------------------------------------------------------------------------
# The published schema must accept everything the loader accepts
#
# 0.9.5 taught `_parse_relationship` the long spelling and four extra fields,
# but the JSON Schema kept `additionalProperties: false` over the short one, so
# it rejected the exact form the release added — and the scaffold ships a
# `yaml-language-server: $schema=` header pointing at it, which means an editor
# underlines valid config. `misata lint` passed the whole time, because lint
# runs the feasibility check and never consults this file.
# ---------------------------------------------------------------------------


def _errors(doc):
    return sorted(
        Draft202012Validator(misata.json_schema()).iter_errors(doc),
        key=lambda e: list(e.absolute_path),
    )


def _minimal(**extra):
    doc = {
        "name": "T",
        "tables": {
            "parents": {"rows": 10, "columns": {"id": {"type": "int"}}},
            "children": {
                "rows": 10,
                "columns": {"parent_id": {"type": "int"}, "amount": {"type": "float"}},
            },
        },
    }
    doc.update(extra)
    return doc


@pytest.mark.parametrize(
    "relationship",
    [
        pytest.param({"parent": "parents", "child": "children"}, id="short"),
        pytest.param({"parent_table": "parents", "child_table": "children"}, id="long"),
        pytest.param(
            {
                "parent_table": "parents", "parent_key": "id",
                "child_table": "children", "child_key": "parent_id",
                "temporal": True, "min_children": 1,
            },
            id="long-with-min-children",
        ),
        pytest.param(
            {
                "parent_table": "parents", "child_table": "children",
                "filters": {"status": "active"},
            },
            id="filters-scalar",
        ),
        pytest.param(
            {
                "parent_table": "parents", "child_table": "children",
                "filters": {"status": ["shipped", "completed"]},
            },
            id="filters-membership",
        ),
        pytest.param(
            {
                "parent_table": "parents", "child_table": "children",
                "partition_by": ["tenant_id"],
            },
            id="partition-by",
        ),
        pytest.param(
            {
                "parent_table": "parents", "child_table": "children",
                "parent_time": "created_at", "child_time": "order_date",
                "child_time_table": "orders",
            },
            id="temporal-eligibility",
        ),
    ],
)
def test_schema_accepts_every_relationship_form_the_loader_does(relationship):
    assert _errors(_minimal(relationships=[relationship])) == []


def test_schema_still_requires_a_parent_and_a_child():
    """The permissiveness above must not extend to a relationship naming neither."""
    assert _errors(_minimal(relationships=[{"parent_key": "id"}])) != []


@pytest.mark.parametrize(
    "point_keys",
    [
        pytest.param(("period", "value"), id="documented"),
        pytest.param(("date", "target_value"), id="date-and-target_value"),
        pytest.param(("date", "value"), id="date-and-value"),
        pytest.param(("period", "amount"), id="period-and-amount"),
    ],
)
def test_schema_accepts_every_curve_point_spelling_the_engine_does(point_keys):
    """fact_engine reads TARGET_KEYS = (target_value, value, target, amount).

    The schema documented `value` alone and required `period`, so a curve using
    `date` — which is the only way to name a bucket outside 1-12, since `month`
    is a calendar month — was reported invalid while generating perfectly.
    """
    name_key, value_key = point_keys
    name = "2024-01-01" if name_key == "date" else "2024-01"
    doc = _minimal(
        outcome_curves=[{
            "table": "children",
            "column": "amount",
            "curve_points": [
                {name_key: name, value_key: 100},
                {name_key: "2024-02-01" if name_key == "date" else "2024-02", value_key: 200},
            ],
        }]
    )
    assert _errors(doc) == []
