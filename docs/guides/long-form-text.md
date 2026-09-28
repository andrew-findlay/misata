---
title: "Text Realism & Enrichment: Domain Microtext, Sentiment Alignment, and enrich_text"
description: "Generate realistic domain-authentic text for support tickets, clinical notes, transaction memos, error messages, and product descriptions with zero Lorem Ipsum."
---

# Text Realism & Enrichment

Misata generates domain-authentic, context-conditioned text across more than 15 real-world industry domains. It does not emit Latin placeholder gibberish ("Lorem Ipsum"), robotic single-sentence templates, or repetitive filler.

Every textual generator uses combinatorial grammar rules, seeded domain vocabulary pools, and cross-column context (e.g. sentiment aligned with star ratings, product descriptions conditioned on item categories, and clinical progress notes matching patient domains).

---

## ⚡ Quick Drop-in: `misata.enrich_text()`

Have an existing pandas DataFrame, Series, or list of values with blank or boring text columns? Enrich them in one line with zero boilerplate:

```python
import misata
import pandas as pd

# 1. Enrich an entire DataFrame with automatic semantic detection
df = pd.read_csv("support_cases.csv")
df_enriched = misata.enrich_text(df, seed=42)

# 2. Enrich a specific Series or column with an explicit text type
df["closing_notes"] = misata.enrich_text(
    df["closing_notes"],
    text_type="resolution_notes",
    seed=42,
)

# 3. Enrich a Python list or array
memos = misata.enrich_text([""] * 50, text_type="transaction_memo", seed=42)
```

### Supported `text_type` Arguments

| `text_type` | Vertical | Description & Sample Output |
|---|---|---|
| `ticket_subject` | SaaS / IT | Short one-line issue subject: `"Webhook 504 gateway timeout under load"` |
| `support_ticket` / `ticket_body` | SaaS / IT | Multi-line customer issue description with symptoms and urgency |
| `resolution_notes` | Support / Ops | Agent closing and resolution actions: `"Investigated root cause; patched deployment rollout"` |
| `transaction_memo` | FinTech / Banking | Bank statement descriptor: `"ACH DIRECT DEBIT - PAYROLL EXPENSE ID:98124"` |
| `error_message` | IT / DevOps | Realistic system exception: `"Connection pool exhausted (50/50 connections active)"` |
| `clinical_notes` | Healthcare | Clinical SOAP progress notes: History of present illness, examination, assessment & plan |
| `chief_complaint` | Healthcare | Patient presenting symptom: `"Acute worsening dyspnea and productive cough for 3 days"` |
| `discharge_instructions` | Healthcare | Post-op & care instructions: `"Follow up with primary care physician in 7-10 days; resume low-sodium diet"` |
| `delivery_instructions` | Logistics | Driver delivery directions: `"Gate code #4492, leave behind planter by front door"` |
| `return_reason` | E-Commerce | Customer return justification: `"Item arrived defective or inoperative"` |
| `churn_reason` | SaaS / B2B | Customer cancellation reason: `"Switched to competitor with native enterprise SSO"` |
| `audit_reason` | Security / FinTech | Compliance / access override: `"Emergency access override authorized by sec ops lead"` |
| `customer_feedback` | UGC / CX | Qualitative feedback: `"Support team was quick to respond, but onboarding docs need refresh"` |
| `product_description` | E-Commerce | Multi-sentence description conditioned on category (electronics, fashion, home, etc.) |
| `review` | E-Commerce / UGC | Multi-sentence review whose sentiment strictly agrees with its star rating |
| `address` | Geospatial | Complete street address with realistic secondary units (Apt, Suite, Bldg, Fl) |
| `caption` | Social Media | Contextual lifestyle caption with emoji and relevant hashtags |
| `bio` | Social Media | Concise user headline and background |
| `email_body` | Communications | Realistic business email with greeting, body, and sign-off |

---

## Domain Showcase & Sample Outputs

### 1. B2B SaaS & IT Helpdesk

```python
# Schema definition
columns = {
    "tickets": [
        Column(name="subject", type="text", distribution_params={"text_type": "ticket_subject"}),
        Column(name="description", type="text", distribution_params={"text_type": "support_ticket"}),
        Column(name="resolution_notes", type="text", distribution_params={"text_type": "resolution_notes"}),
    ]
}
```

**Generated Samples:**
- **Subject**: `"Webhook 504 gateway timeout under load"`
- **Description**: `"The dashboard isn't loading any data since this morning. Several team members reported blank screens across both Chrome and Firefox."`
- **Resolution**: `"Root cause identified as database connection pool saturation. Deployed query index optimization and verified response latency normalized."`

### 2. FinTech, Banking & Payments

- **Transaction Memo**: `"SWIFT WIRE REMITTANCE - INV #84920 REF:77391"`
- **Audit Reason**: `"Quarterly compliance review sign-off completed by chief risk officer"`
- **Account Statement Descriptor**: `"MERCHANT SETTLEMENT - STRIPE PAYMENTS BATCH #2910"`

### 3. Healthcare & Clinical Informatics

- **Chief Complaint**: `"Persistent bilateral knee pain exacerbated by weight-bearing for 2 weeks"`
- **Discharge Summary**: `"Keep surgical incision clean and dry for 48 hours. Take prescribed antibiotics with food. Call clinic immediately if fever exceeds 101F."`
- **Progress Note (SOAP)**: `"Patient presents for follow-up of hypertension. Vital signs stable. Medication well-tolerated. Continued current dosing with repeat lab work ordered."`

### 4. E-Commerce & Retail

Category-conditioned product descriptions automatically inspect the table's `category` column (or explicit parameter) to produce specs matching the product vertical:

```python
# Electronics category:
# "Engineered with high-grade aluminum and precision components. Features advanced Bluetooth 5.3 connectivity with active noise cancellation and 30-hour battery life. Includes braided charging cable and hardshell travel case."

# Apparel category:
# "Crafted from breathable organic cotton blend with reinforced double-needle stitching. Designed with a modern tailored silhouette suitable for all-day comfort. Machine washable and pre-shrunk to retain fit."
```

- **Return Reason**: `"Ordered wrong size or variant"`
- **Delivery Instructions**: `"Leave inside enclosed porch if raining, otherwise place on front step"`

### 5. Sentiment-Calibrated Customer Reviews

When a dataset includes both a rating column (1 to 5) and a review column, Misata couples the text generator to the rating value:
- **1 Star**: `"Extremely frustrated. Arrived damaged and customer service has not answered my emails for three days."`
- **3 Stars**: `"Average quality for the price. Works as advertised but setup instructions were confusing."`
- **5 Stars**: `"Exceptional build quality and exceeded all my expectations. Will definitely purchase again."`

---

## Using in Schemas

### Dict Schema
```python
schema = {
    "support_tickets": {
        "__rows__": 500,
        "id": {"type": "integer", "primary_key": True},
        "subject": {"type": "string", "text_type": "ticket_subject"},
        "resolution": {"type": "string", "text_type": "resolution_notes"},
        "error_trace": {"type": "string", "text_type": "error_message"},
        "churn_reason": {"type": "string", "text_type": "churn_reason"},
    }
}
tables = misata.generate_from_schema(misata.from_dict_schema(schema, seed=42))
```

### YAML Schema
```yaml
tables:
  incidents:
    rows: 1000
    columns:
      incident_id: { type: int, unique: true }
      error_message: { type: text, text_type: error_message }
      resolution_notes: { type: text, text_type: resolution_notes }
      audit_reason: { type: text, text_type: audit_reason }
```
