# Realism benchmark

Misata's default mode never sees real data, so "realistic" cannot be shown the
usual way: fit a model to a dataset, then score it against that dataset. This
benchmark asks the question a user actually faces. Knowing only the shape of
a business (its tables, its column names and types, roughly how many rows),
how close does each tool get to what real data of that kind looks like?

Two public datasets, each split in half:

- **Olist**: real orders from a Brazilian marketplace (2016–2018), 30,000
  orders sampled, with customers, products and payments.
- **NYC taxis**: the seaborn sample of March 2019 trips, 6,389 rides.

Contestants:

| | Sees real data? | What it is given |
|---|---|---|
| `misata_story` | no | one sentence naming the business and its row counts |
| `misata_schema` | no | table and column names and types, the category labels (not their shares), no parameters |
| `faker_script` | no | the script people write: Faker dates, numpy uniform amounts, uniform foreign keys |
| `misata_mimic` | train half | `misata.mimic()` on the real rows |
| `sdv_copula` | train half | SDV's Gaussian copula on the real rows |
| `real_train` | train half | the real train half itself: the noise floor |

Everything is scored against the held-out test half, on scale-free metrics so
a blind generator is not punished for not knowing the currency:

| Metric | Measures | Better |
|---|---|---|
| Amount shape | KS distance between `log(amount / median)` distributions | lower |
| Hour / weekday profile | total variation distance between the histograms | lower |
| Customer / product fan-out | difference in the Gini of children per parent | lower |
| Category balance | difference in normalised entropy (how even, not which labels) | lower |
| Detection AUC | a gradient-boosted classifier telling real from synthetic rows on amount shape, time of day and weekday; 0.5 is indistinguishable | lower |
| Tells score | [`realism_report`](realism.md) on the synthetic tables | higher |

## Results

Misata 0.9.7, seed 7. Reproduce with
`python -m benchmarks.realism_bench --cache .bench_cache`.

### Olist (30,000 orders)

| Metric | `real_train` | `misata_story` | `misata_schema` | `faker_script` | `misata_mimic` | `sdv_copula` |
|---|---|---|---|---|---|---|
| Amount shape (KS) | 0.008 | 0.169 | 0.051 | 0.191 | 0.024 | 0.072 |
| Hour profile (TVD) | 0.018 | 0.149 | 0.145 | 0.289 | 0.017 | 0.286 |
| Weekday profile (TVD) | 0.010 | 0.046 | 0.048 | 0.054 | 0.009 | 0.057 |
| Customer fan-out (ΔGini) | 0.001 | n/a | 0.256 | 0.237 | n/a | n/a |
| Product fan-out (ΔGini) | 0.003 | 0.440 | 0.028 | 0.149 | n/a | n/a |
| Category balance (Δ) | 0.003 | n/a | 0.481 | 0.500 | 0.001 | 0.003 |
| Detection AUC | 0.498 | 0.767 | 0.738 | 0.794 | 0.541 | 0.709 |
| Tells score ↑ | 1.000 | 0.942 | 1.000 | 0.333 | 1.000 | 0.750 |

### NYC taxis (6,389 trips)

| Metric | `real_train` | `misata_story` | `misata_schema` | `faker_script` | `misata_mimic` | `sdv_copula` |
|---|---|---|---|---|---|---|
| Amount shape (KS) | 0.015 | 0.225 | 0.251 | 0.217 | 0.025 | 0.135 |
| Hour profile (TVD) | 0.039 | 0.237 | 0.082 | 0.193 | 0.060 | 0.189 |
| Weekday profile (TVD) | 0.026 | 0.074 | 0.030 | 0.033 | 0.054 | 0.049 |
| Category balance (Δ) | 0.003 | n/a | 0.130 | 0.140 | 0.009 | 0.003 |
| Detection AUC | 0.504 | 0.887 | 0.911 | 0.886 | 0.663 | 0.828 |
| Tells score ↑ | 1.000 | 1.000 | 0.667 | 0.400 | 1.000 | 0.667 |

## What it says

**On e-commerce, blind Misata beats the script and approaches a model fitted
to the data.** With table and column names and the domain, `misata_schema`
reaches a detection AUC of 0.74 against SDV's 0.71 (SDV saw 15,000 real
orders), gets the shape of order amounts closer than SDV (0.05 vs 0.07), and
matches real product popularity almost exactly (ΔGini 0.03), which the
uniform script misses by five times as much. One qualification: the
e-commerce amount prior was set from Olist's public data when Misata's priors
were built, so the amount row is not a blind result. Hours, weekdays and
fan-out are.

**On taxis, the hour rhythm is now right; fares are not.** Declaring the
domain as `transport` gives the night-heavy demand curve ride data has (hour
TVD 0.08, better than SDV fitted on the data at 0.19). Fares are still drawn
from a generic money shape that is wider than taxi totals, so the classifier
still separates Misata from real trips more easily than it separates the
script (AUC 0.91 vs 0.89). That is the remaining miss here, and it is left
in rather than fixed with a taxi-specific fare prior fitted to the same
public data.

**Customer fan-out is better, not solved.** In this Olist sample nearly every
buyer buys once (30,000 orders from 29,651 customers). Declaring the domain
as `marketplace` makes person-like parents (customers, buyers, guests) mildly
weighted while products stay concentrated, which cut the gap from 0.39 to
0.26. Declaring `min_children: 1` on the relationship, which says what this
sample says (every customer in it has ordered), closes it: ΔGini 0.000. That
declaration did not work before 0.9.7 for child tables larger than one
10,000-row batch; it now counts coverage across batches.

**`mimic` is the strongest fitted generator here.** It beats SDV on every
metric except weekday profile and category balance on taxis, and is close to
indistinguishable on Olist (AUC 0.54).

**The tells check calibrates on real data.** Both real datasets score 1.0 on
`realism_report`.

## Changes between runs, and why

The first published run (0.9.6.60 defaults) found five problems, and the
fixes are in 0.9.7. To keep the benchmark honest about what it tests:

- **Bugs, fixed generally:** `mimic` dropped the time of day; story `*_at`
  columns were dates; the fan-out tell flagged real data; product names were
  not reproducible across processes. None of these fixes look at the
  benchmark data.
- **Domain knowledge added after the benchmark exposed its absence:** a
  night-heavy hour curve for `transport`/`taxi`/`mobility`/`nightlife`
  domains, and mild person fan-out for `marketplace`/`travel`/`realestate`.
  These are general facts about those businesses, not parameters fitted to
  these samples, but they were added because these samples showed they were
  missing. Treat the taxi hour row and the Olist customer row as validation
  of a fix, not as a blind first measurement.
- **The Olist blind schema now declares `marketplace`** (it declared
  `ecommerce` before), because that is what Olist is. With the same
  declaration as before, every number but customer fan-out is unchanged.

## Caveats

- Two datasets, both transactional. This is evidence about orders and trips,
  not about every domain.
- The detection classifier sees three scale-free features. A classifier with
  more features (text, cross-table joins) would separate every contestant
  more easily.
- Blind contestants get the row counts and, for the schema variant, the
  category labels. They never see shares, ranges or any row.
- Single seed. Run with `--seed` to check stability.
