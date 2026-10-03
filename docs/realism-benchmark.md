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

Misata 0.9.6.60, seed 7. Reproduce with
`python -m benchmarks.realism_bench --cache .bench_cache`.

### Olist (30,000 orders)

| Metric | `real_train` | `misata_story` | `misata_schema` | `faker_script` | `misata_mimic` | `sdv_copula` |
|---|---|---|---|---|---|---|
| Amount shape (KS) | 0.008 | 0.169 | 0.051 | 0.191 | 0.024 | 0.072 |
| Hour profile (TVD) | 0.018 | 0.149 | 0.145 | 0.289 | 0.017 | 0.286 |
| Weekday profile (TVD) | 0.010 | 0.046 | 0.048 | 0.054 | 0.009 | 0.057 |
| Customer fan-out (ΔGini) | 0.001 | n/a | 0.387 | 0.237 | n/a | n/a |
| Product fan-out (ΔGini) | 0.003 | 0.440 | 0.028 | 0.149 | n/a | n/a |
| Category balance (Δ) | 0.003 | n/a | 0.481 | 0.500 | 0.001 | 0.003 |
| Detection AUC | 0.498 | 0.775 | 0.738 | 0.794 | 0.541 | 0.709 |
| Tells score ↑ | 1.000 | 0.938 | 1.000 | 0.333 | 1.000 | 0.750 |

### NYC taxis (6,389 trips)

| Metric | `real_train` | `misata_story` | `misata_schema` | `faker_script` | `misata_mimic` | `sdv_copula` |
|---|---|---|---|---|---|---|
| Amount shape (KS) | 0.015 | 0.225 | 0.251 | 0.217 | 0.025 | 0.135 |
| Hour profile (TVD) | 0.039 | 0.237 | 0.239 | 0.193 | 0.060 | 0.189 |
| Weekday profile (TVD) | 0.026 | 0.074 | 0.070 | 0.033 | 0.054 | 0.049 |
| Category balance (Δ) | 0.003 | n/a | 0.130 | 0.140 | 0.009 | 0.003 |
| Detection AUC | 0.504 | 0.887 | 0.916 | 0.886 | 0.663 | 0.828 |
| Tells score ↑ | 1.000 | 1.000 | 0.833 | 0.400 | 1.000 | 0.667 |

## What it says

**On e-commerce, blind Misata beats the script and approaches a model fitted
to the data.** With nothing but table and column names, `misata_schema`
reaches a detection AUC of 0.74 against SDV's 0.71 (SDV saw 15,000 real
orders), gets the shape of order amounts closer than SDV does (0.05 vs 0.07),
and matches real product popularity almost exactly (ΔGini 0.03), which the
uniform script misses by five times as much.

**On taxis, it does not.** Misata's default daily rhythm is daytime-weighted,
which suits orders and signups but not taxi demand, which runs into the
night; its default amount prior is wider than taxi fares. The plain script,
with uniform hours, lands closer on both. These defaults were not tuned to
this benchmark, deliberately: tuning them to the test set would make the
numbers meaningless. A user who knows taxi traffic peaks at night can say so
(`hour_weights` on the column); the default does not know it.

**Customer fan-out is a real miss.** In this Olist sample nearly every
buyer buys once (30,000 orders from 29,651 customers). Misata's default popularity weighting assumes repeat buyers
(Gini about 0.55 when there are several orders per customer), so at roughly
one order per customer it still produces too many repeat customers.

**`mimic` is now the strongest fitted generator here.** It beats SDV on
every metric except weekday profile and category balance on taxis, and is
close to indistinguishable on Olist (AUC 0.54). That required a fix this benchmark found: `mimic` used to
profile timestamps as calendar dates, so every mimicked order landed at
midnight (AUC 1.0). It now keeps the time of day and learns the hour and
weekday shares.

**The tells check calibrates on real data.** Both real datasets score 1.0 on
`realism_report`. That too required a fix the benchmark found: the fan-out
check flagged Olist's real orders as fake, because with about one order per
customer the counts are necessarily even.

## Caveats

- Two datasets, both transactional. This is evidence about orders and trips,
  not about every domain.
- The detection classifier sees three scale-free features. A classifier with
  more features (text, cross-table joins) would separate every contestant
  more easily.
- Blind contestants get the row counts and, for the schema variant, the
  category labels. They never see shares, ranges or any row.
- Single seed. Run with `--seed` to check stability.
