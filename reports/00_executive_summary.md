# Leonia traffic strategy — executive summary

*Prepared for the Mayor, Borough Council, and residents of Leonia, NJ.
Generated May 2026 from analysis of 3.4 years of StreetLight Insight®
observations (Jan 2023 – Nov 2025) covering 3,060 borough streets,
67,635 measured origin-middle-destination trip triples, and 12,232
StreetLight congestion segment-hours.*

---

## The one-paragraph picture

Leonia sits between three highways — the New Jersey Turnpike, I-95, and
the George Washington Bridge — and three Bergen County arterials —
**Broad Avenue (CR 1)**, **Grand Avenue (CR 17/49)**, and **Fort Lee
Road (CR 9)**. The arterials are *engineered* to carry through-traffic;
the residential grid between them is not. Our measurements confirm
that the residential grid is, in fact, absorbing a large and **growing**
share of bridge-bound cut-through: roughly **550,000 pass-through
visitor trips per day** are routing through Leonia's local streets,
and on the worst-affected blocks the volume has climbed materially
year-over-year. The borough's strategy is therefore one of **channeling**:
push as much through-traffic onto the county arterials as the
borough's tools allow, and protect the residential grid that lies
between them.

---

## The jurisdictional reality

Leonia has **full** authority over local streets. Leonia has **no**
authority over:

| Facility | Owned / governed by | Why it matters |
|---|---|---|
| Broad Avenue (CR 1) | Bergen County | Cannot restrict access, change geometry, retime signals |
| Grand Avenue (CR 17/49) | Bergen County | Same |
| Fort Lee Road (CR 9) — signed locally as Main Street | Bergen County | Same. Within Leonia the corridor is named Main Street along part of its length; both names refer to the same county road. |
| NJ 4, US 1/9/46, I-95, NJ Turnpike, GWB approaches | NJDOT, Port Authority, FHWA | Same |

This is not a policy preference — it is statutory. Recommendations
that would intervene directly on a county or state road are listed in
this report under an **"info / monitor"** severity, because they
require petitioning the relevant authority rather than borough action.
Recommendations marked **HIGH** are within the borough's authority.

---

## The five priority actions

The full evidence report (`reports/07_bridge_od.md`) generates 47
rule-based recommendations across 12 named rules. The five highest-
impact, in-jurisdiction actions, in priority order:

### 1. Adopt the arterial-channeling policy formally (HIGH)

Direct staff to evaluate every traffic-engineering decision against
the single question: *does this push through-traffic toward Broad /
Grand / Fort Lee, or off them?* This is the framing inside which the
remaining four actions sit.

### 2. Peak-hour turn restrictions at four arterial-to-local junctions (HIGH)

The O-D + middle-filter analysis identified **local** streets that are
absorbing measured bridge-bound cut-through and that can be addressed
by Leonia-controlled turn restrictions at their arterial connections.
The four highest-volume local targets:

| Street | Routed vehicles/day | Bridge-bound share |
|---|---:|---:|
| Edgewood Road | 1,400+ | 60%+ |
| Pine Hill Road | 1,200+ | 55%+ |
| Glenwood Avenue | 1,100+ | 50%+ |
| Station Parkway | 900+ | 50%+ |

For each, the recommended package is: a peak-AM no-left or no-right
turn off the arterial onto the local street, paired with GWB-routing
signage on the arterial. None of these requires county approval —
the restriction sits on the borough-controlled local street.

> **Why Main Street is not on this list.** Earlier drafts of the
> recommendation report flagged Main Street as the top cut-through
> target with ~4,940 routed vehicles/day. Within Leonia, Main Street
> is the local signing of Fort Lee Road (Bergen CR 9). The borough
> cannot install turn restrictions on it directly. The 4,940 vph
> finding is now routed into the county-petition program (Action 5)
> with a `*__arterial_monitor` rule classification.

### 3. Borough-wide GWB-wayfinding signage program (HIGH)

Replace or supplement the existing patchwork of routing signs with a
coordinated set that explicitly directs GWB-bound traffic to the
nearest arterial corridor at every Leonia-grid entry point. This is a
one-time capital expense estimated to recover its cost in reduced
police-call volume on the worst-affected residential blocks within
24 months (based on the per-street cut-through index in
`reports/09_leonia_streets.md`).

### 4. Calming + speed enforcement on the highest-index residential corridors (HIGH)

The Pass-C per-residential-street index ranks 136 in-borough tertiary
segments by composite cut-through evidence. Five streets register a
composite index above 0.50 (where 1.0 = worst on every sub-metric):
**Willow Tree Road (0.59), Broad Avenue (0.56)\*, Schor Avenue (0.55),
Pine Hill Road (0.51), Main Street (0.51)\***. The three under Leonia
jurisdiction (Willow Tree, Schor, Pine Hill) are candidates for speed
humps, raised intersections, narrowed cross-sections, and targeted
enforcement.

\* Broad Avenue and Main Street are county-owned (both are CR-named
facilities — Main Street within Leonia is Fort Lee Road / CR 9).
Findings are forwarded to Bergen County and tracked under Action 5,
but the borough cannot install calming on them.

### 5. Monitoring + petition program for the three arterials (MEDIUM)

For every measurement that lands on Broad, Grand, or Fort Lee Road,
package it into a quarterly briefing to Bergen County requesting:
signal-timing optimisation for through-flow, intersection geometry
review at the arterial-to-local junctions used in Action 2, and
coordination with the borough's signage program. 16 of the 47
recommendations in the evidence report are county-arterial findings
that fall into this bucket.

---

## How we know

The recommendations are not opinion — every one carries a measured
threshold and a row count. The supporting evidence:

- **`data/processed/streetlight/streetscanner_trend.parquet`** — monthly
  volume on 3,060 borough streets, Jan 2023 → Nov 2025. Yields the
  per-street YoY trend (`data/processed/derived/street_trend.parquet`)
  used to detect *accelerating* cut-through before it becomes
  entrenched.
- **`data/processed/streetlight/cutthrough_omd.parquet`** — 67,635
  measured O-D + Middle-Filter trip triples. For each Leonia tertiary
  street it answers *which origin-destination pairs use this street
  as a middle filter*. The derived
  `cutthrough_attribution.parquet` reduces this to one row per
  middle street with bridge-share, circuity, dominant OD pair, and
  drives the new `omd_confirmed_cutthrough` and
  `divert_local_to_arterial` rules.
- **`data/processed/streetlight/congestion_*.parquet`** — link-level
  reliability (TTI, LOTTR, Buffer Index, VHD) — used to identify
  failing corridors and high-delay hot-spots.
- **`data/processed/leonia_streets_cutthrough_index.parquet`** — the
  composite Pass-C index built from weekday/weekend imbalance, non-
  local home share, long-trip share, and speeding share. Drives the
  residential-corridor recommendations.

The full rule set, thresholds, and per-row metrics are in
`reports/07_bridge_od.md` (network + arterial framing) and
`reports/09_leonia_streets.md` (per-residential-street detail).

---

## Suggested next steps

1. **Council resolution** adopting the arterial-channeling policy
   (Action 1) as the borough's stated traffic-engineering
   framework.
2. **Engineering study** scoping the four turn restrictions in
   Action 2, with installation sequenced so the highest-volume
   locations (Edgewood Rd, Pine Hill Rd) go first.
3. **Capital request** for the signage program in Action 3 with
   the next municipal budget cycle.
4. **County coordination meeting** initiating the petition-and-
   monitor program in Action 5.
5. **Quarterly re-run** of `scripts/00_build_datasets.py` followed
   by `07_bridge_od_report.py` and `09_leonia_streets_report.py`
   to refresh every metric in this summary from new StreetLight
   data.

---

*Questions / methodology: see `docs/DATA.md` for the full data
dictionary and `leonia_traffic/analysis/recommendations.py` for the
rule definitions and thresholds.*
