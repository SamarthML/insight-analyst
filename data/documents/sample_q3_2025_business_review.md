# Northwind Analytics — Q3 2025 Quarterly Business Review

*SAMPLE DOCUMENT — synthetic data generated to exercise the ingestion pipeline.
Delete this file and drop in your real corpus.*

## Executive Summary

Northwind Analytics closed Q3 2025 with $42.8M in annual recurring revenue, up
18.4% year-over-year and 4.1% quarter-over-quarter. Growth remained ahead of
the 16% plan, but the composition shifted: net revenue retention carried more
of the quarter than new logo acquisition did, which is the central tension
going into Q4.

New ARR from new customers was $2.1M against a $2.9M target — a 28% miss. The
gap traces almost entirely to the mid-market segment, where the average sales
cycle stretched from 61 days in Q2 to 84 days in Q3. Enterprise closed at 103%
of target and SMB at 97%.

Net revenue retention finished at 114%, the fourth consecutive quarter above
110%. Gross retention held at 91%. Expansion was concentrated in the Insights
add-on, which attached to 34% of renewals versus 21% a year ago.

## Financial Performance

| Metric | Q3 2025 | Q2 2025 | Q3 2024 | YoY |
|---|---|---|---|---|
| ARR | $42.8M | $41.1M | $36.2M | +18.4% |
| New ARR | $2.1M | $2.6M | $2.4M | -12.5% |
| Net revenue retention | 114% | 112% | 108% | +6pts |
| Gross retention | 91% | 92% | 89% | +2pts |
| Gross margin | 78.2% | 77.4% | 74.9% | +3.3pts |
| CAC payback (months) | 19.4 | 16.8 | 15.2 | +4.2 |
| Rule of 40 | 31 | 34 | 29 | +2 |

Gross margin improved 330 basis points year-over-year, driven by the migration
of the ingestion tier off managed Kafka and onto self-hosted infrastructure,
completed in August. That project reduced cost of revenue by approximately
$1.4M annualized and is the single largest margin contributor of the year.

CAC payback deteriorated to 19.4 months from 15.2 a year ago. Two causes:
sales headcount grew 31% while new ARR contracted, and paid acquisition spend
rose 44% against a declining conversion rate. This is the metric the board
flagged as requiring a Q4 remediation plan.

## Segment Detail

### Enterprise

Enterprise revenue reached $24.3M ARR, 57% of total. Eleven new enterprise
logos closed, including three above $250K ACV. The segment's win rate against
Meridian Data improved to 44% from 37%, which the field attributes to the
compliance certifications completed in June (SOC 2 Type II, ISO 27001).

Average enterprise contract value rose to $198K from $171K. Multi-year
commitments now represent 61% of enterprise ARR, up from 48%, materially
improving revenue predictability.

The primary enterprise risk is concentration: the top ten accounts represent
28% of total ARR. Two of those ten are up for renewal in Q1 2026, together
worth $3.1M. Both are currently scored yellow on health.

### Mid-Market

Mid-market is where the quarter was lost. Revenue grew to $13.2M ARR but new
bookings fell 41% quarter-over-quarter. Post-mortem interviews with 22 lost
deals surfaced three recurring themes:

1. **Pricing perception.** Meridian Data introduced a usage-based tier in July
   priced roughly 30% below our entry plan. In 14 of 22 losses, price was cited
   as a primary factor, versus 5 of 19 in Q2.
2. **Time to value.** Prospects consistently raised the six-week implementation
   window. Competitors are advertising two weeks. Our own data shows median
   time to first dashboard is 27 days.
3. **Buying committee expansion.** The median number of stakeholders per deal
   rose from 3.2 to 5.1, adding a security review step that did not previously
   appear in mid-market cycles.

### SMB

SMB contributed $5.3M ARR, growing 22% year-over-year on a self-serve motion
that now accounts for 71% of segment acquisition. Trial-to-paid conversion held
at 8.9%. Support cost per SMB account fell 19% following the knowledge base
rebuild in July.

## Product

Three significant releases shipped in Q3.

**Insights add-on (GA, July).** Adoption exceeded plan: 34% renewal attach rate
against a 25% target, contributing $1.6M in expansion ARR. Weekly active usage
among attached accounts is 63%, which is healthy for a first quarter of GA.

**Query engine v3 (GA, August).** Median query latency dropped from 2.4s to
0.7s on the p50 and from 11.2s to 3.8s on the p95. Support tickets tagged
"slow" fell 67% in the four weeks following rollout.

**Collaborative workspaces (beta, September).** In beta with 47 accounts.
Engagement is encouraging — 71% weekly active among enrolled accounts — but the
permissions model has generated 23 bugs, 6 of them severity 1. GA has slipped
from Q4 2025 to Q1 2026.

## Headcount and Operations

Headcount ended at 284, up from 241 at the start of the year. Engineering is
118, go-to-market 97, customer success 41, and general and administrative 28.

Voluntary attrition ran at 11% annualized, below the 14% benchmark. However,
attrition among engineers with more than three years of tenure was 19%, and
exit interviews repeatedly cited on-call burden. The infrastructure team has
proposed a rotation redesign for Q4.

Two searches remain open and material: VP of Mid-Market Sales (open 94 days)
and Principal Security Engineer (open 71 days).

## Q4 2025 Priorities

1. **Restore mid-market new ARR.** Ship a usage-based entry tier by November 15
   to neutralize the Meridian pricing move. Target: $3.4M new ARR in Q4.
2. **Compress time to value.** Reduce median time to first dashboard from 27
   days to under 14 through guided onboarding and prebuilt templates.
3. **Fix CAC payback.** Freeze paid acquisition spend at Q3 levels and reallocate
   toward partner-sourced pipeline, which currently converts at 2.3x the rate of
   paid at roughly half the cost.
4. **De-risk the Q1 renewals.** Executive sponsor assigned to each of the two
   yellow-health top-ten accounts, with a remediation plan by October 15.
5. **Stabilize collaborative workspaces.** Clear all severity 1 permission bugs
   before expanding the beta beyond 47 accounts.

## Risks

| Risk | Likelihood | Impact | Owner |
|---|---|---|---|
| Meridian price pressure spreads to enterprise | Medium | High | CRO |
| Top-ten renewal churn in Q1 2026 | Medium | High | CCO |
| Engineering attrition accelerates | Medium | Medium | CTO |
| Workspaces GA slips past Q1 2026 | High | Medium | CPO |
| CAC payback exceeds 24 months | Low | High | CFO |
