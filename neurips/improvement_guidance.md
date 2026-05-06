# NTD-PL NeurIPS Improvement Guide

This document summarizes the next revision priorities for the NTD-PL paper.
The intended positioning is:

> NTD-PL is a parameter-tied Tucker rank-lift layer: it enhances a fixed Tucker
> backbone through a shared nonlinear observation link, rather than acting as a
> generic nonlinear tensor decoder or a replacement for rank selection.

## Core Revision Goal

Make the paper more compelling by turning the current scoped claim into a
stronger mechanism story:

1. Standard Tucker is often limited by a linear observation map under a tight
   rank budget.
2. NTD-PL adds only a few scalar link parameters, but induces tied high-order
   Tucker interactions through powers of the same latent tensor.
3. The resulting gain is not explained by ordinary rank increase alone, and a
   fixed-backbone beta refresh can be used as a lightweight diagnostic for
   when joint fitting should help.
4. The method is useful precisely when the Tucker residual contains a coherent
   shared entrywise nonlinear component.

## 1. Reframe the Main Claim Around Tucker Enhancement

Current framing: NTD-PL is a nonlinear Tucker decomposition with a polynomial
link.

Recommended framing: NTD-PL is a Tucker enhancer, or a parameter-tied rank-lift
of Tucker.

Suggested introduction language:

> NTD-PL is best viewed as a parameter-tied Tucker rank lift, not as a generic
> nonlinear decoder. Powers of one learned Tucker latent tensor induce
> Veronese-lifted Tucker factors, while the only new free parameters across
> degrees are scalar link coefficients.

Writing changes:

- Emphasize "same Tucker backbone, richer observation map" early.
- Avoid implying that NTD-PL replaces rank tuning or broad nonlinear tensor
  models.
- Use "Tucker enhancement", "shared-backbone rank lift", and "joint linked
  fitting" consistently.
- Make the boundary condition explicit: NTD-PL helps when the matched-rank
  Tucker residual has a coherent shared nonlinear component.

## 2. Add a Parameter-Efficiency / Rank-Lift Main Figure

The existing rank-lift table is one of the strongest pieces of evidence and
should become central rather than auxiliary.

Recommended experiment/figure:

- Plot reconstruction error versus parameter count or Tucker spatial rank.
- Include:
  - matched-rank Tucker,
  - matched-rank NTD-PL,
  - rank-lifted Tucker sweep,
  - optionally the first Tucker rank that matches NTD-PL RMSE.
- Show both RMSE and SAM, either as two panels or as paired curves.

Main message:

> Tucker can buy back part of the RMSE gap by increasing rank, but NTD-PL reaches
> comparable RMSE with substantially fewer parameters and still preserves a SAM
> advantage at the RMSE-matching Tucker ranks.

Use the existing signal from `rank_lift_tucker_full_sweep.tex`:

- Tucker needs roughly 51%-60% more parameters to match the NTD-PL RMSE.
- At those RMSE-matching ranks, NTD-PL still has better SAM.

Deliverables:

- A compact main-text figure or table.
- A paragraph tying the result directly to the tied-rank-lift theory.
- Keep the full sweep in the appendix if the main text uses a compact version.

## 3. Recast PolyCal as a Fixed-Backbone Beta Diagnostic

This check is still useful, but it should be framed as a restricted NTD-PL
update rather than as generic post-processing.

Recommended interpretation:

- Freeze the Tucker core and factors.
- Update only the scalar polynomial coefficients.
- Measure how much RMSE drops before any backbone reshaping.

Main message:

> If a frozen Tucker backbone already admits a useful scalar-link update, then
> the scene is a good candidate for larger gains from full joint NTD-PL fitting.

## 4. Add a Real-Data Residual Diagnostic

The paper currently says NTD-PL helps when the Tucker residual contains a shared
nonlinear component. Add a diagnostic that directly measures this condition.

Recommended diagnostic:

1. Fit matched-rank Tucker to each real tensor.
2. Compute the Tucker output or latent reconstruction `S_T`.
3. Compute the residual `Y - S_T`.
4. Fit a scalar map from `S_T` to either `Y` or the residual.
5. Report how much residual energy is explained by this scalar map.
6. Correlate this diagnostic with NTD-PL gain across scenes/datasets.

Possible metrics:

- Residual explained variance:
  `1 - ||R - h(S_T)||_F^2 / ||R||_F^2`
- Target scalar-link explained variance:
  `1 - ||Y - h(S_T)||_F^2 / ||Y - S_T||_F^2`
- Correlation between diagnostic score and NTD-PL RMSE/SAM gain.

Main message:

> The diagnostic predicts where NTD-PL should help. Datasets such as Cuprite are
> not merely failures; they are boundary cases where Tucker already leaves little
> coherent shared-link residual.

Expected paper impact:

- Makes the scoped claim falsifiable.
- Turns boundary cases into evidence for the mechanism.
- Gives users a practical way to decide whether NTD-PL is worth trying.

## 5. Add a Joint Link Ablation Beyond Polynomial

Reviewers may ask why the link is polynomial instead of a small neural network
or spline. The current MLPCal baseline is useful, but it is post-hoc; it does
not test joint nonlinear linked fitting.

Recommended ablation:

- Same Tucker backbone and rank.
- Replace the polynomial link with a one-dimensional MLP or spline.
- Train jointly with the Tucker backbone.
- Compare accuracy, stability, runtime, and interpretability.

Interpretation paths:

- If polynomial matches MLP/spline:
  - Emphasize that polynomial is simpler, analyzable, and sufficient.
- If MLP/spline slightly improves:
  - Present NTD-PL as the interpretable and theoretically tractable member of a
    broader linked-Tucker family.
- If MLP/spline is unstable:
  - Emphasize degree continuation and ridge link updates as practical benefits.

Keep this ablation compact. It is a reviewer-risk reducer, not the main story.

## 6. Add Runtime and Cost Evidence

The method claims to enhance Tucker cheaply. Add a small cost table to support
that claim.

Recommended comparison:

- matched-rank Tucker,
- matched-rank NTD-PL,
- RMSE-matching rank-lifted Tucker.

Report:

- parameter count,
- wall-clock training time,
- per-iteration time if available,
- memory if easy to measure,
- RMSE,
- SAM.

Main message:

> NTD-PL adds modest overhead to a Tucker backbone, while rank-lifted Tucker
> requires substantially more parameters to recover the same RMSE.

## 7. Rebalance the Theory Section

The analytic approximation theorem is useful but relatively standard. The more
distinctive theoretical contribution is the shared-backbone rank expansion.

Recommended theory emphasis:

1. Lead with the Veronese-lift interpretation of `S^{\odot p}`.
2. Explain that degree-`p` powers induce Tucker factors with ranks up to
   `binom(R+p-1, p)`.
3. Stress that these lifted factors are tied to the same original Tucker
   backbone.
4. Present the separation theorem as the formal reason this is more than scalar
   output calibration.
5. Keep analytic approximation as supporting evidence for smooth shared
   observation links.

Suggested message:

> The scalar link is small in parameter count but not small in induced tensor
> structure: its powers create tied high-order interactions among the same
> Tucker factors.

## 8. Suggested Main-Text Experimental Narrative

Recommended order:

1. Controlled tensors:
   - Show the mechanism activates when nonlinear residual energy increases.
   - Show degree dependence.

2. CAVE completion:
   - Use held-out random entries as the main generalization test.
   - Emphasize same-rank gains on RMSE and SAM.

3. CAVE reconstruction mechanism checks:
   - Matched-rank gains.
   - Parameter-efficiency/rank-lift comparison.
   - Fixed-backbone beta diagnostic.

4. External HSI validity and boundary cases:
   - Report transfer to Jasper Ridge, Samson, Urban.
   - Present Cuprite as a predicted boundary case using the residual diagnostic.

5. Additional cross-domain checks:
   - Keep in appendix unless space allows a short summary.

## Priority Checklist

High priority:

- [x] Add parameter-efficiency / rank-lift figure to the main text.
- [x] Recast PolyCal as a fixed-backbone beta diagnostic.
- [x] Add real-data residual diagnostic and correlate it with NTD-PL gains.
- [x] Reframe introduction around Tucker enhancement and tied rank lift.

Medium priority:

- [x] Add joint MLP or spline link ablation.
- [x] Add runtime/cost comparison.
- [x] Rebalance theory to emphasize Veronese rank expansion before generic
      polynomial approximation.

Lower priority:

- [ ] Polish terminology so "Tucker enhancement", "shared-backbone rank lift",
      and "joint linked fitting" are used consistently.
- [ ] Use boundary cases as mechanism evidence rather than defensive caveats.
- [ ] Keep broad benchmark claims modest.

## One-Sentence Target Pitch

NTD-PL enhances Tucker by learning a shared polynomial observation link whose
powers induce a parameter-tied rank lift of the same Tucker backbone, yielding
better parameter-efficiency and spectral fidelity when the linear Tucker
residual contains coherent nonlinear structure.
