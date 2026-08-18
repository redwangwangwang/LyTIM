# LyTIM method and code mapping

## 1. Clinical surrogate energy

For refinement step `k`, LyTIM defines

\[
V_k = \lambda_s V_{state}(R_k) + \lambda_\Delta V_{change}(R_k) + \lambda_b V_{back}(R_k).
\]

`models/utils/lytim_controller.py::compute_energy` implements all three terms per sample and per disease.

### Current state term

The report-state and current-image distributions are compared with Jensen–Shannon divergence:

\[
V_{state}=\frac{1}{D}\sum_d JS(q^{text}_{k,d}\|q^{image}_{t,d}).
\]

The image-state head is supervised online by CheXbert labels extracted from the existing current report.

### Temporal change term

For each disease, current and previous report states produce one of four online labels:

- `new`: previous negative, current positive;
- `resolved`: previous positive, current negative;
- `persistent`: previous positive, current positive;
- `stable-negative`: previous negative, current negative.

Blank/not-mentioned and uncertain states are masked. The progression head predicts these transition classes from the frozen TIM previous/current image representations. Its transition probabilities imply previous and current positive probabilities:

\[
\hat p_{prev}=p_{resolved}+p_{persistent},\qquad
\hat p_{curr}=p_{new}+p_{persistent}.
\]

`V_change` is the mean absolute disagreement between these implied probabilities and the report-state probabilities.

### Backward term

During training, a differentiable backward-state head reconstructs the previous disease state from the candidate report vector and image progression vector. During inference it is mixed with CheXbert probabilities from TIM's explicitly generated backward prior report:

\[
q_{back}=\rho q_{generated-prior}+(1-\rho)q_{learned-backward}.
\]

The backward term is `JS(q_back || q_previous-ground-truth)`.

## 2. Monotonic training

The initial Stage-I report and the teacher-forced refined report receive energies `V_0` and `V_1`. The monotonic margin loss is

\[
\mathcal L_{mono}=\max(0,V_1-V_0+m).
\]

The complete implemented objective is

\[
\mathcal L =
\lambda_{LM}\mathcal L_{LM}+
\lambda_{sup}\mathcal L_{state/change/back}+
\lambda_{mono}\mathcal L_{mono}+
\lambda_{keep}\mathcal L_{keep}+
\lambda_{stop}\mathcal L_{stop}+
\lambda_{gate}\mathcal L_{gate}.
\]

All targets are generated online from the original batch.

## 3. Fact-level trust gate

`gate_refinement_prompt` combines the per-disease error vector with the current report representation. It predicts 14 fact gates, projects them to TIM's 16 error-prompt tokens, and adds a learned disease context. A residual floor keeps the original TIM error prompt available early in training.

`L_keep` applies KL preservation to facts that both the initial report head and current image head already classify correctly.

## 4. Accept, rollback, and stop

`models/model_lytim.py::_monotonic_generate` performs batched, per-sample decisions:

```text
R = frozen Stage-I report
V = clinical_energy(R)

for k in 1..K_max:
    candidate = refiner(R)
    candidate_energy = clinical_energy(candidate)
    accept = candidate_energy < V - epsilon

    if accept:
        R = candidate
        V = candidate_energy
    else:
        rollback and deactivate this sample

    deactivate accepted samples when the stop head fires
return R
```

Samples in the same batch can accept different numbers of iterations. Rejected candidates never replace the current report.

## 5. Dataset compatibility

No change is made to `dataset/longitudinal_data_helper.py`. `lytim_seed_source=auto` supports either:

1. custom batches with cached `curr_stage1_text` and `prev_stage1_text`; or
2. the original TIM batch, where frozen Stage-I and backward reports are generated online.

This also removes the original Stage-II `KeyError` on missing cached report fields.

## 6. Recommended ablations

Run at least:

1. TIM Stage II;
2. LyTIM controller supervision only;
3. + monotonic loss;
4. + fact gate / preservation;
5. + accept/rollback;
6. + learned adaptive stop.

Report conventional generation metrics, clinical metrics, average accepted steps, rollback rate, energy drop, and performance by transition class.
