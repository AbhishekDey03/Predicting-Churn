# Kaggle Churn Summary

This project studies how different model classes behave on tabular data, using customer churn prediction as the test case. The focus is not just on performance (AUC), but on understanding how models separate churners from non‑churners and where that separation comes from.

---

## Key Findings from Model Comparison

- **Overall performance**
  - XGBoost achieves higher ranking performance (AUC ≈ 0.933) compared to RealMLP (AUC ≈ 0.917).
  - The gap reflects stronger exploitation of interaction structure by trees.

- **Risk concentration (ranking sharpness)**
  - XGBoost produces slightly more extreme separation between low- and high-risk groups (spread ≈ 0.865 vs 0.829).
  - In practice, this means XGBoost concentrates churners more tightly in the top decile.

- **Feature usage patterns**
  - XGBoost relies heavily on:
    - target-encoded interaction features (e.g. Contract $\times$ InternetService $\times$ PaymentMethod)
    - distributional statistics (e.g. std/max of charges)
  - RealMLP relies more on:
    - continuous ratio features ($x_2, x_{1,0}, x_3$)
    - raw numeric signals and simple digit-derived structure
  
  This indicates:
  - XGBoost $\rightarrow$ structured, interaction-driven decision rules
  - RealMLP $\rightarrow$ smooth, distributed representation of signal

- **Calibration / decile behaviour**
  - Both models show strong alignment between predicted probabilities and observed rates across deciles.
  - XGBoost is slightly sharper at the extremes (very low and very high risk bins).
  - RealMLP is smoother, with more gradual transitions between risk groups.

- **Model complementarity**
  - XGBoost excels at capturing discrete behavioural segments.
  - RealMLP captures continuous relationships between billing and tenure.
  
  The two models are therefore complementary rather than redundant, which justifies combining them or analysing them jointly.

## Problem Setup

We estimate a probability of churn for each customer:

$$\hat p(x_i) = \mathbb{P}(Y_i = 1 \mid X_i = x_i)$$

Evaluation is based on ranking quality (AUC), not calibration. In practice, this means the pipeline is optimised to **order customers correctly by risk**, rather than produce perfectly calibrated probabilities.

---

## High-Level Approach

Two model families are used deliberately with different inductive biases:

- **XGBoost** → structured, interaction-heavy, piecewise behaviour
- **RealMLP** → smooth, continuous function approximation

The idea is that these models capture *different views of the same signal*, which can then be analysed and combined.

---

## XGBoost Pipeline (Structured / Feature-Heavy)

The XGBoost pipeline is built around explicitly constructing signals that separate churners and non‑churners.

### Key ideas

- **Stabilise numerics**
  - log transforms to compress scale
  - exponential decay terms to introduce time sensitivity

- **Encode billing consistency**
  - ratios such as:
    $\text{TotalCharges} / (\text{MonthlyCharges} \cdot \text{tenure})$
  - deviations from expected billing behaviour act as churn signals

- **Explicit behaviour features**
  - number of services
  - contract types (month-to-month vs long-term)
  - auto-payment indicators

- **Interaction-heavy categoricals**
  - e.g. Contract × InternetService
  - allows trees to isolate specific behavioural segments

- **External (reference) features**
  Derived from the original Telco dataset:
  - frequency of values
  - empirical churn rates (priors)
  - percentile positions and distances to quantiles

- **Leakage-safe target encoding**
  - nested CV ensures each row is encoded without seeing its own label

### Model

A gradient boosted tree ensemble optimised with logistic loss. The model is naturally good at:

- picking up discrete interactions
- isolating sharp decision regions

---

## RealMLP Pipeline (Smooth / Representation-Based)

The RealMLP pipeline is intentionally lighter on feature engineering. The goal is to let the model learn a smooth function rather than rely on explicit partitions.

### Core design

- Separate **numerical** and **categorical** pathways
- Embed categorical variables (or one-hot if small)
- Transform numerics into a richer basis before feeding into the network

---

## What are the $x_n$ features?

The $x_n$ features are simple engineered transforms of the core billing variables, designed to expose structure that is otherwise implicit.

They include:

- Ratios between key quantities
  - $x_{1,0} = \text{MonthlyCharges} / \text{TotalCharges}$
  - $x_{1,1} = \text{TotalCharges} / \text{MonthlyCharges}$

- Time-normalised quantities
  - $x_2 = \text{TotalCharges} / \text{tenure}$

- Derived consistency relationships
  - $x_3$ compares monthly charges to the implied average spend

- Simple nonlinear expansions
  - $x_4 = \text{tenure}^2$

### Purpose

These features are **not meant to create hard splits** (as in trees), but to:

- expose smooth relationships between variables
- give the network better coordinates to learn from

---

## RealMLP Model Behaviour

The model applies learned transformations of the form:

$$\phi(x) = \cos(2\pi(wx + b))$$

which act as a basis expansion for numerical features. These are then combined through a small neural network ensemble:

$$f(x) = \frac{1}{K} \sum_{k=1}^K f^{(k)}(x)$$

This leads to:

- smoother decision boundaries
- more distributed feature usage
- better interpolation between observed points

---
