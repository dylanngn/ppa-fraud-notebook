# Defense Presentation: Fraud Detection in Online Real Estate Marketplaces
### Nguyen Hoang Minh — Master of Software Engineering, FPT School of Business and Technology
### Slide Script & Content Guide — Estimated 25–30 Minutes

---

> **How to read this document:**
> - **[SLIDE]** — what is physically on the slide (bullet points, figures, tables)
> - **[SCRIPT]** — what you say aloud, in natural spoken language
> - **[NOTES]** — internal reminders, transitions, or visual cues

---

## SLIDE 1 — Title Slide

**[SLIDE]**
```
Fraud Detection in Online Real Estate Marketplaces
Utilizing a Hybrid Graph and Gradient Boosting Model

Nguyen Hoang Minh
Student ID: 24MSE23152
FPT School of Business and Technology
Master of Software Engineering

Supervisor: Prof. Doan Xuan Huy Minh, Ph.D.
December 2024
```

**[SCRIPT]**
Good morning, respected council members and faculty. My name is Nguyen Hoang Minh, and today I will be presenting my master's thesis on fraud detection in online real estate marketplaces. Over the next 25 minutes, I will walk you through the full arc of this research — from the problem definition to the data, the models we built, the experiments we ran, and ultimately the production recommendation we arrived at, and the reasoning behind it.

The title of this work is straightforward — but I want to flag upfront that the answer this research produced was *not* what we expected going in. We built a sophisticated hybrid graph neural network system, and then the data told us not to use it in production. That tension is at the heart of this defense.

---

## SLIDE 2 — Motivation: Why Real Estate Fraud?

**[SLIDE]**
```
The Problem

🏠 Real estate listing platforms enable phantom listings, identity theft,
   and coordinated fraud rings at industrial scale

$145 million in annual consumer losses (FBI Internet Crime Report, 2023)

SMG Platform (Swiss Marketplace Group):
 • Homegate + ImmoScout24 — Switzerland's two largest property portals
 • 804,717 listing events analyzed in this study
 • 8.11% fraud rate at listing level
 • Median detection time: 1 hour 40 minutes

The Challenge:
 Traditional fraud detection (SEON) analyzes ONE transaction at a time.
 It cannot see patterns formed ACROSS listings, devices, and identities.
```

**[SCRIPT]**
Let's start with why this problem matters. Real estate fraud in online marketplaces takes several forms — phantom listings that collect deposits for apartments that don't exist, identity theft where legitimate listings are cloned, and most critically, *coordinated fraud rings* where attackers operate dozens or hundreds of fake listings simultaneously using shared devices, IP addresses, and phone numbers.

The FBI estimates $145 million in annual losses from this category of crime in the US alone. The dataset I worked with comes from the Swiss Marketplace Group, which operates Homegate and ImmoScout24 — the two dominant real estate portals in Switzerland. Out of 804,717 listing lifecycle events, 8.11% of listings were fraudulent at the listing level.

Now here is the key insight that motivates this whole research: the existing system, SEON, is a third-party fraud intelligence platform that scores each transaction individually. It looks at the IP address, the device fingerprint, the email address — all in isolation. It cannot see that a device is linked to 200 other listings. It cannot see that three listings share the same suspicious burner phone. It looks at one listing at a time and has no memory of the network structure around it.

That is the gap this research tries to fill.

---

## SLIDE 3 — Research Questions & Hypotheses

**[SLIDE]**
```
Four Research Questions

RQ1: Can GNN graph embeddings provide predictive signal beyond
     individual listing attributes (tabular data)?
     → Hypothesis: Yes — relational structure captures fraud rings

RQ2: What entity-sharing patterns distinguish fraud rings
     from legitimate multi-listing users?
     → Hypothesis: Shared device clusters, IP blocks, phone reuse at scale

RQ3: How robust are GNN and XGBoost models under temporal
     distribution shift (concept drift)?
     → Hypothesis: GNN degrades faster due to evolving graph topology

RQ4: What actionable indicators emerge to support
     human analyst review?
     → Hypothesis: SHAP-driven explanations reduce analyst workload
```

**[SCRIPT]**
The study is structured around four research questions. I'll go through all of them briefly now, and the experimental results section will answer each one explicitly.

RQ1 asks the core machine learning question: does building a graph and running a graph neural network over it actually help, compared to just using the 65 tabular features we already have? The hypothesis is yes — that the relational patterns form a signal that cannot be captured feature-by-feature.

RQ2 goes deeper into the data mining angle: what does a fraud ring actually look like in this graph? Are fraudsters dense, are they sparse, do they cluster? This question led to one of the most counter-intuitive findings of this research, which I'll show you shortly.

RQ3 is the operationally critical question: even if the GNN is better on a fixed dataset, does it *stay* better over time? Real-world fraud patterns change. Models deployed in production must be retrained. Do graph-based models survive this process?

And RQ4 is about interpretability — if we are asking human analysts to review flagged listings, we need to give them reasons, not just a probability score. How do we make the model's decisions explainable?

---

## SLIDE 4 — Dataset: What We Worked With

**[SLIDE]**
```
Dataset Overview — Swiss Marketplace Group (SMG)

Source              Volume
────────────────────────────────────
Insertion Events    804,717 events  (86,160 unique listings)
SEON Records        ~140,000 records
Features            509 raw columns → 65 engineered features
Date Range          December 2024 – November 2025 (12 months)

Label Distribution:
  Fraud rate (per listing)  8.11%   (6,987 fraudulent)
  Fraud rate (per event)    1.21%

⚠ Label Leakage Discovery:
  STATUS column showed DELETED for fraudulent listings →
  Removed permanently from feature schema
  (Post-hoc label contamination — temporal integrity violation)
```

**[SCRIPT]**
The dataset consists of 804,717 listing lifecycle events across 86,160 unique listings, collected over 12 months from December 2024 to November 2025. There are two data sources: the insertion event stream containing the actual listing content, pricing, category, location, and platform metadata; and the SEON transaction records which provide device fingerprinting, IP intelligence, email validation, and phone verification signals.

Raw data had 509 columns. After feature engineering, we work with 65 base tabular features.

The fraud rate at listing level is 8.11% — but I want to flag something important about this number. At event level it is only 1.21%, because a single listing generates many events over its lifetime. The label was derived from the downstream moderation system: a listing is fraud if it was eventually flagged and removed by the moderation team.

Each listing passes through a defined set of lifecycle statuses: DRAFT when the user is composing the listing; PENDING_APPROVAL when it is first submitted — this is the moment SEON runs its check and the model is called; APPROVED once it clears review and payment; PUBLISHED when it goes live on ImmoScout24 or Homegate. A published listing can re-enter the approval cycle via REPUBLISHING and PENDING_REPUBLISH_APPROVAL — the latter triggers a new SEON check identical to the original submission, so it also maps to a SEON transaction and is scored by the model. Finally, ARCHIVED or DELETED when the listing is removed — either by the user, or by the moderation team after fraud is confirmed.

Training uses the first event per status transition, excluding DRAFT. The primary training row for each listing is the PENDING_APPROVAL event, because that is the point-in-time at which the model would be called in production.

One critical issue discovered during this work: the STATUS column itself leaks the label. Fraudulent listings disproportionately show STATUS=ARCHIVED or STATUS=DELETED, but these are *post-hoc* transitions — they happen after the fraud is detected, not before. If you include STATUS as a feature, you are training on future information, and the model will appear excellent while being completely useless in production. This column was permanently removed from the feature schema.

---

## SLIDE 5 — Exploratory Analysis: The Fraud Landscape

**[SLIDE]**
```
Key Fraud Patterns Discovered

🕐 TEMPORAL              🌍 GEOGRAPHIC
  Feb 2025: 13.12%          Nepal:    90.5% fraud rate
  Peak: 6–7 AM              Benin:    66.9% (West Africa)
  Weekend uplift: +24%      Romania:  55.3%
  Median detection: 1h40m   France:   31.8%
                            Switzerland: 6.2% (baseline)

🏷 CATEGORY / TYPE       📡 PLATFORM BEHAVIOR
  Studio apartments: 34.97%  Cross-platform: 26.74%
  RENT vs BUY: 10.8x lift    Single platform:  6–10%
  High-risk: Flat, Singles   Fraudsters post on BOTH portals
                             to maximize victim exposure

📱 DEVICE FINGERPRINTING
  Super-connectors (10+ listings): 35.5% fraud rate  (4.4× baseline)
  Top device fingerprint: linked to 6,251 listings
  Single-listing devices: 16.7% fraud rate (higher than super-connectors!)
```

**[SCRIPT]**
Before building any model, we did an extensive exploratory analysis to understand what fraud actually looks like in this dataset. Several patterns emerged.

Temporally, we saw a spike in February 2025 where the fraud rate reached 13.12% — 62% above the annual average of 8.11%. We also see a strong hourly pattern where fraud activity peaks at 6 to 7 in the morning. This is consistent with automated account behavior — bots or scripts scheduled to run before human moderators start their shifts.

Geographically, the signal is stark. Swiss IP addresses have a baseline 6.2% fraud rate. Nepal reaches 90.5%. Several West African countries — Benin, Togo, Ivory Coast — cluster in the 60 to 90% range. France and Germany show elevated rates too, which may reflect VPN routing.

For listing categories, studio apartments and single rooms have nearly 35% fraud rate — these are the lowest entry point for rental scams where a small deposit is all that is requested. By contrast, house and villa listings are almost never fraudulent at only 2%.

The platform finding is interesting operationally: listings posted on *both* Homegate and ImmoScout24 simultaneously have a 26.74% fraud rate — four times higher than single-platform listings. Fraudsters maximise exposure by cross-posting, and we can detect this.

Now the most counter-intuitive finding, which ties directly to RQ2. You would expect that devices with many linked listings are more likely to be fraud — fraud rings reuse devices. But the data shows the opposite: devices linked to 10 or more listings have a 35.5% fraud rate, *lower* than single-listing devices at 16.7%. Why? Because in this marketplace, legitimate letting agencies — property management companies — also list hundreds of properties from the same corporate devices. So high connectivity signals *legitimacy* for established agents, while single-use devices are more suspicious. This is the Heterophily Paradox I describe in the thesis, and it fundamentally shapes how the graph model behaves.

---

## SLIDE 6 — Graph Construction: 31 Million Edges

**[SLIDE]**
```
Heterogeneous Graph Structure

Node Types:
  Listing   54,233 nodes   (primary target)
  User      76,147 nodes
  Device    12,313 nodes
  IP        34,635 nodes

Edge Types:
  shares_device    12,171,416  (dominant signal)
  shares_ip         2,597,653
  shares_user       2,065,691
  shares_email      1,978,469
  shares_phone      1,305,532
  Direct links        ~388,000
  ─────────────────────────────
  TOTAL            ~31,500,000 edges

Node Features: 44-dimensional vectors
  36 base features (normalized, label-encoded, binary)
   8 temporal encoding features (sin/cos for hour, weekday, etc.)
```

**[SCRIPT]**
The core idea of the graph-based approach is to connect listings together through shared entities. If two listings were created from the same device, they share a `shares_device` edge. If two listings share the same IP address, a `shares_ip` edge. If the same email pattern, a `shares_email` edge, and so on.

The resulting graph is massive: over 31.5 million edges across four node types. The dominant edge type is device-sharing with 12 million edges, which makes sense — device fingerprints are the most stable identifier in this dataset.

Each node carries a 44-dimensional feature vector: 36 base features normalized from the raw tabular data, plus 8 temporal encoding features using sine and cosine transformations of the hour and weekday. This cyclic encoding ensures that 11 PM and 1 AM are understood as close in time, not distant in integer space.

The challenge with this graph is scale. A naive dense representation would require gigabytes of memory. We address this with neighbor sampling: the GNN only samples 15 neighbors in the first hop and 10 in the second hop during training. This limits the receptive field but enables training on commodity hardware.

---

## SLIDE 7 — Model Architecture: The Four Variants

**[SLIDE]**
```
Four Model Variants Evaluated

┌─────────────────────────────────────────────────────────────┐
│  Variant 1: Logistic Regression (Baseline)                  │
│  65 tabular features → linear classifier                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Variant 2: Random Forest                                   │
│  65 tabular features → 100 trees, max_depth=10              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Variant 3: Vanilla XGBoost  ← PRODUCTION RECOMMENDATION   │
│  65 tabular features → gradient boosted trees               │
│  500 estimators, max_depth=12, HPO-tuned 50 trials          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Variant 4: GNN + XGBoost (Hybrid)                         │
│  Heterogeneous Graph → GraphSAGE → 16-dim embeddings        │
│  65 tabular + 16 GNN = 81 features → XGBoost               │
└─────────────────────────────────────────────────────────────┘
```

**[SCRIPT]**
We evaluated four model variants, designed as an ascending complexity ladder. This is important: the goal was not just to build the best possible model, but to understand *how much value each additional layer of complexity adds*, and at what operational cost.

Logistic regression is the pure baseline — linear decision boundary, no non-linearity, no interaction terms. It tells us the floor.

Random Forest is the second baseline — ensemble of trees, can capture non-linear patterns, does not require feature scaling.

Vanilla XGBoost is the primary candidate. Gradient boosted trees, 500 estimators, max depth 12, tuned with 50 Optuna trials over a 5-hour hyperparameter optimization run. This model operates purely on tabular features.

The hybrid GNN+XGBoost variant first runs GraphSAGE over the 31 million edge graph to produce 16-dimensional embedding vectors for each listing node. These embeddings encode the relational neighborhood structure. They are then concatenated to the 65 tabular features, giving XGBoost 81 total features to work with.

The key question: does that 17.3% extra signal from the GNN justify the operational complexity of maintaining a 31 million edge graph?

---

## SLIDE 8 — XGBoost: Theory and Why It Fits This Problem

**[SLIDE]**
```
XGBoost — Extreme Gradient Boosting

Principle: Sequentially build weak learners, each correcting
           the residual errors of the previous ensemble

Objective Function:
  L(Φ) = Σ l(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)

  where Ω(f) = γT + ½λ||w||²
        T = number of leaves, w = leaf weights
        γ = leaf count penalty, λ = L2 weight regularization

Key Strengths for Fraud Detection:
  ✓ Handles high-dimensional sparse binary features (SEON signals)
  ✓ Built-in regularization prevents overfitting on class-imbalanced data
  ✓ scale_pos_weight: directly weights minority class (fraud = 5.3×)
  ✓ Native handling of missing values
  ✓ Consistent with literature: dominant model in fraud benchmarks
  ✓ Inference time: sub-millisecond per sample

Tuned Hyperparameters (50 Optuna TPE trials):
  n_estimators: 500    max_depth: 12     learning_rate: 0.151
  min_child_weight: 8  subsample: 0.818  colsample_bytree: 0.985
  gamma: 0.236         reg_alpha: 0.078  reg_lambda: 0.406
  scale_pos_weight: 5.297
```

**[SCRIPT]**
XGBoost deserves its own slide because understanding its mechanics is important to understanding why it performs so well on this problem.

At its core, XGBoost builds an ensemble of weak learners — decision trees — sequentially. Each new tree is trained to correct the errors of the current ensemble, fitting the *negative gradient* of the loss function. This is gradient boosting. The "extreme" part refers to the regularization and computational optimizations that make it fast and resistant to overfitting.

The objective function has two terms: the training loss, which measures how well the model fits the data, and the regularization term Omega. The regularization penalizes both the *number of leaves* (through gamma) and the *magnitude of leaf weights* (through lambda). This is why XGBoost handles high-dimensional data so well — it is actively discouraged from memorizing rare patterns.

For fraud detection specifically, XGBoost has several structural advantages. The SEON signals are largely Boolean — tor, vpn, proxy, disposable email, data center IP — plus categorical signals like IP country and ISP name. Decision trees naturally handle this mixed feature space without transformation. The `scale_pos_weight` parameter directly addresses class imbalance: at 5.3, it tells the model that each fraud case is worth 5.3 times a legitimate case during training.

The hyperparameters you see here were not guessed — they were found through 50 Optuna TPE trials, which is a Bayesian optimization approach that intelligently explores the hyperparameter space based on prior evaluations. The result was max_depth of 12, which is relatively deep, and learning rate 0.151, which is moderate — suggesting the model needed many estimators at moderate step size rather than few at high step size.

---

## SLIDE 9 — GraphSAGE: Theory and Why Not HGT or CARE-GNN

**[SLIDE]**
```
GraphSAGE — Inductive Graph Representation Learning

Core Idea: Learn an aggregation function over neighborhoods,
           not a fixed per-node embedding

h_v^k = σ( W^k · CONCAT(h_v^(k-1), AGG({h_u^(k-1), u ∈ N(v)})) )

Two Message-Passing Layers:
  Layer 1: Aggregate from 15 sampled neighbors
  Layer 2: Aggregate from 10 sampled neighbors
  Output:  16-dimensional embedding per listing node

Architecture Evaluated:
  Architecture   AUC-PR  Train Time   Why Rejected / Chosen
  ─────────────────────────────────────────────────────────
  GraphSAGE     0.6856   ~1 min      ✓ CHOSEN — best AUC-PR, fastest
  HGT (4 heads) 0.6351   ~2 min      ✗ Attention overhead, no benefit
  CARE-GNN      0.6542   ~45 min     ✗ 22× slower, 3× memory, worse
  PC-GNN        0.6488   ~2.5 min    ✗ Complex, marginal gains

Key Design Decision:
  Graph structure is MOSTLY HOMOGENEOUS after filtering
  (listing-to-listing edges dominate at 31M / 31.5M total)
  → HGT heterogeneous attention provides no structural benefit
```

**[SCRIPT]**
For the graph neural network component, we evaluated four architectures. Let me explain the choice of GraphSAGE and why the more sophisticated alternatives were rejected.

GraphSAGE — Inductive Graph Sampling and Aggregation — works by learning an aggregation function rather than a fixed embedding for each node. This is critical for our use case because new listings arrive continuously. A transductive method like vanilla GCN would need to be retrained from scratch each time a new node appears. GraphSAGE learns *how to aggregate* from neighborhoods, so it can generalize to nodes it has never seen during training.

The update rule is: at each layer k, take the previous-layer embedding of node v, concatenate it with an aggregation of its neighbors' previous-layer embeddings, apply a weight matrix and a non-linearity. After two layers, each node has a representation informed by its 2-hop neighborhood.

We also evaluated HGT — the Heterogeneous Graph Transformer — which uses attention mechanisms specialized for heterogeneous graphs where edges have different types and semantic meanings. In theory, this should help because our graph has four node types and six edge types. In practice, after filtering for the most meaningful connections, listing-to-listing edges dominate overwhelmingly at 31 million out of 31.5 million total edges. The graph is effectively homogeneous, and HGT's attention mechanisms have no structure to exploit. The result: HGT scored 0.6351 AUC-PR versus GraphSAGE's 0.6856.

CARE-GNN was designed specifically for fraud detection with camouflage-resistant aggregation — the idea being that fraudsters deliberately mimic legitimate behavior to confuse graph models. While theoretically appropriate, the implementation reality was brutal: 45 minutes per epoch versus GraphSAGE's 2 minutes — a 22x slowdown — with 3x memory overhead and worse final performance. The theoretical appeal did not survive contact with the data.

---

## SLIDE 10 — MLOps Stack: Hydra + MLflow

**[SLIDE]**
```
Why MLOps Infrastructure Matters in Research

Without It                    With It
──────────────────────────────────────────────────────
Manual parameter changes   →  Hydra config management
"Which run was that?"      →  MLflow experiment tracking
Lost model weights         →  MLflow model registry
Irreproducible results     →  Every run logged with params + metrics
"Re-run but change X"      →  Hydra override: train-vanilla max_depth=10

HYDRA — Configuration Management
  ┌─ configs/experiment.yaml ─────────────────────────┐
  │  MLflow tracking URI, dates, model variant,        │
  │  GNN architecture, XGBoost hyperparameters         │
  │  Override from CLI: training.mode=expanding         │
  └────────────────────────────────────────────────────┘

MLFLOW — Experiment Tracking + Registry
  Three Experiments:
    fraud-detection            → single-split runs
    fraud-detection-expanding  → drift validation
    fraud-detection-hpo        → 50 Optuna trials each

  Model Registry:
    @production alias → promoted after evaluation
    Loaded by API at startup → zero downtime model swap
```

**[SCRIPT]**
I want to spend a moment on the operational infrastructure because this is often under-explained in academic work, but it is what makes the difference between a research prototype and a deployable system.

Hydra is a configuration management framework developed by Facebook Research. The core idea is that *every setting that affects a run* — training dates, model variant, GNN architecture, learning rate — is stored in a YAML configuration file, not hardcoded in Python. You can override any parameter from the command line at run time. This enabled me to run every model variant, every expanding window, and every HPO trial with a single `make` command. The configuration is immutable after the run starts, which means every experiment is fully reproducible by re-running with the same config.

MLflow is the experiment tracking backbone. Every single training run — and there were hundreds — automatically logs: the full set of hyperparameters from Hydra, every metric (AUC-PR, AUC-ROC, precision, recall, F1), the model artifact as a serializable pickle, and SHAP plots when requested. The MLflow Model Registry provides versioning and aliasing: when a new model is evaluated and found production-worthy, the `@production` alias is moved to that version. The FastAPI service loads the model by alias at startup, meaning a model promotion requires no code deployment.

From a research integrity standpoint, Hydra plus MLflow together ensure that *every number in this thesis was produced by a run that can be re-executed identically*. No manual parameter tracking, no Excel spreadsheet of results, no "I think it was this configuration." The run is the record.

---

## SLIDE 11 — Validation Methodology: Two Regimes

**[SLIDE]**
```
Two Distinct Evaluation Protocols

┌─────────────────────────────────────────────────────────┐
│  PROTOCOL 1: Fixed Temporal Split                       │
│                                                         │
│  Train: 2024-12-01 → 2025-06-01  (380,190 samples)     │
│  ──── 7-day gap (label maturation) ────                 │
│  Test:  2025-06-08 → 2025-07-01  ( 39,645 samples)     │
│                                                         │
│  Answers: "What is the best achievable performance      │
│            given sufficient training data?"             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  PROTOCOL 2: Expanding Window (Concept Drift)           │
│                                                         │
│  W1: Train → Jan 2025  (103K) | Test → Feb 2025        │
│  W2: Train → Mar 2025  (174K) | Test → Apr 2025        │
│  W3: Train → Apr 2025  (241K) | Test → May 2025        │
│  W4: Train → May 2025  (308K) | Test → Jun 2025        │
│  W5: Train → Jun 2025  (373K) | Test → Jul 2025        │
│                                                         │
│  Answers: "How robust is the model when the world       │
│            evolves and we must retrain frequently?"     │
└─────────────────────────────────────────────────────────┘

⚠ These two protocols answer DIFFERENT questions.
  Ignoring Protocol 2 would lead to the WRONG production choice.
```

**[SCRIPT]**
The evaluation methodology is where this research makes an important methodological contribution. We ran two completely different validation protocols, and they tell very different stories.

Protocol 1 is a fixed temporal split. We train on data from December 2024 to June 2025 — 380,000 samples — with a mandatory 7-day gap before testing, and evaluate on the following month. The 7-day gap simulates label maturation: in reality, a fraud label may not be assigned immediately. A listing that was fraudulent might not be removed and flagged in the moderation system for several days after the event occurred. Training without this gap would introduce future labels into the training set.

Protocol 2 is the expanding window protocol, designed to simulate concept drift conditions. We run five consecutive training and evaluation cycles. Each cycle accumulates all data up to a cutoff point, trains a model, evaluates on the next period, then expands the training window and repeats. This simulates the production reality: every month we retrain the model on all historical data and deploy it to score the next month's traffic.

Critically — and this is the central argument for the production recommendation — these two protocols answer *different questions*. Protocol 1 asks: given all the data we ever have, what is the ceiling? Protocol 2 asks: when the world changes and we must keep retraining, which model stays reliable? Any study that only runs Protocol 1 and then deploys the winner into production is making an unsound inference.

---

## SLIDE 12 — Results: Fixed Split Comparison

**[SLIDE]**
```
Model Performance — Fixed Temporal Split (3 independent runs)

Model               AUC-PR         AUC-ROC       Precision  Recall
────────────────────────────────────────────────────────────────────
Logistic Regression 0.577 ± 0.117  0.946 ± 0.017  0.214     0.872
Random Forest       0.621 ± 0.078  0.953 ± 0.008  0.488     0.827
Vanilla XGBoost     0.733 ± 0.063  0.958 ± 0.003  0.746     0.671
GNN + XGBoost       0.733 ± 0.058  0.950 ± 0.009  0.758     0.673
────────────────────────────────────────────────────────────────────
SEON Baseline       0.598          0.943          varies    varies

Precision @ K  (analysts review top-K predictions):
  Model            P@50    P@100   P@200
  ─────────────────────────────────────
  Vanilla XGBoost  99.3%   97.7%   87.0%
  GNN + XGBoost    96.7%   98.0%   88.0%
  Random Forest    85.3%   81.7%   80.0%

⚡ Key Insight:
   GNN improvement = +0.08% AUC-PR (not statistically significant, p=0.92)
   Our system improvement over SEON = +17% AUC-PR
```

**[SCRIPT]**
These are the headline numbers from the fixed-split evaluation. A few things to read carefully here.

First, the jump from logistic regression to vanilla XGBoost is large: from 0.577 to 0.733 AUC-PR — that is a 27% improvement. This reflects XGBoost's ability to capture non-linear interactions between features, which logistic regression cannot.

Second, our system shows a 17% improvement over the SEON baseline. This is the primary commercial value claim: even without the graph model, adding XGBoost over the same SEON signals produces substantially better fraud detection.

Third — and this is the result that needs to be read carefully — GNN+XGBoost achieves 0.733 AUC-PR, identical to vanilla XGBoost at 0.733. The difference is less than 0.1%. A paired t-test yields p=0.92, meaning we cannot distinguish these two models from random noise in three runs. There is no statistically significant advantage from the graph neural network in the fixed-split setting.

The Precision@K table shows the picture for analyst workflows. If you show analysts the top 100 model predictions, both XGBoost variants have 97-98% precision — meaning almost every listing in that queue is genuinely fraudulent. This is operationally excellent.

But the honest conclusion from this slide is: in the fixed-split regime, the GNN does not justify its complexity. The question becomes whether there is *any* regime in which it does.

---

## SLIDE 13 — Expanding Window: The Critical Experiment

**[SLIDE]**
```
Expanding Window Results — The Production Question

Window  Train End   Train N   XGB AUC-PR   GNN AUC-PR
───────────────────────────────────────────────────────
W1      Jan 2025    103,018     0.6846       0.2219
W2      Mar 2025    173,739     0.5346       0.1850
W3      Apr 2025    240,660     0.5788       0.2325
W4      May 2025    308,071     0.7467       0.2376
W5      Jun 2025    373,484     0.5982       0.2164
───────────────────────────────────────────────────────
MEAN                           0.629 ±0.077  0.219 ±0.018

                     Vanilla XGBoost    GNN + XGBoost
Fixed-split AUC-PR      0.733              0.733
Expanding mean AUC-PR   0.629 (86%)        0.219 (30%)
Performance Retention    86%                30%

📉 GNN degradation: -70% performance loss under temporal drift
✅ XGBoost retention: 86% of fixed-split performance maintained
```

**[SCRIPT]**
This slide answers Research Question 3, and in my view it is the most important result in the entire thesis.

When we move from the fixed-split setting to the expanding window setting, the two models diverge completely. Vanilla XGBoost maintains 86% of its fixed-split performance — a mean AUC-PR of 0.629 across the five windows, with moderate variance. That is a robust, deployable model.

The GNN+XGBoost collapses. Its mean AUC-PR across the five windows is 0.219 — a 70% performance loss from the 0.733 it achieved in the fixed split. In Window 1, with only 103,000 training samples, the GNN produces 0.22 AUC-PR — barely better than random chance on this imbalanced dataset.

Why does this happen? Fraud rings evolve. When a new fraud ring forms in February 2025, it creates new device fingerprints, new IP addresses, new phone numbers — entities that were never in the training graph. The GNN learned to aggregate over the specific topology of the training period's fraud rings. When those fraud rings change or new ones form, the learned message-passing aggregations no longer apply. The graph structural patterns do not generalize across time the way tabular feature patterns do.

The tabular features — price, category, device fingerprint flags from SEON, hour of day — these have more stable statistical distributions. A tor node is always suspicious. A 6 AM submission from a Nepalese IP address for a studio apartment rental is always suspicious. These patterns persist month over month.

Notice also that the GNN's variance is very *low* at ±0.018. That is not stability — that is consistently bad performance. It is stuck at approximately 0.22 across all five windows regardless of how much training data accumulates.

---

## SLIDE 14 — The Data Requirement Threshold

**[SLIDE]**
```
When Does the GNN Start to Work? (RQ3 Answer)

Expanding Window AUC-PR vs Training Set Size:

  Training N   XGB AUC-PR   GNN AUC-PR   Difference
  ──────────────────────────────────────────────────
  103,018       0.685         0.222        XGB +46%
  173,739       0.535         0.185        XGB +35%
  240,660       0.579         0.233        XGB +25%
  308,071       0.747         0.238        XGB +51%
  373,484       0.598         0.216        XGB +38%

  GNN does NOT close the gap even at 373K samples.

Fixed-Split (full 380K training, ~420K inference available):
  GNN = 0.733  XGB = 0.733  → parity achieved at scale

Thesis Estimate: ~300,000 samples needed for meaningful
                 GNN pattern extraction

But: The fixed-split uses a DIFFERENT (favorable) data distribution
     where the test set follows immediately after large contiguous training
```

**[SCRIPT]**
This slide addresses the question: is the GNN failure a data volume problem? If we had even more training data, would the GNN eventually catch up?

Looking at the expanding windows, the GNN's AUC-PR hovers around 0.20 to 0.24 regardless of whether we have 103,000 or 373,000 training samples. XGBoost consistently outperforms it by 25 to 51 percentage points across all windows.

The thesis estimates that approximately 300,000 samples are needed before the graph structure becomes dense enough for meaningful pattern extraction — this is where the entity-sharing networks become dense enough that message passing can detect fraud rings. And indeed, the fixed-split evaluation uses 380,000 training samples and shows parity. But the fixed-split is a *favorable* evaluation: the training period runs for six months continuously before testing the very next month. The data distributions are temporally adjacent.

In the expanding window setting, the model is repeatedly retrained as the world evolves — and in this operational reality, the GNN never catches up, even with large training sets. The reason is not volume; it is structural temporal shift. Each new fraud ring is a new connected component in the graph with no edges to previous training nodes. The GNN has no knowledge transfer mechanism for truly novel subgraphs.

---

## SLIDE 15 — SHAP Analysis: What Drives the Predictions?

**[SLIDE]**
```
SHAP Feature Importance (GNN+XGBoost, 500 test samples)

Top 10 Features by Mean |SHAP|:
  Rank  Feature                      Category            SHAP
  ────────────────────────────────────────────────────────────
  1     session/screen_resolution    Browser fingerprint  1.271
  2     payment_mode                 Listing attribute    1.116
  3     LISTING_PRICES_RENT_NET      Price signal         0.781
  4     LISTING_CATEGORIES           Property type        0.756
  5     ip_isp_name                  Network signal       0.546
  6     LISTING_PRICES_BUY_PRICE     Price signal         0.482
  7     action_type                  User behavior        0.431
  8     session/font_count           Browser fingerprint  0.417
  9     session/browser              Browser identity     0.403
  10    LISTING_PLATFORMS            Multi-platform flag  0.392

Feature Category Breakdown:
  Browser / Session Signals   28.5%  ████████████████████
  Listing Attributes          23.2%  ████████████████
  GNN Embeddings (16 dims)    17.3%  ████████████
  Network / IP Signals        14.5%  ██████████
  Temporal Features            9.1%  ██████
  Payment / Action Meta        5.4%  ████
  Phone Features               2.1%  █
```

**[SCRIPT]**
SHAP — SHapley Additive exPlanations — provides a theoretically grounded way to decompose each prediction into per-feature contributions. The SHAP value for a feature is the average marginal contribution of that feature across all possible orderings of feature introduction, derived from cooperative game theory. For XGBoost, SHAP values are computed exactly rather than approximated.

Looking at the top features, screen resolution ranks first with a mean absolute SHAP value of 1.271. This reflects the device fingerprinting pattern we saw in EDA: fraudsters using headless browsers, browser automation tools, or low-resolution spoofed devices create distinctive screen resolution signatures that strongly predict fraud.

Payment mode comes second. This is the billing and payment configuration chosen for the listing — certain payment channels correlate strongly with fraudulent activity.

Rental price and listing category follow, reflecting the category-level fraud rates we quantified in EDA: studio apartments, specific price ranges, RENT-type listings.

The category-level breakdown is the most policy-relevant insight: 82.7% of predictive signal comes from tabular features that the vanilla XGBoost already has access to. GNN embeddings contribute 17.3% — a meaningful but minority contribution.

This answers RQ1 directly: yes, graph embeddings provide predictive signal beyond tabular features. But that signal is one-sixth of the total. It complements but does not dominate.

---

## SLIDE 16 — GNN Differential Detection: What Can Only the Graph See?

**[SLIDE]**
```
Model Disagreement Analysis (threshold = 0.5)

Category                    Count   Interpretation
──────────────────────────────────────────────────────────
Both models detect fraud      614   Shared capability
Both models miss fraud        141   Hard cases (sophisticated fraud)
GNN-only detections            13   ← Unique GNN value
XGBoost-only detections         1   XGBoost rarely misses what GNN catches

Net GNN advantage: +12 additional fraud cases

Profile of GNN-Unique Detections:
  Avg device connections: 3,565 (dataset average: 847)
  Characteristic: listings appear NORMAL individually
  Network profile:  part of MASSIVE device-sharing clusters
  Why XGBoost fails: no visibility into relational context

These 13 cases represent organized crime-scale fraud rings.
Individual signals are deliberately engineered to be benign.
Only multi-hop graph traversal reveals the coordination.

→ GNN is a superior FORENSIC tool for fraud ring discovery
→ GNN is NOT YET a reliable OPERATIONAL scoring tool
```

**[SCRIPT]**
Even though the GNN doesn't outperform vanilla XGBoost on aggregate metrics, there is a qualitatively important story about what the GNN can detect that XGBoost cannot.

We analyzed model disagreements at the 0.5 threshold. Out of 768 fraud cases in the test set, both models correctly detected 614. Both missed 141 — these are the hard cases, likely sophisticated fraud with carefully engineered features. But the GNN caught 13 cases that vanilla XGBoost completely missed. XGBoost caught only 1 that GNN missed.

The profile of those 13 GNN-only detections is striking. They had an average of 3,565 device-sharing connections — four times the dataset average of 847. These listings, viewed in isolation, appear entirely normal. Legitimate-looking prices, standard listing categories, no SEON flags. But in the graph, they are embedded in massive coordinated fraud ring clusters. The message-passing in GraphSAGE aggregates from hundreds of neighboring fraud-related nodes, producing an embedding that signals high risk despite the superficially clean individual features.

This is exactly the type of organized fraud that is most dangerous — sophisticated actors who have specifically tuned their individual listing attributes to pass automated screening. The only way to catch them is to see the network they operate in.

This leads to a nuanced conclusion. The GNN is not a better classifier in the statistical sense. But it is a better *detective* for a specific category of organized fraud. The right deployment model is: use vanilla XGBoost for real-time production scoring, and use GNN+XGBoost as an offline forensic tool for periodic fraud ring analysis and case investigation.

---

## SLIDE 17 — Why We Optimize for Operations, Not Raw Performance

**[SLIDE]**
```
The Operational Argument for Vanilla XGBoost

Dimension          Vanilla XGBoost      GNN + XGBoost
─────────────────────────────────────────────────────────────
Fixed-split AUC-PR   0.733 ± 0.063      0.733 ± 0.058  (equal)
Expanding AUC-PR     0.629 ± 0.077      0.219 ± 0.018  (3× worse)
Training time        ~30 seconds        ~5–10 minutes
Retraining overhead  Just data → model  Data → graph rebuild → GNN → model
Graph dependency     None               31M edges must be current
Inference latency    Sub-millisecond    Graph lookup on new node?
Deployment           Pickle + API       Pickle + active graph store
Failure mode         Degrades slowly    Collapses at distribution shift

The Drift Argument:
  Monthly retraining on fresh data is sustainable.
  Monthly graph reconstruction on 31M edges is possible but fragile.
  Novel fraud rings → zero in-graph representation → GNN blind spot.

Production Infrastructure:
  FastAPI + MLflow Registry + Hydra config
  → THRESHOLD_HIGH=0.7 → DECLINE (automated)
  → THRESHOLD_MEDIUM=0.3 → REVIEW (human analyst queue)
  → < 0.3 → APPROVE

We did not optimize for the highest AUC-PR number.
We optimized for the model that stays reliable in production.
```

**[SCRIPT]**
This slide is the core of the thesis's practical contribution, and I want to explain the philosophy behind it carefully because it represents a deliberate departure from how most ML research papers are evaluated.

Most papers report fixed-split performance and declare the highest number the winner. In research terms, if GNN and XGBoost are equal at 0.733, the GNN variant adds value because it occasionally detects novel fraud ring members.

But we are not building a research artifact. We are recommending a production system to a commercial marketplace operator. The question is not "which model is best on this fixed evaluation?" The question is "which model can an operations team confidently deploy, monitor, and maintain over 12 months?"

When you frame it that way, the choice inverts. The GNN requires: maintaining a current graph of 31 million edges, retraining the graph encoder each cycle, handling cold-start for new entities not in the graph, rebuilding the graph when new relationship data arrives, debugging model degradation when the graph topology drifts. Any one of these is manageable. Together, they represent a substantial operational burden with multiple failure modes.

Vanilla XGBoost requires: a Parquet file of labeled events, a Hydra config, a 30-second training job, and an MLflow artifact. Monthly retraining is a cron job. Model promotion is an MLflow alias change. The entire system fits in a weekend deployment by one engineer.

The API we built reflects this. Two thresholds split predictions into three tiers: scores above 0.7 are automatically declined, scores between 0.3 and 0.7 go to a human analyst queue for review, scores below 0.3 are approved. The thresholds are configurable environment variables, not hardcoded, so the fraud operations team can tune them as the business evolves without touching code.

The thesis recommendation is: deploy XGBoost now, use GNN offline for forensic investigation, and consider GNN for production only when a graph-aware drift detection mechanism and graph-topology monitoring infrastructure are in place.

---

## SLIDE 18 — Negative Results: What Did Not Work

**[SLIDE]**
```
Negative Results — What We Tried That Failed

Approach 1: Handcrafted Graph Features
  Manually encode: degree counts, historical fraud rates,
  link recency, entity-level similarity scores
  Result: -36.6% degradation (0.6639 → 0.4833 AUC-PR)
  Why: Distribution shift between training/test graph,
       spurious correlations, "crowding out" real features

Approach 2: Filter to High-Risk Edges Only
  Keep only edges where endpoint fraud rate > 15%
  (31M → ~8M edges)
  Result: -0.3% change (0.6990 → 0.6967)
  Why: Supervised GNN already down-weights low-signal edges;
       manual filtering removed potentially useful negative signals

Approach 3: CARE-GNN (Camouflage-Resistant)
  Designed specifically for fraud detection
  Result: 45 min/epoch, 3× memory, 0.6945 AUC-PR
  Why: GraphSAGE's structural simplicity outperforms complexity
       when graph is effectively homogeneous

Approach 4: Self-Supervised GNN (Link Prediction)
  Train without fraud labels; use topology alone
  Result: 0.6621 vs 0.6990 supervised (+5.6% supervised advantage)
  Why: Fraud signals in labels far more valuable than raw topology
```

**[SCRIPT]**
Negative results are often omitted from presentations, but they deserve their own slide because they contain real scientific value.

The handcrafted graph features experiment was motivated by parsimony: instead of training a full GNN, what if we just encoded the graph structure as explicit tabular features — degree counts, historical fraud rates per entity, connection recency? The result was a 36.6% performance drop. The features we engineered amplified distribution shift artifacts: a device's historical fraud rate in the training period may have nothing to do with its fraud rate in the test period if fraud rings rotate. The explicit features also crowded out the stable SEON signals. The GNN's learned embeddings are more resistant to this because they encode relational structure in a softer, distributed representation rather than brittle explicit counts.

The edge filtering experiment asked: can we make the GNN faster and more focused by removing low-risk edges? Keeping only edges where the endpoint has above 15% historical fraud rate reduces the graph from 31 to 8 million edges. But performance barely changed at -0.3%. The supervised GNN training already learns to down-weight uninformative neighborhoods. Manual filtering removed potentially useful negative contrast signals. Let the model decide.

The CARE-GNN experiment was the most disappointing because the architecture is theoretically the most appropriate. It was designed to resist fraud camouflage — exactly the problem we are solving. But 45 minutes per epoch made it impossible to tune, and its final performance was worse than the simpler GraphSAGE. Theory does not always survive data.

The unsupervised GNN experiment confirms something important: the fraud labels are the most valuable signal. Training the GNN with only topological objectives — predicting link existence — produces worse embeddings than training directly on fraud labels. The graph structure by itself is not enough; you need the downstream task to guide the representation.

---

## SLIDE 19 — Ethical Considerations

**[SLIDE]**
```
Bias, Fairness, and the Ethics of Geographic Signals

Risk Identified:
  Geographic features (IP country, ISP) are among the top predictors.
  Nepal: 90.5% fraud rate. Benin: 66.9%. Switzerland: 6.2%.

  Using nationality as a fraud signal can:
  • Create systematic false positives for legitimate users from high-risk countries
  • Amplify existing discrimination
  • Create feedback loops: flagged → no appeal → data reinforces pattern

Mitigations Implemented:
  1. Multivariate scoring: IP country is ONE of 65 features;
     no single feature can trigger a DECLINE alone
  2. Local SHAP explanations: each prediction shows which features
     drove the score — transparent to analysts and to appeals
  3. Human-in-the-loop: REVIEW tier routes borderline cases to humans
  4. False positive monitoring by country recommended for production
  5. Appeal mechanism required before actioning geographic-flagged listings

Ground Truth Caveat:
  Labels derived from existing moderation → biased toward previously
  detected fraud patterns. Sophisticated evasion remains unlabeled.
  Active learning with analyst input is recommended future work.
```

**[SCRIPT]**
I want to address the ethical dimension directly, because any system that uses geographic origin as a predictive feature carries real discrimination risk.

The data shows that Swiss IP addresses have a 6.2% fraud rate while Nepalese IP addresses have a 90.5% fraud rate. These are empirical observations. But a model that declines listings primarily because of IP origin would systematically harm legitimate users from these countries — students studying in Switzerland, diaspora communities, legitimate international renters.

We address this through three design choices. First, the model uses 65 features, and no single feature drives the decision alone. SHAP analysis confirms that IP country contributes through ISP name as a proxy, not directly as a country code. The geographic signal is real but contextual.

Second, the three-tier decision architecture — DECLINE, REVIEW, APPROVE — means borderline geographic signals go to human review rather than automatic rejection. This is critical: the model is an analyst prioritization tool, not an autonomous enforcement system.

Third, SHAP local explanations mean that every prediction is explainable. When an analyst reviews a case, they can see exactly why the model scored it high. When a lister appeals a decision, there is a factual basis for the discussion.

The ground truth limitation is important to acknowledge: labels come from the existing moderation system, which may have its own historical biases. If the previous moderation system was more likely to investigate listings from certain countries, the training data will amplify that pattern. Active learning — where analyst feedback continuously updates the training labels — is the recommended path forward to address this structural issue.

---

## SLIDE 20 — Production System Architecture

**[SLIDE]**
```
Deployed System Architecture

  New Listing Event (JSON)
          │
          ▼
    FastAPI /predict
          │
    ┌─────▼──────────────────────────────────────────┐
    │  Feature Engineering (real-time)               │
    │  EventPayload → 65-feature vector              │
    │  + link counts from EventStore                 │
    └─────┬──────────────────────────────────────────┘
          │
    ┌─────▼──────────────────────────────────────────┐
    │  XGBoost (loaded from MLflow Registry)         │
    │  → fraud_probability: 0.0 – 1.0               │
    └─────┬──────────────────────────────────────────┘
          │
    ┌─────▼──────────────────────────────────────────┐
    │  Decision Thresholds                           │
    │  ≥ 0.70 → DECLINE  (HIGH risk)                 │
    │  0.30–0.70 → REVIEW (MEDIUM risk)              │
    │  < 0.30  → APPROVE  (LOW risk)                 │
    └─────┬──────────────────────────────────────────┘
          │
    Response: fraud_probability, risk_tier, decision,
              top_risk_factors (SHAP), model_version

Key Endpoints:
  POST /predict        Score a listing event
  POST /ingest         Store event for link analysis
  POST /rebuild-graph  Refresh entity link indexes
  GET  /model-info     Current model version + thresholds
  GET  /health         Service health + store stats
```

**[SCRIPT]**
The production system is a FastAPI service backed by the MLflow Model Registry. The flow is straightforward: a listing event arrives as a JSON payload containing all the fields we engineered features from — listing attributes, SEON signals, session data, hashed identity fields. The feature engineering layer transforms this into the 65-feature vector in real time, augmenting it with link counts from the in-memory EventStore.

The XGBoost model, loaded from the MLflow Registry at startup under the `@production` alias, scores the vector and returns a fraud probability. The decision thresholds — 0.7 for high, 0.3 for medium — are environment variables configurable without code deployment.

The response object includes not just the probability and decision, but the top risk factors: specific features and their SHAP contributions, human-readable. When an analyst opens a review queue case, they see exactly: "This listing scored HIGH because screen resolution was associated with headless browser patterns, listing category is studio apartment, and the IP ISP was flagged in 78% of fraud cases at this network range."

The EventStore is an in-memory data structure that maintains a history of listing events indexed by email hash, phone hash, IP hash, and device hash. This enables the real-time link counting that approximates the graph features at inference time without requiring a full graph rebuild. When a new listing arrives, the store can immediately report how many other listings share its device fingerprint.

---

## SLIDE 21 — Conclusion: Answering the Research Questions

**[SLIDE]**
```
Research Questions — Answered

RQ1: Do GNN embeddings add signal?
  ✓ YES — 17.3% feature importance, +12 GNN-exclusive detections
  ✗ BUT — statistically indistinguishable from XGBoost (p=0.92)

RQ2: What characterizes fraud ring topology?
  ✓ Heterophily Paradox: dense clusters = legitimate agents
  ✓ Super-connectors (10+ listings): 35.5% fraud rate, 4.4× lift
  ✓ Geographic clustering: West Africa 60–90% fraud rates
  ✓ Temporal signatures: 6–7 AM peak, Feb 2025 spike at 13.12%

RQ3: How robust is GNN under temporal drift?
  ✓ XGBoost: 86% performance retention in expanding window
  ✗ GNN: 70% performance LOSS, collapses to 0.219 AUC-PR
  → GNN NOT suitable for production without drift mitigation

RQ4: What actionable indicators support analysts?
  ✓ Screen resolution, payment mode, category → top SHAP drivers
  ✓ Three-tier decision architecture routes cases appropriately
  ✓ Local SHAP explanations provided per prediction

Overarching Conclusion:
  Complexity must be justified by operational reliability, not peak metrics.
  Vanilla XGBoost is the production recommendation.
  GNN is a powerful forensic tool, not yet a real-time classifier.
```

**[SCRIPT]**
Let me close by directly answering each research question.

RQ1: Yes, GNN embeddings add predictive signal. The 17.3% feature importance is real, and the 13 GNN-exclusive detections of organized fraud ring cases demonstrate genuine complementary capability. However, in a fixed-split evaluation, the aggregate performance difference is not statistically distinguishable from noise.

RQ2: The graph analysis revealed a nuanced and counter-intuitive topology. Fraud rings in real estate do not form the dense interconnected clusters you see in e-commerce fraud literature. Instead, the Heterophily Paradox means that high connectivity signals legitimacy for established property agents. The fraud signal comes from the specific nature of connections — particularly device super-connectors, geographic clustering, temporal signatures at 6 AM, and cross-platform behavior.

RQ3: Under temporal drift, XGBoost retains 86% of its performance. The GNN loses 70%. This is the decisive result. Two models that look identical in the fixed-split setting are revealed to have fundamentally different temporal robustness profiles when evaluated under realistic operational conditions.

RQ4: The SHAP analysis provides actionable, explainable insights that support both automated triage and human analyst review. Browser fingerprinting, payment mode, listing category, and network signals are the dominant factors — all with clear operational interpretability.

The overarching conclusion: research that optimizes purely for peak performance metrics may produce models that fail in the field. This thesis deliberately chose operational reliability over marginal performance gains. Vanilla XGBoost is the production recommendation.

---

## SLIDE 22 — Limitations and Future Work

**[SLIDE]**
```
Limitations
  1. Statistical power: 3 fixed-split runs; differences not significant (p=0.92)
  2. Ground truth bias: labels from existing moderation system
  3. Generalizability: Swiss real estate only (12 months, single market)
  4. GNN receptive field: 2-hop neighbor sampling limits long-range detection
  5. Ethical exposure: geographic features carry discrimination risk

Future Work
  Short-term:
  • Graph-aware drift detection (topology change monitoring)
  • Shadow deployment: run model alongside SEON in production
  • Decision threshold optimization via business cost analysis

  Medium-term:
  • Real-time GNN inference (streaming graph updates)
  • Cross-platform fraud pattern federation
  • Active learning: analyst feedback → continuous label refinement

  Long-term:
  • Temporal Graph Networks (TGN/TGAT) for time-aware architectures
  • Transfer learning across European marketplaces
  • Adversarial robustness analysis (GNN camouflage resistance)
  • Heterophily-aware architectures exploiting inverse homophily structure
```

**[SCRIPT]**
Let me be transparent about the limitations. The statistical validation is limited — three runs per model variant is not a deep evaluation, and the p-value of 0.92 on the fixed split means we cannot make strong claims about GNN superiority even there. With more computation, a broader grid of runs, we might get cleaner confidence intervals.

The ground truth carries the biases of the existing moderation system. Labels exist only for fraud that was previously detected. Sophisticated evasion tactics that have never been caught remain unlabeled as legitimate. This means the model is, to some degree, learning to detect the types of fraud the moderation team already knows how to find.

The dataset is geographically and temporally limited. Whether these findings transfer to a German, French, or Southeast Asian real estate marketplace is an open question. Temporal patterns specific to Swiss market dynamics may not generalize.

For future work, the highest-priority item is graph-aware drift detection. The current system can detect when model performance degrades but cannot distinguish between tabular drift and graph topology drift. A dedicated topology monitoring system would allow targeted GNN retraining triggers rather than blanket monthly retraining. Shadow deployment — running the model in parallel with SEON in production without actioning its predictions — would generate the ground truth needed to calibrate real-world performance without operational risk. And temporal graph networks, which explicitly model time as part of the graph structure, would directly address the GNN's temporal brittleness.

---

## SLIDE 23 — Closing

**[SLIDE]**
```
Thank You

Key Takeaways for Practitioners:
  1. Evaluate under operational conditions, not just fixed splits
  2. Complexity is only justified by sustained, reliable improvement
  3. Explainability is not optional in fraud detection
  4. GNNs are powerful forensic tools; XGBoost wins in production today

The research code, Hydra configs, MLflow experiments,
and thesis LaTeX source are fully reproducible.

Questions?
```

**[SCRIPT]**
Thank you. I'd like to leave you with four practitioner takeaways from this research.

First: always evaluate models under the operational conditions they will face — not just on the most favorable fixed dataset evaluation. The protocol matters as much as the model.

Second: complexity must be justified by reliable, sustained improvement. The GNN is a sophisticated system with genuine capabilities. But if those capabilities do not survive the retraining cycles of a production deployment, they are academic.

Third: in fraud detection specifically, explainability is not a nice-to-have. Analysts need reasons. Fraudsters deserve appeal processes. Regulators require audit trails. SHAP is not overhead — it is part of the system's correctness.

Fourth: the GNN's story is not over. Temporal graph networks, adversarial robustness work, and better drift detection could make graph-based real-time fraud detection viable. This thesis establishes the baseline and the conditions under which that viability needs to be demonstrated.

I am happy to take questions.

---

---

# APPENDIX: QA PREPARATION

## Category 1 — Model Choice Challenges

**Q: Your GNN and XGBoost achieve the same 0.733 AUC-PR in the fixed split. Does that mean the GNN adds nothing?**

A: Not exactly. In aggregate fixed-split performance, they are statistically indistinguishable (p=0.92). However, the disagreement analysis shows 13 cases caught only by the GNN, compared to 1 caught only by XGBoost. Those 13 cases have device connection counts averaging 3,565 — four times the dataset average — and represent organized fraud ring activity invisible to tabular models. So the GNN adds qualitative value at the tail of the distribution, even if aggregate AUC-PR does not separate. The operational recommendation is not that the GNN is useless — it is that the GNN is not yet reliable enough for real-time production scoring given the expanding window results.

**Q: You tuned XGBoost with 50 Optuna trials but the GNN with only 20. Is the comparison fair?**

A: This is a valid concern. The GNN HPO was memory and time constrained — each GNN trial takes 5-10 minutes versus 30 seconds for XGBoost, so 50 GNN trials would be computationally prohibitive without dedicated GPU infrastructure. However, the expanding window results do not change if we give the GNN better hyperparameters — the fundamental problem is that the learned graph topology does not generalize across temporal distribution shifts, not that the specific GNN hyperparameters are suboptimal. We also ran GNN HPO across separate axes: joint GNN+XGBoost tuning (hpo_gnn_xgboost.yaml, 50 trials), GNN-only with fixed best XGBoost params (hpo_gnn_only.yaml, 20 trials), and HGT encoder search separately (hpo_hgt.yaml, 20 trials). The results were consistent.

**Q: Why did you not try a transformer-based model for the tabular data?**

A: TabNet, FT-Transformer, and similar architectures were considered. The literature on fraud detection consistently shows XGBoost with careful hyperparameter tuning as the dominant approach for tabular fraud data — this is validated in the Kaggle fraud benchmarks and IEEE-CIS survey. Given the class imbalance, sparse Boolean features, and the operational constraints, gradient boosted trees remain the strongest baseline. For this work, adding another tabular architecture would have expanded the comparison matrix without directly addressing the core RQ about graph-based methods. It is noted as future work.

---

## Category 2 — Evaluation Methodology

**Q: Your cross-validation gives 0.278 AUC-PR but your fixed split gives 0.733. That is a 62% gap. Should we trust the 0.733 result at all?**

A: The gap is real and important to contextualize. The 5-fold temporal cross-validation partitions the data into five time-ordered segments and evaluates each in sequence. This surface's distribution shift between early and late periods — fraud rates declined from about 8% in early 2025 to about 4.5% by mid-2025, which means test sets in later folds have much sparser fraud. A model trained on high-fraud periods will appear to "degrade" on low-fraud test periods even if it is correctly rank-ordering. The fixed-split evaluation uses a dedicated, held-out test period with a 7-day label maturation gap that more closely simulates the actual deployment scenario. The expanding window protocol sits between these two: it cycles over operationally realistic retraining scenarios and shows 0.629 mean AUC-PR, which I consider the most operationally honest estimate. The 0.733 is an upper bound under favorable distribution conditions; 0.629 is a more realistic production estimate.

**Q: Why use AUC-PR as the primary metric rather than F1 or AUC-ROC?**

A: AUC-ROC is insensitive to class imbalance because it weights true positive and false positive rates equally regardless of class proportions. In a dataset where fraud is 8.11% at listing level and 1.21% at event level, AUC-ROC will appear high even for poor models because there are so many legitimate events. AUC-PR directly measures performance on the minority class — it is the area under the precision-recall curve and rewards models that achieve high precision AND high recall simultaneously against the fraud class. F1 is a single operating point; AUC-PR summarizes performance across all thresholds. For a fraud detection system where the operating threshold is a business decision (not a fixed model property), threshold-independent AUC-PR is the correct primary metric. The Precision@K results add interpretability for the analyst workload dimension.

---

## Category 3 — Graph Construction and GNN

**Q: You have 31 million edges. Is that graph biologically plausible? Could it be overcrowded with noise?**

A: The dominant edge type is device-sharing with 12 million edges. This high volume reflects the fact that many listings on the platform come from the same device fingerprint — both because of legitimate letting agencies posting many properties and because of fraudsters. The density is real, not artifactual. We tested aggressive edge filtering — keeping only edges between nodes with above 15% historical fraud rate, reducing from 31M to 8M edges — and the performance change was -0.3%. This suggests the model is already learning to weight edges appropriately and manual filtering adds no signal.

**Q: The Heterophily Paradox is counter-intuitive. Can you explain in more detail why dense device connectivity predicts legitimacy?**

A: In classic fraud detection literature — credit card fraud, e-commerce fraud — fraudsters create many accounts sharing the same device because they need volume. But in the real estate rental market, the buyer side is individuals looking for one apartment. The supply side — listing creators — includes legitimate property management companies that manage hundreds of properties from a single corporate device or account. A large letting agency in Zurich might have one administrative device that submits 500 listings per year. That device has 500+ links and a near-zero fraud rate. Meanwhile, a fraudster who creates a single fake listing and abandons the device appears as a single-listing node — which is precisely the pattern that correlates with higher fraud probability. The model must learn this inverse relationship, which is why supervised training of the GNN outperforms self-supervised: the fraud labels are needed to orient the message-passing toward this non-intuitive pattern.

**Q: Why GraphSAGE over a simpler GCN?**

A: Traditional GCN is a transductive method — it learns a fixed embedding for each node during training, and cannot embed nodes not seen during training. A production fraud detection system must score new listings as they arrive. GraphSAGE learns an *inductive* aggregation function that can produce embeddings for new nodes by aggregating their neighbors. It is the minimum complexity GNN variant that supports this operational requirement. The more complex architectures — HGT, CARE-GNN — were evaluated and performed worse, as I detailed in the architecture comparison slide.

---

## Category 4 — MLOps and Reproducibility

**Q: How confident are you that the results are reproducible?**

A: Every training run is logged to MLflow with: the complete Hydra configuration, all hyperparameters, random seeds, the exact training and test date boundaries, every metric, and the serialized model artifact. The Makefile provides a single command to reproduce any run — for example, `make train-vanilla` runs the exact configuration used for the fixed-split XGBoost evaluation. The ETL pipeline is deterministic. The only source of non-determinism is XGBoost's internal tree construction which has minor seed-dependent variation — this is why we report mean ± standard deviation over 3 independent runs rather than single-run results.

**Q: Why Hydra over a simpler approach like argparse or environment variables?**

A: Hydra provides hierarchical, composable configuration files with type checking and override syntax. The key benefit for research is configuration immutability at run time — once the run starts, the configuration is locked and logged. With argparse or environment variables, it is easy to accidentally launch a run with the wrong settings and not realize until the results look unexpected. Hydra also enables configuration composition: the hpo_*.yaml files override subsets of experiment.yaml, which means HPO searches are guaranteed to use the same base configuration as the manual runs they are comparing against. This is critical for experimental validity.

---

## Category 5 — Production and Business

**Q: How would you handle concept drift in production beyond monthly retraining?**

A: Three layers are in place or recommended. First, the DDM (Drift Detection Method) and ADWIN monitors run as online drift detectors on the model's prediction error stream. When DDM triggers a 3-sigma warning, it initiates an unscheduled retraining cycle — this caught the March 2025 drift event in the expanding window simulation. Second, the MLflow Registry tracks model versions with promotion history, so rollback to a previous stable version is possible if a newly trained model underperforms. Third, the API exposes `/model-info` and `/health` endpoints that include store statistics and drift state — these can feed a monitoring dashboard that alerts the operations team to unusual patterns in approval/review/decline rate distributions.

**Q: The thresholds you chose — 0.7 and 0.3 — feel arbitrary. How were they selected?**

A: They are operationally motivated starting points, not model-tuned thresholds. The Precision@K analysis shows that at the 0.5 probability threshold, XGBoost achieves 74.6% precision and 67.1% recall. Lowering to 0.3 as the review trigger gives more recall at the cost of larger analyst queues. Raising the DECLINE threshold to 0.7 reduces false positives in automated rejection. The exact values should be calibrated by the business against the cost ratio of missed fraud versus false positives for legitimate listers. The recommendation is to establish this cost function with the operations and legal team and then run a threshold optimization sweep using the production data once the model is in shadow deployment. The thresholds are environment variables precisely so this tuning requires no code deployment.

**Q: SEON already has 81.18% precision at maximum recall. Why would the marketplace need your model?**

A: SEON's high-precision operating point uses the SEON State DECLINE category — effectively their internal blacklist-based hard rules. That achieves 99.87% recall but at the cost of extensive human review of every borderline case. Our model produces a continuous probability score that enables *prioritized* review: at P@100 = 97.7%, an analyst reviewing the top 100 predictions can be confident that nearly every case is genuine fraud. The value is not in replacing SEON but in stacking: SEON provides the first-pass rule-based screen, our model provides the precision-focused second-layer scoring that enables intelligent review queue management. A fraud operations team working 40 cases per day is far more effective if those 40 cases have 97% fraud probability than if they are drawn from an unsorted pool with 8% base rate.

---

## Category 6 — Data and Features

**Q: You removed the STATUS column as a label leakage source. How do you know you didn't remove other leaked features?**

A: The STATUS column was identified through causal analysis: it changes value *after* fraud detection — a listing transitions to STATUS=ARCHIVED or STATUS=DELETED after the moderation team removes it, not before. No feature in the remaining 65 has this causal structure — they are all recorded at event ingestion time, before any moderation action. The SEON signals are computed at the time of the listing event. The listing attributes — price, category, location — are set when the listing is created. The temporal features are derived from the event timestamp. The potential remaining concern is features that correlate with post-hoc moderation signals without being causally derived from them — for example, if listings that stay active longer have lower fraud rates because the moderation team removes fraudulent ones. This is addressed by the point-in-time labeling protocol: the fraud label is propagated only to events that occurred *before* the first moderation action on that listing, not to subsequent events.

**Q: 509 raw features reduced to 65. What happened to the other 444?**

A: The 509 raw columns include many redundant, near-zero-variance, or data-leaking fields. The feature engineering process removed: free-text fields not suitable for tree-based models without embedding, highly sparse identifier fields with cardinality above threshold, columns that are post-hoc records of moderation actions, columns with above 90% missing rate in the training period, and columns determined to be functionally redundant with lower-cardinality equivalents. The 65 retained features represent all unique information content in the dataset that can be computed at event ingestion time without access to future events.

---

*End of Defense Presentation Script and QA Guide*
*Total estimated delivery time: 25–30 minutes*
*Prepared for: FPT School of Business and Technology Defense Council*
