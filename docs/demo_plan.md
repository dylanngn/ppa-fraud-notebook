# Presentation Demo Plan: Real Estate Fraud Detection Framework

Since running the live models directly on the company system during the council presentation is not permitted, this document serves as a structured script and simulated software demonstration plan. You can use screenshots, architectural diagrams, and pre-computed metrics/plots to walk the council through the software's capabilities.

---

## 🏗️ Part 1: Architecture & The Core Pipeline

**Objective:** Explain *how* the framework operates securely and independently from a high-level operational perspective.

**Talking Points:**
1. Show the **Feature Flow Diagram** (from `docs/architecture.md`). Explain that our input is a real-time event pipeline merging internal `insertion_events` with third-party `seon_transactions`.
2. Emphasize **Data Privacy & Anonymization:** Explain that the first step hashes all PII (`LISTING_LISTER_EMAIL`, `ip`, `session/device_hash`) using SHA-256 to ensure no raw user data touches the core models.
3. Call out the **Point-In-Time (PIT)** aspect: Fraud is caught using only information strictly known at the specific event timestamp, removing critical label leakage traps (like the downstream `STATUS` identifier which we removed after analysis).

---

## 🔎 Part 2: Fraud Inference Execution (The 'Submission' Check)

**Objective:** Demonstrate exactly *when* the model runs and how classification occurs.

**Talking Points:**
1. **Initial Submission Evaluation:** The software is designed to execute immediately during the **first submission** (when `STATUS = DRAFT`). The moment a lister attempts to draft a real estate property, an inference call is made. 
2. **Why not every event?** We process the first attempt explicitly because our exploratory pattern discovery highlighted that 75% of fraudulent attempts are captured natively at draft generation via anomalous signals (e.g., VPN usage out of unusual geographic clusters like West Africa with spoofed device fingerprints like "Desktop as TV").
3. **Model Highlight:** We utilize the **Vanilla XGBoost** variant. Despite building a massive Heterogeneous Graph (GNN) structure, computational complexity and temporal shift instability proved XGBoost with robust tabular integration features to be the most responsive for this real-time deployment.

---

## 📈 Part 3: Explainable AI via SHAP Analysis

**Objective:** Show the council that the software isn't just a black box; it gives actionable intelligence to fraud analysts.

**Talking Points:**
1. Discuss the integrated `src/evaluation/shap_analysis.py` module. Explain that every positive fraud flag is accompanied by a local waterfall plot calculation.
2. **Feature Importance (Global):** Show the `global_importance_summary.png`. Point out that native `session/screen_resolution`, `vpn`, and real estate `LISTING_CATEGORIES` (e.g., Studio vs Single_Room) dominate the prediction context.
3. **Local Waterfall Scenarios:** Present a sample (pre-rendered screenshot) of a true-positive local SHAP explanation showing *exactly why* listing X was flagged:
   - "This listing was flagged because the IP originated from Benin (`ip_country=BJ`), the user is mapping multiple drafts aggressively under `email/domain/disposable=True`, and it was scoped as a high-risk Studio apartment format." This grounds the research in actionable, operational value.

---

## 🕰️ Part 4: Concept Drift & Continuous Evaluation

**Objective:** Outline how the software handles the most critical challenge in fraud detection: evolving attacker tactics over time.

**Talking Points:**
1. Fraudsters adapt. Show the **Expanding Window Pipeline** logic handled inside `src/evaluation/drift.py`.
2. Explain that fraud rates spiked in February (13.12%) but plummeted significantly in June (4.56%). A static model degrades by nearly 70% in predictive power (AUC-PR drop down to ~0.2) when facing these temporal distribution shifts.
3. **Software Action:** The pipeline systematically retrains itself on expanding 30-day temporal windows. If the tracked `AUC-PR` triggers an alert (dips under the acceptable threshold, like 0.70), it initiates an automated alert for the data science team. 

---

## 🎯 Conclusion for Council

Conclude by reinforcing your core research contribution: 
*"While Graph Neural Networks accurately identify and map intricate real estate fraud rings retrospectively, for an active Marketplace production system facing volatile conceptual drift and latency constraints, a purely tabular XGBoost model paired with comprehensive automated alert monitoring (Drift Pipeline) and Analyst explainability (SHAP) presents the optimal software solution."*
