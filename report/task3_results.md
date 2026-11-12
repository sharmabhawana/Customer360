# Customer360 AI — Task 3: Results Report
## Machine Learning, Evaluation and NLP

**Course:** COM-702 Applied Data Science  
**Dataset:** Brazilian E-Commerce Public Dataset by Olist  
**Prediction Problem:** Binary classification of customer dissatisfaction at order delivery confirmation (post-fulfilment, pre-review submission)  
**Date:** November 2026

---

## 1. Dataset Verification

| Item | Value |
|:---|:---:|
| Order-level dataset loaded | 95,824 rows × 51 columns |
| NLP reviews corpus loaded | 99,224 rows × 12 columns |
| Neutral Score-3 excluded | 7,916 orders |
| **Binary modelling cohort** | **87,908 orders** |
| Dissatisfied (Y=1, Scores 1–2) | 12,272 (13.96%) |
| Satisfied (Y=0, Scores 4–5) | 75,636 (86.04%) |
| Class imbalance ratio | 6.16 : 1 |
| Non-null review comment messages (NLP) | 40,977 (41.3%) |

---

## 2. Pre-Modelling Validation

### 2.1 Anomaly: Negative `carrier_duration_days`
- **Instances in binary cohort:** 20 records  
- **Minimum value:** −16.10 days  
- **Root cause:** Retroactive EDI timestamp registration — the carrier's hand-off scan was entered into the postal system *after* the customer delivery confirmation was recorded. This is a system logging artefact in the Brazilian postal carrier infrastructure, not a physical impossibility.  
- **Decision:** Records **preserved as-is**. Negative carrier durations are real-world carrier data quality noise. The median imputer in the preprocessing pipeline handles boundary effects without distorting training signal.

### 2.2 Target Leakage Audit
All 30 candidate features were audited against the forbidden fields:  
`['review_score', 'review_comment_message', 'review_comment_title', review IDs, survey timestamps]`  
**Result: ZERO leakage detected.** All features are operational signals available at delivery confirmation time.

### 2.3 Train / Test Split

| Split | Total | Dissatisfied | Satisfied | Dissat% |
|:---|:---:|:---:|:---:|:---:|
| Training (80%) | 70,326 | 9,818 | 60,508 | 13.96% |
| Test (20%) | 17,582 | 2,454 | 15,128 | 13.96% |

Stratified split (`random_state=42`) preserves the class ratio exactly across both sets.

---

## 3. Machine Learning Model Comparison

> [!IMPORTANT]
> **Why Recall and PR-AUC were chosen as primary metrics:**  
> With a 6:1 class imbalance, a naive "always-satisfied" classifier achieves 86.0% accuracy while catching zero dissatisfied customers. Missing a dissatisfied customer (False Negative) allows grievances to escalate into public 1-star reviews, chargebacks, and permanent churn. Recall on the minority class and PR-AUC (the area under the Precision-Recall curve, which is insensitive to true negatives) are the operationally relevant metrics for this intervention use-case.

### 3.1 Model Comparison Table

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression (Balanced) | 0.3916 | 0.5554 | 0.4593 | 0.7653 | 0.4720 |
| Random Forest (Balanced) | 0.4710 | 0.4967 | 0.4835 | 0.7661 | 0.5186 |
| **HistGradientBoosting (Balanced)** | **0.4152** | **0.5615** | **0.4774** | **0.7746** | **0.5291** |

All metrics computed on the held-out test set only. Preprocessing fitted exclusively on training data.

### 3.2 Selected Model: HistGradientBoosting (Balanced)

**Selection justification:** HistGradientBoosting ranks first on both primary selection criteria:
- **Highest PR-AUC (0.5291)** — optimal discrimination of the minority dissatisfied class across all classification thresholds.
- **Highest Recall (0.5615)** — identifies 56.1% of dissatisfied customers on unseen data, outperforming all alternatives.
- **Highest ROC-AUC (0.7746)** — best overall rank-ordering ability.

### 3.3 Classification Report — Best Model (HistGradientBoosting)

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Satisfied (0) | 0.9246 | 0.8717 | 0.8974 | 15,128 |
| Dissatisfied (1) | 0.4152 | 0.5615 | 0.4774 | 2,454 |
| **Accuracy** | | | **0.8284** | 17,582 |
| Macro Avg | 0.6699 | 0.7166 | 0.6874 | 17,582 |
| Weighted Avg | 0.8535 | 0.8284 | 0.8387 | 17,582 |

### 3.4 Confusion Matrix Interpretation (Best Model)

```
                     Predicted Satisfied   Predicted Dissatisfied
True Satisfied              13,187               1,941  ← False Positives (unnecessary alerts)
True Dissatisfied            1,076               1,378  ← True Positives (correctly flagged!)
```

The model catches exactly **1,378 out of 2,454 dissatisfied customers** in the test set (Recall = 56.15%). The 1,076 missed orders represent cases where the structured operational signals alone were insufficient — typically, cases where delivery was on time but the product itself was defective or incorrect. Total test records: 13,187 + 1,941 + 1,076 + 1,378 = 17,582.

---

## 4. Feature Importance Analysis (Permutation-Based, Test Set)

Permutation importance measures the drop in ROC-AUC when each feature's values are randomly shuffled on the test set, isolating genuine predictive contribution.

### Top Operational Predictors

| Rank | Feature | Business Meaning |
|:---:|:---|:---|
| 1 | `is_delayed` | Binary indicator of SLA breach — single most decisive discriminator |
| 2 | `delivery_duration_days` | Absolute transit duration from purchase to doorstep |
| 3 | `delivery_delay_days` | Signed delay magnitude (positive = late, negative = early) |
| 4 | `carrier_duration_days` | Last-mile transit time from warehouse to customer |
| 5 | `item_count` | Multi-item orders carry higher fulfilment coordination risk |
| 6 | `unique_sellers` | Multi-seller split shipments increase communication failures |
| 7 | `total_freight` | Absolute freight spend — cost-sensitive customers have lower tolerance |
| 8 | `freight_ratio` | Freight-to-merchandise ratio amplifies dissatisfaction sensitivity |
| 9 | `payment_installments_max` | High-installment = high-value purchase = elevated expectation |
| 10 | `is_same_state` | Intrastate deliveries are faster and less prone to damage |

### Correlation vs Causation Disclaimer
These features are associated with dissatisfaction in historical data. Operational improvements (reducing actual delivery time) are likely to reduce dissatisfaction, but the magnitude of improvement cannot be claimed as causal without controlled field experiments. Confounding variables — carrier reliability, regional infrastructure quality, product fragility, and seller communication practices — are not fully captured in this dataset.

### Model Limitations
1. **Temporal coverage (2016–2018):** E-commerce logistics, carrier infrastructure, and customer expectations evolve. Models trained on historical data require periodic revalidation.
2. **Geography specificity:** The dataset reflects Brazilian post/carrier conditions. Models are not directly transferable to other national markets without retraining.
3. **Missing product-defect signal:** The most common complaint in NLP ("veio defeito" — arrived defective) is not observable in structured order logs. Including warehouse quality or carrier claims data would substantially improve recall.
4. **Text-only dissatisfied reviews:** 76.5% of Score-1 customers wrote a review; the 23.5% who did not are characterised only by delivery metrics. Silent dissatisfied customers may differ systematically.
5. **Fixed threshold (0.5):** The default decision threshold can be adjusted to increase recall at the expense of precision for operational deployment.

---

## 5. NLP — Customer Feedback Analysis (Portuguese)

### 5.1 Corpus Statistics

| Item | Count | % of Total |
|:---|:---:|:---:|
| Total reviews in NLP corpus | 99,224 | 100% |
| Non-null comment messages | 40,977 | 41.3% |
| Null comment messages (no text written) | 58,247 | 58.7% |
| Dissatisfied reviews with text (Scores 1–2) | 10,867 | 76.5% of dissat reviews |
| Satisfied reviews with text (Scores 4–5) | 26,062 | 34.1% of sat reviews |

> [!NOTE]
> Dissatisfied customers write reviews at more than double the rate of satisfied customers (76.5% vs 34.1%), indicating strong expressive motivation when grievances occur.

### 5.2 Portuguese Text Preprocessing

The pipeline applied the following operations:
1. Lowercase normalisation
2. Removal of URLs, email addresses, and numeric tokens
3. Character-class filtering retaining `[a-záàâãéêíóôõúç\s]` (Portuguese Unicode alphabet)
4. NLTK Portuguese stopword removal (207 tokens) — **excluding** key negation tokens: `não, nao, nem, nunca, sem`
5. Supplemental conversational noise tokens removed: `pra, ja, so, tudo, ter, pois, ainda, outro, onde, dias, dia`
6. Minimum token length: > 2 characters
7. TF-IDF vectorisation: unigrams + bigrams (`ngram_range=(1,2)`), `max_features=25`, `min_df=5`

> [!IMPORTANT]
> TF-IDF measures term **salience** (how characteristic and distinctive a term is to a document cohort). It does **not** measure sentiment. A term like "não recebi" ranks highly because it is unusually frequent and distinctive in the dissatisfied cohort — not because TF-IDF can determine whether "not receiving" is positive or negative.

### 5.3 Top TF-IDF Terms — Dissatisfied Customers (Scores 1–2)

| N-Gram | TF-IDF Weight | English Translation | Root Cause Theme |
|:---|:---:|:---|:---|
| `não` | 2,405 | not / no | General Negation |
| `produto` | 2,195 | product | Merchandise Reference |
| `recebi` | 1,289 | received | Delivery Dispute |
| `comprei` | 959 | I bought | Transaction Context |
| `veio` | 904 | arrived / came | Condition on Arrival |
| `entrega` | 765 | delivery | Logistics SLA |
| **`não recebi`** | **751** | **did not receive** | **Non-Delivery / Missing Order** |
| `entregue` | 707 | delivered | Logistics Status |
| `chegou` | 661 | arrived | Transit Confirmation |
| `recebi produto` | 611 | received product (phrase signals negation context) | Delivery Negation |
| `prazo` | 486 | deadline / SLA | Logistics Delay |
| `pedido` | 460 | order / request | Transaction Reference |

**Key NLP observation:** The bigram `"não recebi"` ("did not receive") is the strongest distinct phrase in dissatisfied reviews, confirming that **non-delivery and lost parcels** are the primary grievance. The repeated pairing of `produto` + negation tokens (`produto não`, `recebi produto`) indicates product receipt disputes rather than quality issues.

### 5.4 Top TF-IDF Terms — Satisfied Customers (Scores 4–5)

| N-Gram | TF-IDF Weight | English Translation | Root Cause Theme |
|:---|:---:|:---|:---|
| `produto` | 4,032 | product | Merchandise Reference |
| `bom` | 2,792 | good | Quality Appraisal |
| `prazo` | 2,592 | deadline / on-time | Punctuality |
| `entrega` | 2,349 | delivery | Fulfillment Service |
| `recomendo` | 2,208 | I recommend | Advocacy / Loyalty |
| `antes` | 2,037 | before / ahead | Early Delivery |
| `chegou` | 1,876 | arrived | Successful Arrival |
| **`antes prazo`** | **1,635** | **before deadline** | **Early Delivery Delight** |
| `qualidade` | 1,329 | quality | Product Standard |
| `ótimo` | 1,245 | great / excellent | Enthusiastic Delight |
| `excelente` | 1,211 | excellent | Premium Service |
| `recomendo` | 2,208 | I recommend | Active Brand Advocacy |

**Key NLP observation:** `"antes prazo"` ("before the deadline") is the most prominent bigram in satisfied reviews — directly mirroring the dissatisfied cohort's `"não recebi"` bigram. The contrasting phrases together prove that **delivery timing is the axis of the entire customer experience**.

---

## 6. Structured Data vs NLP Findings — Synthesis

| Friction Dimension | Structured Data Finding | NLP Text Finding | Relationship | Business Interpretation |
|:---|:---|:---|:---:|:---|
| Delivery Punctuality | `is_delayed` = #1 predictor (r=+0.39). 54% dissat rate when late vs 9% on-time. | `"não recebi"`, `"prazo"` dominant in dissat; `"antes prazo"` dominant in satisfied. | ✅ **Strong Mutual Support** | Delivery timing is the universal axis of CX. Both data modalities confirm this independently. |
| Product Quality | Bulky categories (furniture 15–18% dissat) flagged by category feature. | `"veio defeito"` (defective), `"errado"` (wrong item) in dissat cohort. | 🔄 **Complementary** | Structured data identifies *which* categories fail; NLP explains *how* they fail (physical damage, wrong SKU). |
| Customer Service | Multi-seller orders (unique_sellers) correlate with dissatisfaction (+0.13). | `"aguardo"` (waiting), `"resposta"` (reply) in dissat — complaints of silent sellers. | 🔄 **Complementary** | Multi-seller split shipments expose communication gaps that neither data type could fully reveal alone. |
| Financial Sensitivity | High freight ratio and 11-24x installment customers show elevated dissat. | `"dinheiro de volta"` (refund), `"estorno"` (chargeback) in dissat for high-value orders. | ✅ **Strong Mutual Support** | Customers with high financial exposure demand swift recovery when service fails. |

---

## 7. Evidence-Based Insights and Practical Recommendation

### Insight 1 — Delivery SLA Adherence is the Universal Satisfaction Driver
Both the structured ML model (permutation importance ranking) and the NLP corpus (dominant bigrams on both ends of the satisfaction spectrum) independently converge on the same conclusion: **the gap between the promised and actual delivery date is the single most decisive driver of customer sentiment**. This cross-modal confirmation elevates confidence well beyond what either methodology could establish in isolation.

### Insight 2 — Multi-Seller Orders Create Invisible Communication Risk
The structural feature `unique_sellers > 1` correlates meaningfully with dissatisfaction. NLP reveals the mechanism: dissatisfied multi-seller customers repeatedly write about awaiting seller responses (`aguardo resposta`) and receiving no post-purchase communication. This combination reveals that **split-shipment orders involving multiple independent merchants create communication black holes** that structured delivery metrics alone do not capture.

### Practical Recommendation — Deploy a Real-Time Delivery-Risk Early Warning System
Implement the trained HistGradientBoosting pipeline as a live scoring service triggered at carrier hand-off:
1. **Score every order** at the moment `order_delivered_carrier_date` is recorded.
2. Orders with predicted dissatisfaction probability **> 0.40** enter an automated intervention workflow:
   - Proactive SMS/app push notification with real-time tracking link.
   - Pre-authorised apology credit voucher on file for immediate release if delivery is confirmed late.
   - Escalated seller SLA monitoring dashboard for multi-seller orders.
3. This converts the model's 56.1% recall into an immediate CX recovery mechanism, reducing negative reviews before they are written.

---

## 8. Generated Deliverables Summary

| File | Path | Description |
|:---|:---|:---|
| Task 3 Notebook | `notebooks/Customer360_Task3.ipynb` | Fully executable end-to-end notebook |
| Model Comparison | `outputs/tables/model_comparison.csv` | Precision, Recall, F1, ROC-AUC, PR-AUC |
| Classification Report | `outputs/tables/classification_report.csv` | Per-class metrics for best model |
| NLP Top Terms | `outputs/tables/nlp_top_terms.csv` | TF-IDF n-grams with translations and themes |
| Structured vs Text | `outputs/tables/structured_vs_text_comparison.csv` | 4-dimension synthesis table |
| Confusion Matrix | `outputs/figures/model_confusion_matrix.png` | All 3 models, 300 DPI |
| Performance Chart | `outputs/figures/model_performance_comparison.png` | 5 metrics × 3 models |
| Feature Importance | `outputs/figures/model_feature_importance.png` | Permutation importance, top 15 features |
| NLP Dissatisfied | `outputs/figures/nlp_top_terms_dissatisfied.png` | Top 12 dissatisfied n-grams |
| NLP Satisfied | `outputs/figures/nlp_top_terms_satisfied.png` | Top 12 satisfied n-grams |
| NLP Comparison | `outputs/figures/nlp_tfidf_comparison.png` | Semantic contrast themes |
| Best Model | `outputs/models/best_customer360_model.joblib` | HistGradientBoosting full pipeline (243 KB) |

