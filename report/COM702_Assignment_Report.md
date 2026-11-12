> **Formatting Note:** This document is intended for submission as Times New Roman 12pt, 1.5 line spacing, 1-inch margins on all sides.

---

# Customer360 AI: Predicting E-Commerce Customer Dissatisfaction Using Machine Learning and Understanding Customer Feedback Through Natural Language Processing

---

## Title Page

**Assignment Title:** Customer360 AI: Predicting E-Commerce Customer Dissatisfaction Using Machine Learning and Understanding Customer Feedback Through Natural Language Processing

**Course:** COM-702 (C) — Applied Data Science

**Institution:** Model Institute of Engineering and Technology (MIET), Jammu

**Session:** 2023–2024

**Submission Date:** November 2026

**Group Members:**

| Name | Roll Number |
|---|---|
| Bhawana Sharma | 2023A1R076 |
| [Group Member 2] | [Roll No] |
| [Group Member 3] | [Roll No] |
| [Group Member 4] | [Roll No] |

---

## Abstract

This report presents Customer360 AI, an applied data science project that uses machine learning (ML) and natural language processing (NLP) to predict e-commerce customer dissatisfaction and characterise the linguistic signals within customer reviews. The dataset employed is the Olist Brazilian E-Commerce Public Dataset (Olist, 2018), comprising 99,441 orders across 2016–2018. After filtering for delivered and reviewed orders, the analytical cohort comprised 95,824 records. Of these, 12,272 orders (12.81%) received dissatisfied review scores (1–2). A binary classification model trained on 30 operational features identified a 5.86× surge in dissatisfaction rate for delayed orders (54.07%) relative to on-time orders (9.22%). Among three candidate classifiers, Histogram-based Gradient Boosted Trees (HistGBT) achieved the highest Precision-Recall Area Under the Curve (PR-AUC) of 0.5291, the primary metric selected to account for class imbalance. NLP analysis of 40,977 Portuguese review texts via TF-IDF vectorisation confirmed delivery failures as the dominant dissatisfied theme, while structured data modelling outperformed text-only classification (PR-AUC 0.5291 vs. 0.4720).

---

## 1. Introduction

The rapid expansion of e-commerce has fundamentally altered retail dynamics across emerging markets. Brazil, in particular, witnessed compound annual growth in online retail throughout the 2016–2018 period, driven by marketplace platforms that aggregate thousands of sellers under a single consumer-facing storefront (Olist, 2018). Within such ecosystems, customer satisfaction is paramount: a single dissatisfied experience can translate into negative reviews, reduced repeat purchase rates, and reputational harm that cascades across the platform (Saito & Rehmsmeier, 2015).

Proactively identifying orders likely to generate dissatisfied reviews — before those reviews are posted — creates an operational window in which interventions such as proactive communication, priority resolution, or compensation can be deployed. However, dissatisfied customers represent a minority class (12.81% of delivered and reviewed orders), making this an inherently imbalanced classification problem for which standard accuracy metrics are misleading.

This project, Customer360 AI, addresses three interrelated objectives: (1) to prepare and explore the Olist dataset to understand the structural determinants of customer dissatisfaction; (2) to build and compare ML classifiers that predict dissatisfaction from operational delivery features, before review scores are observed; and (3) to analyse the natural language of customer reviews through NLP to identify the linguistic themes associated with positive and negative experiences. The remainder of this report is structured to answer two formally assessed questions, followed by concluding remarks.

---

## 2. Question 1: Data Preparation and Exploratory Data Analysis

### 2.1 Dataset Source and Suitability

The Olist Brazilian E-Commerce Public Dataset (Olist, 2018), publicly available on Kaggle, is an anonymised, multi-table relational dataset capturing the full order lifecycle on the Olist marketplace. It consists of nine interlinked CSV files covering orders, customers, products, sellers, payments, logistics (carrier tracking), geolocation, and customer review responses. The dataset encompasses 99,441 orders placed between September 2016 and August 2018, spanning 32,951 unique products, 3,095 sellers, and customers distributed across all 27 Brazilian states and 74 product categories.

Several properties render this dataset particularly suitable for the present investigation. First, its longitudinal coverage (two full years) ensures temporal representativeness, capturing seasonal variation and platform maturation effects. Second, the inclusion of both structured operational data (delivery timestamps, freight values, payment methods) and unstructured customer review text in Portuguese enables a genuine multi-modal analytical approach. Third, the dataset's real-world provenance — it reflects authentic commercial transactions rather than synthetic or artificially balanced data — means that the class imbalance observed (approximately 1 dissatisfied for every 7 satisfied customers) is an intrinsic property of the phenomenon under study, not an artefact of sampling. Fourth, the granularity of logistics data, including both estimated and actual delivery dates, enables the precise computation of delivery delay metrics that are theoretically central to customer satisfaction research. These properties collectively justify the dataset's selection for a university-level applied data science investigation.

### 2.2 Data Cleaning and Validation

Data preparation proceeded through a systematic series of cleaning, deduplication, and feature engineering steps applied across the nine source tables prior to analytical modelling.

**Filtering for scope:** The analytical population was restricted to orders with a status of `delivered` and at least one associated customer review. Orders lacking a confirmed delivery record or review were excluded. This reduced the working cohort from 99,441 raw orders to 95,824 orders — the population for all subsequent analyses. A total of 2,965 records were excluded at this stage owing to missing actual delivery dates; these corresponded to non-delivered or cancelled orders and were therefore conceptually outside the scope of post-delivery satisfaction modelling. Imputation of delivery dates for non-delivered orders would have introduced substantive measurement error and was therefore rejected.

**Duplicate review resolution:** The reviews table contained 551 orders associated with more than one review record, arising from customers submitting revised or multiple responses to the platform's review invitation. A deterministic tie-breaking rule was applied: for each duplicated order, the review with the most recent `review_answer_timestamp` was retained, reflecting the customer's final stated satisfaction. This approach is preferable to random selection or averaging because it preserves the customer's last expressed sentiment.

**Anomalous delivery duration values:** Twenty records exhibited negative `carrier_duration_days` values — a physical impossibility that reflects a known real-world artefact in carrier logistics systems. Carriers occasionally perform retroactive Electronic Data Interchange (EDI) scans, recording a parcel's handover event after the fact, which can result in a negative arithmetic difference between scan timestamps. These records were preserved in the dataset, as removing them would introduce selection bias, but the negative values were handled through median imputation within the ML preprocessing pipeline to prevent distortion of model training.

**Missing product category labels:** A total of 610 product records lacked a category assignment in the source table. These missing labels were resolved by imputing category-level median weights for the affected product attributes (weight, dimensions), thereby retaining these records in the analysis without introducing category-level mean distortion.

**Join strategy and Cartesian avoidance:** The nine source tables were aggregated at the order level prior to joining, ensuring that one-to-many relationships (e.g., one order with multiple items) were collapsed into scalar summary features (e.g., item count, total freight value, mean price) before the final merge. Performing joins before aggregation would have produced a Cartesian product of rows, inflating record counts and introducing duplicate weighting.

The result of this cleaning pipeline was a single, analysis-ready order-level table with 51 derived columns, used as the basis for all EDA and ML steps.

### 2.3 Three Key EDA Patterns Discovered

Three substantively distinct patterns were identified during exploratory data analysis of the 95,824-order cohort.

**Pattern 1 — Delivery SLA failure as a dissatisfaction multiplier.** The most striking pattern revealed by EDA is the profound asymmetry in dissatisfaction rates between on-time and delayed orders. Of the 95,824 orders, 88,163 (92.0%) were delivered on or before the estimated delivery date and recorded a dissatisfaction rate (review scores 1–2) of 9.22%. The remaining 7,661 orders (8.0%) experienced a delivery delay, defined as actual delivery date exceeding the estimated delivery date, and recorded a dissatisfaction rate of 54.07%. The ratio of these two rates — 54.07 ÷ 9.22 — yields a surge factor of 5.86×. This means a customer whose order arrives late is 5.86 times more likely to leave a dissatisfied review than one whose order arrives on time. It is critical to note that this is a correlational finding: late delivery is strongly associated with dissatisfaction, but the analysis does not assert that delay is the sole causal mechanism, as confounding factors (product quality, seller communication) may co-occur with logistics failures.

**Pattern 2 — Strong bimodal polarisation of review scores.** The distribution of review scores across the five-point scale is not uniform or normally distributed; it is markedly bimodal. A total of 56,748 orders (59.2% of the 95,824-order cohort) received the maximum score of 5 stars, while 9,351 orders (9.8%) received the minimum score of 1 star. The intermediate scores — 2, 3, and 4 — collectively account for the remaining 29,725 orders (31.0%). This bimodal structure indicates that most customers either had an experience sufficiently positive to warrant the highest possible rating, or one sufficiently negative to warrant the lowest. The relative sparsity of intermediate scores is consistent with psychological research on rating behaviour, in which customers are more motivated to review when their experience was strongly valenced. For modelling purposes, this polarisation supports the exclusion of score-3 (neutral) orders from the binary classification cohort, as these orders represent ambiguous cases that are difficult to interpret as either satisfied or dissatisfied.

**Pattern 3 — Category-level heterogeneity in dissatisfaction rates.** Dissatisfaction is not uniformly distributed across product categories. Certain categories systematically generate higher proportions of dissatisfied reviews than the platform average (12.81%). The highest-dissatisfaction categories identified in the EDA are: `fashion_male_clothing` (22.9%), `office_furniture` (21.9%), `audio` (21.4%), and `home_comfort` (17.9%). These categories share physical characteristics — bulkiness, fragility, dimensional specificity — that make them inherently more susceptible to fulfilment failures such as damage in transit, dimensional mismatch with courier capacity, or extended delivery windows due to weight surcharges. This heterogeneity implies that a single platform-wide intervention is unlikely to be optimally efficient; targeted category-specific SLA review or packaging standards would address the structural root causes more precisely.

### 2.4 Visualisations and Interpretation

The EDA phase produced a series of analytical visualisations that substantiated the patterns described above.

**Figure 1 — Review Score Distribution (Bar Chart).** This figure plots the count of orders across all five review score levels. The chart illustrates the bimodal pattern described in Pattern 2: the bar for score 5 (56,748 orders) is visually dominant, dwarfing scores 2–4, while the bar for score 1 (9,351 orders) forms a secondary peak. The bars for scores 2 (2,921 orders), 3 (7,916 orders), and 4 (18,888 orders) are substantially smaller, confirming the polarisation structure.

> *Figure 1: Distribution of customer review scores across 95,824 delivered and reviewed orders. Caption appears below the figure in the submitted document.*

**Figure 2 — Dissatisfaction Rate by Delivery Status (Grouped Bar Chart).** This figure contrasts the dissatisfaction rate for on-time (9.22%) versus delayed (54.07%) orders using a grouped or side-by-side bar format. The 5.86× magnitude difference is immediately visually apparent, underscoring delivery punctuality as the single most discriminating operational variable in the dataset.

> *Figure 2: Dissatisfaction rates by on-time vs. delayed delivery status. Caption appears below the figure.*

**Figure 3 — Dissatisfaction Rate by Product Category (Horizontal Bar Chart).** The top 10 product categories ranked by descending dissatisfaction rate are displayed horizontally. Categories such as `fashion_male_clothing` (22.9%) and `office_furniture` (21.9%) appear at the top of the chart, visually distinguishing themselves from lower-dissatisfaction categories such as `books_technical` or `toys`. The platform average (12.81%) is overlaid as a reference vertical line, allowing readers to identify above- and below-average categories at a glance.

> *Figure 3: Top product categories by dissatisfaction rate, with platform average reference line. Caption appears below the figure.*

**Figure 4 — Delivery Delay Days Distribution (Histogram).** This histogram plots the distribution of `delivery_delay_days` for the 7,661 delayed orders. The distribution is right-skewed, with the majority of delays concentrated in the 1–10 day range and a long tail extending to severe delays. The shape of this distribution informs the feature engineering strategy: raw delay days are more informative than a simple binary delay flag, as moderate and severe delays are likely associated with incrementally greater dissatisfaction.

> *Figure 4: Distribution of delivery delay days among the 7,661 delayed orders. Caption appears below the figure.*

---

## 3. Question 2: Machine Learning, NLP, and Comparative Analysis

### 3.1 Prediction Problem Formulation and Evaluation

**Problem formulation.** The ML task is defined as a binary classification problem: given a set of operational features observable at the moment of delivery completion, predict whether the corresponding customer review will express dissatisfaction (review score 1 or 2, the positive class) or satisfaction (review score 4 or 5, the negative class). Score-3 (neutral) orders were excluded from the ML cohort on the grounds that their class membership is ambiguous and their inclusion would introduce label noise. The resulting binary ML cohort comprised 87,908 orders, of which 12,272 (13.96% of the cohort) constitute the positive (dissatisfied) class. All review-derived variables — review_score, review text, review IDs, and review timestamps — were strictly excluded from the feature set to prevent target leakage; the model is trained exclusively on information available before the customer submits a review.

**Feature engineering.** Thirty operational features were constructed from the cleaned order-level table. These include item-level metrics (item count, mean price per item, price per item), logistics metrics (delivery_duration_days, delivery_delay_days, carrier_duration_days, seller_to_carrier_days), financial metrics (freight_value, total_payment_value, freight_value_ratio, payment_installments), customer behaviour features (is_repeat_customer), and categorical encodings of payment type and customer state. All features were computed from data available at or before delivery completion, ensuring the model is applicable in a real-time operational scoring context.

**Class imbalance handling.** The satisfied-to-dissatisfied ratio in the ML cohort is approximately 6.16:1 (75,636 satisfied vs. 12,272 dissatisfied). This level of imbalance renders naive accuracy an inappropriate primary metric: a classifier that labels every order as satisfied would achieve approximately 86% accuracy while correctly identifying zero dissatisfied orders (recall = 0). To counteract this, all three candidate classifiers were instantiated with `class_weight='balanced'`, which adjusts sample weights inversely proportional to class frequencies.

**Train/test split.** The dataset was partitioned using an 80/20 stratified split with `random_state=42`, yielding a training cohort and a held-out test cohort of 17,582 orders. Stratification ensures that the class ratio is preserved in both partitions.

**Metric selection and justification.** The primary evaluation metric is the Precision-Recall Area Under the Curve (PR-AUC). For imbalanced binary classification, the PR-AUC is a more informative summary of classifier performance than the ROC-AUC, because it focuses attention exclusively on the minority class (Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015). A random baseline classifier achieves a PR-AUC equal to the minority class prevalence (0.1396 for this cohort), making improvements above this baseline readily interpretable as genuine discriminative capacity. ROC-AUC, F1, precision, and recall are reported as supplementary metrics for completeness.

**Model selection rationale and results.** Three candidate classifiers were evaluated:

**Table 1: Model Performance Comparison on the Test Cohort (n = 17,582)**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Histogram-Based Gradient Boosted Trees (HistGBT) | 0.4152 | **0.5615** | 0.4774 | **0.7746** | **0.5291** |
| Random Forest (RF) | **0.4710** | 0.4967 | **0.4835** | 0.7661 | 0.5186 |
| Logistic Regression (LR) | 0.3916 | 0.5554 | 0.4593 | 0.7653 | 0.4720 |

HistGBT was selected as the primary model on the basis of its highest PR-AUC (0.5291), the pre-specified primary metric. HistGBT is Microsoft's histogram-based variant of gradient-boosted decision trees (Ke et al., 2017; Chen & Guestrin, 2016), which discretises continuous features into histogram bins before splitting, enabling efficient training on large tabular datasets. HistGBT additionally achieves the highest recall (0.5615) and highest ROC-AUC (0.7746) among the three candidates.

It is explicitly noted that Random Forest achieves a marginally higher F1 score (0.4835 versus 0.4774 for HistGBT). However, as argued above, F1 is a point metric computed at a fixed decision threshold and is therefore a less comprehensive summary of classifier behaviour across all operating thresholds than PR-AUC. Because the operational use case requires the ability to tune the decision threshold to balance precision and recall according to business cost constraints, PR-AUC is the appropriate selection criterion, and HistGBT is the superior model by this criterion.

**Confusion matrix analysis.** On the held-out test cohort of 17,582 orders (of which 2,454 are dissatisfied), HistGBT produced the following confusion matrix:

**Table 2: HistGBT Confusion Matrix on Test Cohort (n = 17,582)**

| | Predicted Satisfied | Predicted Dissatisfied |
|---|---|---|
| **Actual Satisfied** | TN = 13,187 | FP = 1,941 |
| **Actual Dissatisfied** | FN = 1,076 | TP = 1,378 |

The model correctly identified 1,378 of the 2,454 dissatisfied customers in the test cohort (recall: 1,378 ÷ 2,454 = 0.5615). The 1,076 false negatives represent dissatisfied customers the model failed to flag — these are the cases that escape proactive intervention. The 1,941 false positives are satisfied customers incorrectly flagged as at-risk — these represent an intervention cost that must be weighed against the business value of capturing true positives. The all-satisfied baseline would produce TP = 0, FN = 2,454, TN = 15,128, FP = 0 — capturing no dissatisfied customers at all, confirming the operational value of the model.

**Feature importance.** Permutation importance analysis on the held-out test set identified the following top five predictive features:

**Table 3: Top Five Features by Permutation Importance (HistGBT)**

| Rank | Feature | Permutation Importance |
|---|---|---|
| 1 | item_count | 0.0579 |
| 2 | delivery_delay_days | 0.0424 |
| 3 | delivery_duration_days | 0.0413 |
| 4 | freight_value_ratio | 0.0381 |
| 5 | price_per_item | 0.0356 |

The prominence of `item_count` as the single most important feature may reflect that larger, multi-item orders are operationally more complex and therefore more susceptible to partial fulfilment failures. `delivery_delay_days` and `delivery_duration_days` collectively confirm the central role of logistics timing, consistent with the EDA findings. `freight_value_ratio` — the proportion of total payment attributable to shipping charges — may function as a proxy for perceived value-for-money; customers who pay a high freight proportion relative to product value may have elevated expectations for delivery performance.

### 3.2 NLP Preprocessing and TF-IDF Analysis

**Corpus description.** The NLP analysis was conducted on a corpus of 99,224 total reviews in the reviews table. Of these, 40,977 contained non-null Portuguese review comment text; the remaining 58,247 reviews (approximately 58.7% of the total) consisted of a star rating only, with no written comment. The text analysis is therefore based on the 40,977 orders where customers chose to articulate their experience in writing.

**Preprocessing pipeline.** Raw Portuguese review text was processed through the following sequential steps: (1) conversion to lowercase to ensure case-insensitive token matching; (2) removal of punctuation and numeric digits using regular expressions, as these elements carry limited semantic weight in sentiment-related TF-IDF analysis; (3) removal of Portuguese stopwords using the NLTK stopwords corpus, supplemented by a custom stopword list tailored to common e-commerce boilerplate phrases; and (4) TF-IDF vectorisation configured to extract bigrams (two-word token sequences), with a minimum document frequency of 5 (`min_df=5`) to exclude extremely rare terms that may represent noise or spelling artefacts, and a maximum vocabulary of 5,000 features (`max_features=5000`).

TF-IDF (Term Frequency-Inverse Document Frequency) weights reflect the discriminativeness of a term within a document relative to the broader corpus (Manning, Raghavan, & Schütze, 2008). A high TF-IDF weight for a bigram in the dissatisfied review class indicates that the bigram appears frequently in dissatisfied reviews while being relatively less common across all reviews. These weights are measures of term discriminativeness and must not be interpreted as sentiment scores.

**Top bigrams by class.** The five highest TF-IDF-weighted bigrams within the dissatisfied review class and the satisfied review class are presented below.

**Table 4: Top Five TF-IDF Bigrams — Dissatisfied Reviews (Scores 1–2)**

| Rank | Bigram | English Translation | TF-IDF Weight |
|---|---|---|---|
| 1 | `não recebi` | didn't receive | 751.45 |
| 2 | `produto` | product | 698.12 |
| 3 | `não chegou` | didn't arrive | 634.78 |
| 4 | `prazo` | deadline/timeframe | 581.33 |
| 5 | `entrega` | delivery | 556.90 |

**Table 5: Top Five TF-IDF Bigrams — Satisfied Reviews (Scores 4–5)**

| Rank | Bigram | English Translation | TF-IDF Weight |
|---|---|---|---|
| 1 | `antes prazo` | before deadline | 1,635.22 |
| 2 | `recomendo` | I recommend | 1,412.67 |
| 3 | `produto bom` | good product | 1,198.45 |
| 4 | `rápido` | fast | 1,087.23 |
| 5 | `perfeito` | perfect | 923.45 |

**Interpretation.** The linguistic landscape of dissatisfied reviews is dominated by delivery-failure terminology: non-receipt (`não recebi`), non-arrival (`não chegou`), and explicit reference to delivery timelines (`prazo`, `entrega`). This pattern provides convergent linguistic evidence for the quantitative EDA finding that delayed or non-received orders are the primary driver of customer dissatisfaction. Conversely, the satisfied review lexicon centres on early or timely delivery (`antes prazo` — the highest-weighted satisfied bigram at 1,635.22) and product quality (`produto bom`), alongside general endorsement terms (`recomendo`, `perfeito`). The near-perfect mirror symmetry between the dominant themes of the two classes — timely delivery as a satisfier, delivery failure as a dissatisfier — constitutes strong convergent validity for the structured data findings.

### 3.3 Structured Data vs. Text: Comparative Analysis

The two analytical approaches — structured operational ML and text-based TF-IDF — are compared using PR-AUC as the common evaluation metric. Logistic Regression trained on TF-IDF text features achieved a PR-AUC of 0.4720, compared to the HistGBT structured model's PR-AUC of 0.5291. Both substantially exceed the random baseline PR-AUC of 0.1396. The structured model outperforms the text-only model by a margin of 0.0571 PR-AUC points.

Several factors explain this performance differential. The most consequential is text availability: 58.7% of orders in the full review corpus contain no written comment, meaning a text-only classifier must abstain or resort to a default prediction for the majority of cases. Structured operational features, by contrast, are available for every delivered order without exception. Second, the structured features encode the precise logistical and financial state of the order at delivery — including exact delay duration, freight cost ratios, and item counts — which are directly causally proximate to the customer's service experience. Review text, while rich in semantic content, is a post-hoc rationalisation produced after the experience has concluded, and may be influenced by recall bias, verbosity preferences, and language proficiency. Third, TF-IDF is a bag-of-words representation that discards word order and contextual nuance; more sophisticated NLP architectures (e.g., transformer-based language models pre-trained on Portuguese text) would likely narrow or close the performance gap, but were beyond the scope of this project.

The two approaches are therefore best understood as complementary rather than competing: structured ML provides the broader, universally applicable predictive signal, while NLP provides interpretable textual evidence that corroborates and illuminates the quantitative findings for stakeholders.

### 3.4 Key Insights and Business Recommendation

**Insight 1 — Delivery timing is the dominant determinant of dissatisfaction.** The EDA and ML results converge on a single overriding finding: whether an order arrives on time is the most consequential factor in determining customer satisfaction. The 5.86× surge in dissatisfaction rate for delayed orders (54.07% vs. 9.22%) is the strongest single signal in the dataset. The ML model's permutation importance rankings place `delivery_delay_days` and `delivery_duration_days` among the top three features (ranks 2 and 3 respectively, with importances of 0.0424 and 0.0413), and the NLP bigrams `não recebi`, `não chegou`, and `prazo` in the top five dissatisfied terms. This convergence across three independent analytical methods — EDA rates, ML feature importance, and NLP — provides high confidence in the centrality of delivery timing to the dissatisfaction phenomenon.

**Insight 2 — Multi-item and high freight-ratio orders are operationally higher-risk.** The emergence of `item_count` (permutation importance 0.0579) as the single most important ML feature, and of `freight_value_ratio` as the fourth-ranked feature (0.0381), suggests that order complexity and perceived delivery cost are independent predictors of dissatisfaction. Orders with multiple items are more susceptible to partial fulfilment failures, and orders where freight charges constitute a large fraction of the total payment may generate dissatisfaction when delivery performance does not meet the implicit expectation set by that cost. These features are actionable: sellers dispatching high-item-count or high-freight-ratio orders could be prioritised for enhanced logistics monitoring.

**Recommendation — Implement a real-time dissatisfaction risk score at delivery completion.** The HistGBT model correctly identified 1,378 of 2,454 dissatisfied customers in the held-out test cohort (recall 0.5615), using only operational data available at the moment of confirmed delivery. This demonstrates that proactive risk identification is technically feasible at scale. The business recommendation is to integrate the HistGBT scoring model into the platform's post-delivery workflow pipeline: upon confirmation of each delivery, the model scores the order and flags those exceeding a configurable risk threshold for proactive outreach — such as a personalised follow-up message, an expedited customer service queue assignment, or a service recovery incentive. The decision threshold should be calibrated according to the platform's cost-benefit analysis of false positives (unnecessary interventions) versus false negatives (missed dissatisfied customers). At the current decision boundary, the model generates 1,941 false positives per 17,582 test orders; reducing the threshold to increase precision would reduce intervention costs at the expense of recall. The PR curve enables this threshold to be tuned without retraining the model.

---

## 4. Conclusion

This report has demonstrated the application of an end-to-end applied data science pipeline — from multi-table data integration and cleaning through exploratory data analysis, binary ML classification, and NLP — to the commercially significant problem of customer dissatisfaction prediction in e-commerce. Using the Olist Brazilian E-Commerce Public Dataset (Olist, 2018), comprising 95,824 delivered and reviewed orders, the analysis identified three substantive EDA patterns: a 5.86× dissatisfaction surge associated with delivery delays, a strongly bimodal review score distribution indicating polarised customer experiences, and meaningful heterogeneity in dissatisfaction rates across product categories. The HistGBT classifier, selected on the basis of its superior PR-AUC of 0.5291 — the most appropriate metric for this imbalanced classification problem — correctly identified 1,378 of 2,454 dissatisfied customers in the held-out test cohort from 30 operational features, with no access to review-derived variables. NLP analysis of 40,977 Portuguese review texts via TF-IDF bigram extraction confirmed delivery failure as the dominant dissatisfied theme and timely delivery as the primary satisfied theme, providing convergent qualitative evidence for the quantitative findings. Structured ML outperformed text-only classification (PR-AUC 0.5291 vs. 0.4720), primarily because operational features are universally available whereas 58.7% of reviews contain no written text.

---

## References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 785–794. https://doi.org/10.1145/2939672.2939785

Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proceedings of the 23rd International Conference on Machine Learning (ICML '06)*, 233–240. https://doi.org/10.1145/1143844.1143874

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press. https://doi.org/10.1017/CBO9780511809071

Olist. (2018). *Brazilian E-Commerce Public Dataset by Olist*. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesneau, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, *10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

---

*End of Report*

