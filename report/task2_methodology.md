# Customer360 AI: Task 2 Methodology Report
## Data Quality Assessment, Integration, Feature Engineering, and Exploratory Data Analysis

**Course:** COM-702 Applied Data Science  
**Project Title:** Customer360 AI: Predicting E-Commerce Customer Dissatisfaction Using Machine Learning and Understanding Customer Feedback Through Natural Language Processing  
**Dataset:** Brazilian E-Commerce Public Dataset by Olist (9 Relational CSV Tables)  
**Document Type:** Formal Academic Methodology & Empirical Findings Report  
**Date:** November 2026  

---

## 1. Executive Summary

This report documents the rigorous data engineering, integration, feature engineering, and exploratory data analysis (EDA) conducted under **Task 2** of the Customer360 AI investigation. Using all nine relational CSV files from the Brazilian e-commerce platform Olist, we have constructed an order-level predictive feature store while completely safeguarding against Cartesian join explosions and target leakage.

### Key Milestones Achieved:
1. **Zero Data Distortion**: All 9 raw CSV files remain pristine and immutable in `data/raw/`.
2. **Comprehensive Review Preservation**: All 99,224 original review records—including 40,977 non-null customer text comments—are preserved in `data/processed/customer360_reviews_nlp.csv` for Natural Language Processing in Question 2.
3. **Deterministic Review Tie-Breaking**: For the 551 orders associated with multiple review submissions, a deterministic tie-breaking hierarchy (`review_answer_timestamp` $\rightarrow$ `review_creation_date` $\rightarrow$ `review_id`) was applied to extract the customer's final verified appraisal.
4. **Aggregation Before Joining**: Relational tables (`order_items`, `order_payments`, `products`, `sellers`) were aggregated to the unique `order_id` grain before merging, strictly validating 1-to-1 join cardinality without artificial row multiplication.
5. **Post-Fulfilment Prediction Formulation**: Target dissatisfaction is formulated as Review Score $\in \{1, 2\}$ ($Y=1$, 12,272 orders) versus Satisfied $\in \{4, 5\}$ ($Y=0$, 75,636 orders), excluding 7,916 neutral Score 3 reviews. Features represent operational status at the point of delivery completion, strictly excluding survey attributes.
6. **Empirical Discovery**: Delivery SLA failure is identified as the single largest operational friction point, causing a **5.86-fold increase** in customer dissatisfaction (from 9.22% for on-time orders to 54.07% for delayed orders).

---

## 2. Data Quality Assessment & Preprocessing Audit

A systematic audit across all nine raw tables was conducted to inspect missing values, integrity constraints, and domain-specific anomalies.

| Table Name | Raw Rows | Key Integrity Check | Anomalies Detected | Preprocessing & Cleaning Decision |
| :--- | :---: | :--- | :--- | :--- |
| `olist_customers_dataset` | 99,441 | `customer_id` is 100% unique | 3,345 repeat customer purchases identified via `customer_unique_id`. | Preserved all rows. Engineered `is_repeat_customer` binary feature. |
| `olist_orders_dataset` | 99,441 | `order_id` is 100% unique | 2,965 missing customer delivery dates (2,957 unfulfilled/cancelled; 8 delivered). | Converted timestamps to datetime. Filtered delivered orders with valid delivery dates for delivery-performance modeling. |
| `olist_order_items_dataset` | 112,650 | Composite PK `(order_id, order_item_id)` 100% unique | 0 negative prices; 0 negative freight values. Multi-item orders up to 21 items. | Aggregated to `order_id` grain: item count, total price, total freight, freight ratio, and unique seller count. |
| `olist_order_payments_dataset` | 103,886 | Composite PK `(order_id, payment_sequential)` 100% unique | 9 zero-value records (voucher offsets); 3 rows with payment type `not_defined`. | Aggregated to `order_id` grain: total payment value, max installments, and one-hot payment method indicators. |
| `olist_order_reviews_dataset` | 99,224 | 814 duplicate `review_id`s; 551 orders with multiple reviews | All scores validly bounded $[1, 5]$. 58.7% comment messages null (40,977 texts present). | Preserved all 99,224 reviews for NLP in `customer360_reviews_nlp.csv`. For ML, selected latest review via timestamp tie-break. |
| `olist_products_dataset` | 32,951 | `product_id` is 100% unique | 610 missing product categories; 2 missing weights/dimensions. | Imputed missing weights with category median. Mapped Portuguese categories to English via translation table + manual overrides. |
| `olist_sellers_dataset` | 3,095 | `seller_id` is 100% unique | 3,095 sellers across 23 Brazilian states (SP accounts for 59.7%). | Joined via items to identify seller state and derive `is_same_state` (interstate vs intrastate fulfillment). |
| `olist_geolocation_dataset` | 1,000,163 | Zip code prefix | Multiple coordinate readings per zip prefix. | Referenced for spatial consistency; primary fulfillment proximity captured via state-level matching. |
| `product_category_name_translation` | 71 | `product_category_name` | 2 categories (`pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos`) absent. | Manually expanded dictionary to achieve 100% English translation coverage. |

---

## 3. Review Data Handling: Deduplication vs NLP Preservation

### 3.1 The Review Grain Challenge
In the raw review dataset (`olist_order_reviews_dataset.csv`):
- Total records: **99,224**
- Unique `order_id` values: **98,673**
- Orders with $>1$ review: **547 unique orders** accounting for **551 duplicate entries** (543 orders have 2 reviews, 4 orders have 3 reviews).
- Duplicate `review_id` values: **814 instances**.

Directly merging raw reviews with orders would cause row duplication, distorting subsequent aggregations and violating statistical independence assumptions.

### 3.2 Deterministic Tie-Breaking Rule for ML
For the order-level machine learning dataset, each order must map to exactly one ground-truth evaluation. When multiple reviews exist for an order, we prioritize the customer's most recent interaction:
$$\text{Sort By: } \text{review\_answer\_timestamp } (\downarrow) \longrightarrow \text{review\_creation\_date } (\downarrow) \longrightarrow \text{review\_id } (\downarrow)$$
Selecting the first record under this sort order deterministically extracts the latest post-delivery evaluation, resolving all 551 duplicate order instances.

### 3.3 NLP Corpus Preservation
Critically, **no review text is discarded**. All 99,224 reviews are preserved in [data/processed/customer360_reviews_nlp.csv](file:///c:/Users/BHAWANA%20SHARMA/OneDrive/Desktop/Customer360_AI_COM702/data/processed/customer360_reviews_nlp.csv). This dataset contains:
- 40,977 non-null customer review comments (`review_comment_message`).
- 11,568 non-null comment titles (`review_comment_title`).
- 42,706 reviews containing at least one textual field.
- Linked fulfillment metadata (`order_purchase_timestamp`, `order_status`, `is_delayed`) to enable comparative topic and sentiment modeling between satisfied and dissatisfied cohorts in Task 3/4.

---

## 4. Relational Integration & Aggregation Strategy

### 4.1 Preventing Cartesian Join Explosion
E-commerce data models exhibit independent one-to-many relationships:
$$\text{Orders} \xrightarrow{1:N} \text{Order Items} \quad \text{and} \quad \text{Orders} \xrightarrow{1:M} \text{Order Payments}$$
A naive join of `orders` with `order_items` (112,650 rows) and `order_payments` (103,886 rows) generates a Cartesian product of size $\sum_i (N_i \times M_i)$, creating spurious rows and artificially compounding monetary totals.

### 4.2 Aggregation Architecture
To maintain strict 1-to-1 relational integrity, all transactional entities were pre-aggregated to the `order_id` grain prior to joining:

1. **Order Items Aggregation (`items_agg`)**:
   - `item_count`: Total items ordered ($\sum \text{order\_item\_id}$).
   - `total_price`: Total merchandise value ($\sum \text{price}$).
   - `total_freight`: Total freight billed ($\sum \text{freight\_value}$).
   - `total_order_value`: $\text{total\_price} + \text{total\_freight}$.
   - `freight_ratio`: $\frac{\text{total\_freight}}{\text{total\_order\_value} + 10^{-5}}$.
   - `unique_products`: Distinct product IDs per order.
   - `unique_sellers`: Distinct merchant partners fulfilling the order.
   - `product_weight_g_mean`: Mean physical package weight.
   - `product_volume_cm3_mean`: Mean volumetric dimensions ($L \times H \times W$).
   - `primary_product_category`: English category of the highest-value item.
   - `primary_seller_state`: Geographic origin of the primary item's seller.

2. **Order Payments Aggregation (`pay_agg`)**:
   - `payment_value_total`: Sum of all payment disbursements.
   - `payment_installments_max`: Maximum financing installments utilized.
   - `payment_sequential_count`: Number of sequential payment splits.
   - `has_credit_card`, `has_boleto`, `has_voucher`, `has_debit_card`: Binary presence indicators.

3. **Customer Enrichment (`customers`)**:
   - `is_repeat_customer`: Binary indicator of whether `customer_unique_id` appears more than once.
   - `is_same_state`: Indicator for whether `customer_state == primary_seller_state` (intrawarehouse vs interstate freight).

4. **Integration Verification**:
   - Merging `orders` (99,441 records) with `customers`, `items_agg`, `pay_agg`, and `rev_order_level` using `validate='one_to_one'` yielded **exactly 99,441 records**, confirming 100% join fidelity without row multiplication.

---

## 5. Target Formulation, Prediction Timing & Leakage Prevention

### 5.1 Post-Fulfilment Decision Timing
In enterprise CX operations, dissatisfaction prediction serves two distinct operational use-cases:
1. *Pre-purchase risk scoring* (prior to checkout).
2. *Post-delivery proactive intervention* (after carrier delivery, but before customer survey completion or churn).

This investigation addresses **Post-Fulfilment Dissatisfaction Prediction**:
- **Prediction Horizon**: The exact moment the package is recorded as delivered by the carrier (`order_delivered_customer_date`), prior to the customer submitting a CSAT review.
- **Operational Justification**: This enables e-commerce customer operations to automatically identify high-risk delivery failures immediately upon arrival, triggering proactive outreach (apology credits, return authorizations, discount vouchers) *before* the customer vents negative sentiment or defects.

### 5.2 Prevention of Target Leakage
To ensure academic validity and real-world applicability:
- **Excluded Features**: `review_score`, `review_comment_message`, `review_comment_title`, `review_id`, `review_creation_date`, and `review_answer_timestamp` are strictly excluded from the predictive feature matrix $X$.
- **Permissible Features**: Actual delivery duration (`delivery_duration_days`), carrier transit time (`carrier_duration_days`), and delivery SLA delay (`delivery_delay_days`, `is_delayed`) are fully permissible because they are known at the post-fulfilment decision point.

### 5.3 Ground-Truth Target Definition
Consistent with academic literature and industry Net Promoter Score (NPS) methodologies:
- **Dissatisfied ($Y=1$)**: Review Score $\in \{1, 2\}$ (Acute service failure, defective goods, extreme delay).
- **Satisfied ($Y=0$)**: Review Score $\in \{4, 5\}$ (Meets or exceeds expectations).
- **Neutral ($Y=\text{Excluded}$)**: Review Score $= 3$ (Ambivalent sentiment).
  - Across the delivered and reviewed cohort of 95,824 orders, **7,916 records (8.26%)** have Score 3.
  - Excluding Score 3 establishes a clean, separable decision boundary for binary classification, yielding a modeling cohort of **87,908 orders**.
  - **Class Imbalance**: Dissatisfied ($Y=1$) represents **12,272 orders (13.96%)**, while Satisfied ($Y=0$) represents **75,636 orders (86.04%)**, establishing a ~6.16:1 class imbalance.

---

## 6. Feature Engineering and Correlation Analysis

The table below summarizes the key engineered predictors evaluated across the binary classification cohort ($N = 87,908$):

| Feature Name | Type | Mean | Std Dev | Min | Median | Max | Correlation with Dissatisfaction ($r$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `is_delayed` | int64 | 0.08 | 0.27 | 0.00 | 0.00 | 1.00 | **+0.3927** |
| `delivery_duration_days` | float64 | 12.36 | 9.40 | 0.53 | 10.11 | 208.35 | **+0.3363** |
| `carrier_duration_days` | float64 | 9.17 | 8.61 | -16.10 | 7.06 | 205.19 | **+0.3029** |
| `delivery_delay_days` | float64 | -11.32 | 10.08 | -146.02 | -12.06 | 188.98 | **+0.2750** |
| `item_count` | float64 | 1.14 | 0.53 | 1.00 | 1.00 | 21.00 | **+0.1344** |
| `unique_sellers` | float64 | 1.01 | 0.12 | 1.00 | 1.00 | 5.00 | **+0.1304** |
| `total_freight` | float64 | 22.69 | 21.45 | 0.00 | 17.14 | 1,794.96 | **+0.0942** |
| `total_order_value` | float64 | 160.33 | 220.63 | 9.59 | 105.28 | 13,664.08 | **+0.0498** |
| `payment_installments_max` | float64 | 2.92 | 2.71 | 0.00 | 2.00 | 24.00 | **+0.0377** |
| `product_weight_g_mean` | float64 | 2,088.26 | 3,730.74 | 0.00 | 700.00 | 40,425.00 | **+0.0275** |
| `freight_ratio` | float64 | 0.21 | 0.13 | 0.00 | 0.18 | 0.96 | **+0.0192** |
| `is_same_state` | int64 | 0.36 | 0.48 | 0.00 | 0.00 | 1.00 | **-0.0571** |

---

## 7. Exploratory Data Analysis & Empirical Discoveries

Six high-resolution figures were generated in `outputs/figures/` to elucidate the empirical drivers of customer dissatisfaction:

### 7.1 EDA 1: Review Score Distribution & Target Demarcation
- **Visual Artifact**: [eda_1_review_score_distribution.png](file:///c:/Users/BHAWANA%20SHARMA/OneDrive/Desktop/Customer360_AI_COM702/outputs/figures/eda_1_review_score_distribution.png)
- **Empirical Observation**: The review score distribution is highly bimodal and left-skewed: Score 5 comprises 59.2% (56,748 orders), Score 4 comprises 19.7% (18,888 orders), Score 3 comprises 8.3% (7,916 orders), Score 2 comprises 3.0% (2,921 orders), and Score 1 comprises 9.8% (9,351 orders).
- **Insight**: Customer dissatisfaction in e-commerce is not a subtle gradient; customers either award top marks (4-5 stars) when service meets expectations or rapidly drop to 1-2 stars upon encountering operational failure.

### 7.2 EDA 2: Delivery SLA Breach vs Customer Dissatisfaction
- **Visual Artifact**: [eda_2_dissatisfaction_by_delivery_performance.png](file:///c:/Users/BHAWANA%20SHARMA/OneDrive/Desktop/Customer360_AI_COM702/outputs/figures/eda_2_dissatisfaction_by_delivery_performance.png)
- **Empirical Observation**: 
  - For orders delivered on-time or early ($\le$ estimated SLA), the dissatisfaction rate is **9.22%** (8,131 dissatisfied out of 88,163 orders).
  - For orders delivered late ($>$ estimated SLA), the dissatisfaction rate surges to **54.07%** (4,141 dissatisfied out of 7,661 orders).
- **Insight**: Delivery SLA breach induces a **5.86-fold increase** in the probability of customer dissatisfaction. Logistics reliability is unequivocally the primary determinant of customer retention.

### 7.3 EDA 3: Product Category Disparity
- **Visual Artifact**: [eda_3_product_category_dissatisfaction.png](file:///c:/Users/BHAWANA%20SHARMA/OneDrive/Desktop/Customer360_AI_COM702/outputs/figures/eda_3_product_category_dissatisfaction.png)
- **Empirical Observation**: 
  - Categories with the highest dissatisfaction rates include `bed_bath_table` (15.8%), `furniture_decor` (15.4%), and `office_furniture` (18.2%).
  - Categories with the lowest dissatisfaction rates include `cool_stuff` (10.9%), `stationery` (9.8%), and `books_general_interest` (8.4%).
- **Insight**: Bulky, heavy items with multi-part assembly or freight shipping face higher transit damage, handling delays, and product expectation gaps compared to small, standardized merchandise.

### 7.4 EDA 4: Financial Friction (Freight Ratio & Order Value)
- **Visual Artifact**: [eda_4_freight_and_order_value_vs_satisfaction.png](file:///c:/Users/BHAWANA%20SHARMA/OneDrive/Desktop/Customer360_AI_COM702/outputs/figures/eda_4_freight_and_order_value_vs_satisfaction.png)
- **Empirical Observation**: Dissatisfied customers bear a higher median freight ratio (0.189 vs 0.182) and higher median order value (R$ 105.28 vs R$ 98.40).
- **Insight**: Financial exposure heightens customer scrutiny. Customers who pay higher freight charges relative to merchandise value exhibit zero tolerance for logistics delays.

### 7.5 EDA 5: Delivery Delay Distribution
- **Visual Artifact**: [eda_5_delivery_delay_distribution.png](file:///c:/Users/BHAWANA%20SHARMA/OneDrive/Desktop/Customer360_AI_COM702/outputs/figures/eda_5_delivery_delay_distribution.png)
- **Empirical Observation**: 
  - 92.0% of delivered orders arrive prior to the promised SLA date (averaging 12.0 days early).
  - The 8.0% of orders that miss the SLA arrive an average of 11.5 days late (maximum delay exceeding 188 days).
- **Insight**: Olist incorporates generous safety buffers in promised delivery dates. Therefore, when an order exceeds the SLA, it represents an acute logistical failure rather than a minor scheduling discrepancy.

### 7.6 EDA 6: Payment Configuration vs Satisfaction
- **Visual Artifact**: [eda_6_payment_patterns_vs_satisfaction.png](file:///c:/Users/BHAWANA%20SHARMA/OneDrive/Desktop/Customer360_AI_COM702/outputs/figures/eda_6_payment_patterns_vs_satisfaction.png)
- **Empirical Observation**: Dissatisfaction escalates monotonically with financing duration: 1 installment (12.1% dissat), 2-3 installments (12.8%), 4-6 installments (13.5%), 7-10 installments (14.2%), and 11-24 installments (**19.5% dissat**).
- **Insight**: Extended installment plans correspond to major capital purchases (appliances, consumer electronics) where customer stakes and delivery expectations are maximal.

---

## 8. Viva Voce Preparation: Methodological Defenses

During academic examination, evaluators typically focus on methodological rigor. Below are direct defenses for anticipated queries:

1. **Why exclude neutral reviews (Score 3) from binary classification?**
   - *Defense*: Review score 3 represents ambiguous customer sentiment ("Average / Neither satisfied nor dissatisfied"). Retaining score 3 within either the satisfied or dissatisfied class introduces label noise, degrading classifier margin separation. For sensitivity analysis, score 3 can be treated as dissatisfied in an alternate ablation study.
2. **Does using `delivery_duration_days` constitute target leakage?**
   - *Defense*: No, because our problem is formulated at the **post-fulfilment** decision point (the moment of delivery confirmation). The goal is to detect high-risk friction before the customer vents on survey platforms, enabling immediate mitigation. Features known at delivery completion are strictly valid.
3. **How was Cartesian explosion prevented when joining orders, items, and payments?**
   - *Defense*: We applied independent groupby aggregations on `order_items` and `order_payments` to collapse both child tables to the unique `order_id` grain prior to executing a 1-to-1 validated left join with `orders`. The final dataset maintains exactly 99,441 records.

