# Customer360 AI: Power BI Data Dictionary
**Module:** COM-702 (C) Applied Data Science — MIET Jammu  
**Project:** Customer360 AI Dissatisfaction Prediction & Feedback Analysis  
**Author:** Bhawana Sharma (Roll No: 2023A1R076)

---

## 1. Overview of Data Architecture

The Power BI delivery kit organizes customer experience, logistics, machine learning benchmarks, and textual feedback into an optimized star schema with one central fact table (`Fact_Orders`) and six supporting dimension tables.

```text
outputs/powerbi/data/
├── Fact_Orders.csv                  # Central Fact Table (95,824 delivered & reviewed orders)
├── Dim_Categories.csv               # Product Category Dimension (74 categories)
├── Dim_Customer_Geo.csv             # Customer State & Geographic Region (27 states)
├── Dim_Model_Benchmark.csv          # Algorithm Comparison Table (3 models)
├── Dim_Confusion_Matrix.csv         # Champion Model Confusion Matrix (4 cells, 17,582 cases)
├── Dim_Feature_Importance.csv       # Permutation Importance Rankings (30 features)
├── Dim_NLP_Terms.csv                # Top Portuguese TF-IDF Terms & Themes (30 rows)
└── Dim_Structured_vs_Text.csv       # Multimodal Triangulation Matrix (4 friction themes)
```

---

## 2. Table Specifications & Schema Definitions

### Table 1: `Fact_Orders.csv`
* **Grain:** One row per delivered and reviewed customer order.
* **Row Count:** 95,824 rows.
* **Primary Key:** `order_id` (100% unique, zero duplicates).

| Column Name | Data Type | Sample Value | Description & Business Rules | Join Foreign Key |
|---|---|---|---|---|
| `order_id` | Text | `e481f51cbdc54...` | Unique identifier of the e-commerce order (Primary Key). | — |
| `customer_id` | Text | `9ef44fb54ee1f...` | Order-specific customer token. | — |
| `customer_unique_id` | Text | `7c396fd4830fd...` | Unique identifier of the individual customer. | — |
| `customer_state` | Text | `SP` | Two-letter Brazilian state code of delivery location. | `Dim_Customer_Geo[customer_state]` |
| `customer_city` | Text | `sao paulo` | Delivery city name. | — |
| `primary_product_category` | Text | `housewares` | Primary product category for multi-item orders. | `Dim_Categories[primary_product_category]` |
| `order_purchase_timestamp` | DateTime | `2017-10-02 10:56:33` | Timestamp when order was placed. | — |
| `order_delivered_customer_date` | DateTime | `2017-10-10 21:25:13` | Courier delivery scan timestamp at customer doorstep. | — |
| `order_estimated_delivery_date` | DateTime | `2017-10-18 00:00:00` | Promised delivery deadline given at checkout (SLA). | — |
| `purchase_year` | Integer | `2017` | Year of purchase (2016–2018). | — |
| `purchase_month` | Integer | `10` | Month of purchase (1–12). | — |
| `purchase_dayofweek` | Integer | `0` | Day of week (0 = Monday, 6 = Sunday). | — |
| `delivery_duration_days` | Decimal | `8.44` | Actual duration from purchase to physical delivery. | — |
| `estimated_delivery_duration_days`| Decimal | `15.54`| Expected duration from purchase to estimated SLA date. | — |
| `delivery_delay_days` | Decimal | `-7.10` | Difference: Actual delivery date minus Estimated SLA date ($>0$ = late, $\le 0$ = on-time). | — |
| `is_delayed` | Integer | `0` | Binary indicator: `1` if late (`delivery_delay_days > 0`), else `0`. | — |
| `delivery_status` | Text | `On-Time (<= SLA)` | Categorical label: `On-Time (<= SLA)` vs `Delayed (> SLA)`. | — |
| `item_count` | Integer | `1` | Total quantity of merchandise items in the order. | — |
| `total_price` | Decimal | `29.99` | Cumulative item price in Brazilian Reais (BRL). | — |
| `total_freight` | Decimal | `8.72` | Cumulative shipping charge in BRL. | — |
| `total_order_value` | Decimal | `38.71` | Total transaction value ($Price + Freight$). | — |
| `freight_ratio` | Decimal | `0.2253` | Ratio of freight charges to total order value. | — |
| `payment_value_total` | Decimal | `38.71` | Total settled payment amount. | — |
| `payment_installments_max` | Integer | `1` | Maximum financing installments chosen (1 to 24). | — |
| `installment_tier` | Text | `1 Installment (Single)`| Categorical grouping of financing installments. | — |
| `has_credit_card` | Integer | `1` | Flag: order paid using credit card. | — |
| `has_boleto` | Integer | `0` | Flag: order paid using Brazilian bank slip (Boleto). | — |
| `has_voucher` | Integer | `0` | Flag: order paid using promotional voucher. | — |
| `has_debit_card` | Integer | `0` | Flag: order paid using debit card. | — |
| `review_score` | Integer | `4` | Customer rating submitted on 1-to-5 star scale. | — |
| `review_score_label` | Text | `4 - Good` | Human-readable star rating description. | — |
| `satisfaction_status` | Text | `Satisfied` | Cohort classification: `Satisfied`, `Dissatisfied`, `Neutral (Score 3)`. | — |
| `dissatisfied` | Integer | `0` | Target flag: `1` if `review_score` $\in \{1, 2\}$, else `0`. | — |
| `is_neutral` | Integer | `0` | Flag: `1` if `review_score` $= 3$, else `0`. | — |
| `is_satisfied` | Integer | `1` | Flag: `1` if `review_score` $\in \{4, 5\}$, else `0`. | — |
| `is_binary_sample` | Integer | `1` | Flag: `1` for ML cohort ($N = 87,908$, excluding Score 3), else `0`. | — |

---

### Table 2: `Dim_Categories.csv`
* **Grain:** One row per unique primary product category.
* **Row Count:** 74 categories.
* **Primary Key:** `primary_product_category`.

| Column Name | Data Type | Description |
|---|---|---|
| `primary_product_category` | Text | Normalized category name in English (Primary Key). |
| `total_orders` | Integer | Number of delivered orders in this category. |
| `dissatisfied_orders` | Integer | Number of dissatisfied orders (Scores 1–2). |
| `avg_review_score` | Decimal | Mean customer review score (1.00 to 5.00). |
| `avg_delivery_delay` | Decimal | Mean delivery delay in days. |
| `delayed_orders` | Integer | Number of delayed orders. |
| `avg_freight_ratio` | Decimal | Mean freight-to-value ratio. |
| `avg_order_value` | Decimal | Mean order ticket size in BRL. |
| `dissatisfaction_rate` | Decimal | Percentage of orders with scores 1–2. |
| `delay_rate` | Decimal | Percentage of orders delivered past SLA deadline. |
| `category_group` | Text | High-level macro department (e.g., Home, Tech, Fashion, Leisure). |

---

### Table 3: `Dim_Customer_Geo.csv`
* **Grain:** One row per Brazilian Federative Unit (State).
* **Row Count:** 27 states.
* **Primary Key:** `customer_state`.

| Column Name | Data Type | Description |
|---|---|---|
| `customer_state` | Text | 2-letter state code (Primary Key). |
| `state_name` | Text | Full official name of the state (e.g., São Paulo, Rio de Janeiro). |
| `macro_region` | Text | Official geographic region (Southeast, South, Northeast, Central-West, North). |
| `total_orders` | Integer | Total delivered order volume. |
| `dissatisfied_orders` | Integer | Count of dissatisfied orders. |
| `delayed_orders` | Integer | Count of delayed orders. |
| `avg_review_score` | Decimal | Average review rating. |
| `avg_delivery_duration`| Decimal | Average courier transit days. |
| `dissatisfaction_rate` | Decimal | Dissatisfaction rate in the state. |
| `delay_rate` | Decimal | Percentage of orders delayed in the state. |

---

### Table 4: `Dim_Model_Benchmark.csv`
* **Grain:** One row per evaluated machine learning classification algorithm.
* **Row Count:** 3 models.

| Column Name | Data Type | Description |
|---|---|---|
| `Model Name` | Text | Name and weighting strategy of model pipeline. |
| `Precision (Dissatisfied)`| Decimal | Precision for dissatisfied class ($Y=1$). |
| `Recall (Dissatisfied)` | Decimal | Recall for dissatisfied class ($Y=1$). |
| `F1-Score (Dissatisfied)` | Decimal | Harmonic mean of precision and recall. |
| `ROC-AUC` | Decimal | Area under the Receiver Operating Characteristic curve. |
| `PR-AUC (Avg Precision)` | Decimal | Area under the Precision-Recall curve (Primary Metric). |

---

### Table 5: `Dim_Confusion_Matrix.csv`
* **Grain:** One row per 2x2 confusion matrix outcome cell for champion HistGradientBoosting model.
* **Row Count:** 4 cells ($N_{\text{test}} = 17,582$).

| Column Name | Data Type | Sample Value | Description |
|---|---|---|---|
| `Actual Class` | Text | `Dissatisfied (1)` | True customer status in holdout test cohort. |
| `Predicted Class` | Text | `Dissatisfied (1)` | Model prediction at delivery confirmation. |
| `Count` | Integer | `1378` | Exact number of test cases falling into this quadrant. |
| `Outcome Type` | Text | `True Positive (TP)`| Diagnostic label: TP, FN, TN, FP. |
| `Business Meaning` | Text | `Correctly caught...` | Operational retention impact. |

---

### Table 6: `Dim_Feature_Importance.csv`
* **Grain:** One row per operational feature in the predictive model.
* **Row Count:** 30 features.

| Column Name | Data Type | Description |
|---|---|---|
| `Feature` | Text | Operational feature name. |
| `Importance Mean (ROC-AUC drop)` | Decimal | Mean degradation in ROC-AUC when feature values are permuted. |
| `Importance Std` | Decimal | Standard deviation of importance across permutation repeats. |
| `Rank` | Integer | Priority rank from 1 to 30. |

---

### Table 7: `Dim_NLP_Terms.csv`
* **Grain:** One row per extracted unigram/bigram.
* **Row Count:** 30 rows (Top 15 Dissatisfied + Top 15 Satisfied).

| Column Name | Data Type | Description |
|---|---|---|
| `Cohort` | Text | Cohort: `Dissatisfied (Scores 1-2)` vs `Satisfied (Scores 4-5)`. |
| `N-Gram` | Text | Portuguese term or bigram (e.g., `não recebi`, `antes prazo`). |
| `TF-IDF Sum Weight` | Decimal | Cumulative TF-IDF weight across customer reviews. |
| `English Translation` | Text | Direct English translation. |
| `Root Cause Theme` | Text | Operational categorization of friction or delight. |

---

### Table 8: `Dim_Structured_vs_Text.csv`
* **Grain:** One row per friction dimension.
* **Row Count:** 4 dimensions.

| Column Name | Data Type | Description |
|---|---|---|
| `Friction Dimension` | Text | Operational friction theme (e.g., Delivery SLA, Quality, Service, Financial). |
| `Structured Data Findings` | Text | Empirical findings from tabular columns and model feature importance. |
| `NLP Text Feedback Findings`| Text | Corroborating voice-of-customer terms extracted via TF-IDF. |
| `Relationship` | Text | Synthesis relationship: `Strong Mutual Support` or `Complementary Insight`. |
| `Business Interpretation` | Text | Managerial impact and operational root cause. |

