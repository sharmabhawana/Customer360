"""
Generate Jupyter Notebook for Customer360 AI Task 2.
"""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Header
cells.append(nbf.v4.new_markdown_cell("""# Customer360 AI: Predicting E-Commerce Customer Dissatisfaction
## Task 2: Data Cleaning, Integration, Feature Engineering, and Exploratory Data Analysis (EDA)
**Course:** COM-702 Applied Data Science  
**Dataset:** Brazilian E-Commerce Public Dataset by Olist (9 Relational CSVs)  
**Author:** CSE Data Science Research Team  

---

### Notebook Overview
This notebook presents the complete, reproducible data engineering and exploratory data analysis pipeline for the Customer360 AI project:
1. **Data Quality Assessment**: Structural inspection of all 9 relational tables, missingness patterns, and integrity validation.
2. **Review Handling & Corpus Preservation**: Deterministic tie-breaking for order-level ML while preserving all 99,224 original review texts for NLP diagnostics.
3. **Relational Integration & Aggregation**: Aggregating line items, products, and payments to the unique `order_id` grain prior to 1-to-1 merging (preventing Cartesian explosion).
4. **Target Formulation & Post-Fulfilment Timing**: Ground-truth dissatisfaction defined as Review Score $\\in \\{1, 2\\}$ vs Satisfied $\\in \\{4, 5\\}$, cleanly excluding neutral Score 3.
5. **Feature Engineering**: Deriving operational delivery metrics, financial friction ratios, physical item properties, and payment structures.
6. **Exploratory Data Analysis**: 6 publication-standard visual analyses revealing key empirical drivers of customer friction.
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""# Imports & Configuration
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

RAW_DIR = os.path.join("..", "data", "raw")
PROCESSED_DIR = os.path.join("..", "data", "processed")
FIGURES_DIR = os.path.join("..", "outputs", "figures")
TABLES_DIR = os.path.join("..", "outputs", "tables")
"""))

# Section 1: Data Loading & Verification
cells.append(nbf.v4.new_markdown_cell("""## 1. Load Raw Datasets & Data Quality Audit
We load all nine CSV files from `data/raw/` and inspect primary keys, dimensions, and anomalies.
"""))

cells.append(nbf.v4.new_code_cell("""# Load the 9 Olist tables
customers = pd.read_csv(os.path.join(RAW_DIR, 'olist_customers_dataset.csv'))
orders = pd.read_csv(os.path.join(RAW_DIR, 'olist_orders_dataset.csv'))
items = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_items_dataset.csv'))
payments = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_payments_dataset.csv'))
reviews = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_reviews_dataset.csv'))
products = pd.read_csv(os.path.join(RAW_DIR, 'olist_products_dataset.csv'))
sellers = pd.read_csv(os.path.join(RAW_DIR, 'olist_sellers_dataset.csv'))
geolocation = pd.read_csv(os.path.join(RAW_DIR, 'olist_geolocation_dataset.csv'))
translation = pd.read_csv(os.path.join(RAW_DIR, 'product_category_name_translation.csv'))

print(f"Customers:    {customers.shape}")
print(f"Orders:       {orders.shape}")
print(f"Order Items:  {items.shape}")
print(f"Payments:     {payments.shape}")
print(f"Reviews:      {reviews.shape}")
print(f"Products:     {products.shape}")
print(f"Sellers:      {sellers.shape}")
print(f"Geolocation:  {geolocation.shape}")
print(f"Translation:  {translation.shape}")
"""))

# Section 2: Data Quality Report
cells.append(nbf.v4.new_markdown_cell("""## 2. Data Quality & Cleaning Decisions
We review the structured data quality audit table generated during pipeline execution.
"""))

cells.append(nbf.v4.new_code_cell("""# Display Data Quality Report
dq_df = pd.read_csv(os.path.join(TABLES_DIR, 'data_quality_report.csv'))
display(dq_df)
"""))

# Section 3: Review Handling
cells.append(nbf.v4.new_markdown_cell("""## 3. Review Deduplication & Corpus Preservation
- **Preservation**: All 99,224 original review records are preserved in `data/processed/customer360_reviews_nlp.csv` for Question 2 NLP analysis.
- **Order-Level ML Selection**: For the 551 orders with multiple reviews, we apply a deterministic tie-break: latest `review_answer_timestamp` $\\rightarrow$ latest `review_creation_date` $\\rightarrow$ highest `review_id`.
"""))

cells.append(nbf.v4.new_code_cell("""# Review deduplication verification
reviews['answer_dt'] = pd.to_datetime(reviews['review_answer_timestamp'], errors='coerce')
reviews['creation_dt'] = pd.to_datetime(reviews['review_creation_date'], errors='coerce')

sorted_rev = reviews.sort_values(by=['answer_dt', 'creation_dt', 'review_id'], ascending=[False, False, False])
rev_order_level = sorted_rev.drop_duplicates(subset=['order_id'], keep='first')

print(f"Total reviews: {len(reviews):,}")
print(f"Unique orders with reviews: {rev_order_level['order_id'].nunique():,}")
print(f"Multi-review orders resolved: {reviews['order_id'].duplicated().sum():,}")
"""))

# Section 4: Data Integration & Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 4. Relational Integration & Feature Engineering
We aggregate `order_items` and `order_payments` to the unique `order_id` grain, derive delivery SLA metrics from `orders`, and merge with 1-to-1 validation.
"""))

cells.append(nbf.v4.new_code_cell("""# Inspect the final processed order-level dataset
cohort = pd.read_csv(os.path.join(PROCESSED_DIR, 'customer360_order_level.csv'))
print(f"Delivered Cohort Shape: {cohort.shape}")
print(f"Columns: {list(cohort.columns)[:10]} ... ({len(cohort.columns)} total columns)")

print(f"\\nBinary Sample Distribution (Excluding Score 3):")
bin_cohort = cohort[cohort['is_binary_sample'] == 1]
print(bin_cohort['dissatisfied'].value_counts(normalize=True).round(4) * 100)
"""))

# Section 5: Feature Summary Table
cells.append(nbf.v4.new_markdown_cell("""## 5. Statistical Feature Summary Table
The table below documents descriptive statistics and Pearson linear correlation with customer dissatisfaction ($Y=1$).
"""))

cells.append(nbf.v4.new_code_cell("""# Feature summary
fs_df = pd.read_csv(os.path.join(TABLES_DIR, 'feature_summary.csv'))
display(fs_df)
"""))

# Section 6: Visual Exploratory Data Analysis
cells.append(nbf.v4.new_markdown_cell("""## 6. Publication-Quality Visual Exploratory Data Analysis
Below are the six diagnostic charts examining the operational, category, financial, temporal, and payment determinants of dissatisfaction.
"""))

cells.append(nbf.v4.new_markdown_cell("""### EDA 1: Review Score Distribution & Target Demarcation
Demarcates Dissatisfied ($1\\star, 2\\star$), Neutral ($3\\star$), and Satisfied ($4\\star, 5\\star$).
"""))
cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(FIGURES_DIR, 'eda_1_review_score_distribution.png')))\n"""))

cells.append(nbf.v4.new_markdown_cell("""### EDA 2: Delivery SLA Breach vs Customer Dissatisfaction
Demonstrates the dramatic 5.86x jump in dissatisfaction when orders miss promised delivery dates.
"""))
cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(FIGURES_DIR, 'eda_2_dissatisfaction_by_delivery_performance.png')))\n"""))

cells.append(nbf.v4.new_markdown_cell("""### EDA 3: Product Category Disparity
Compares dissatisfaction rates across the top 15 product categories by volume.
"""))
cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(FIGURES_DIR, 'eda_3_product_category_dissatisfaction.png')))\n"""))

cells.append(nbf.v4.new_markdown_cell("""### EDA 4: Financial Friction vs Customer Satisfaction
Boxplot comparison of Freight Ratio and Total Order Value between satisfied and dissatisfied customers.
"""))
cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(FIGURES_DIR, 'eda_4_freight_and_order_value_vs_satisfaction.png')))\n"""))

cells.append(nbf.v4.new_markdown_cell("""### EDA 5: Delivery Delay Distribution
KDE and histogram distribution of delivery delays relative to SLA estimates.
"""))
cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(FIGURES_DIR, 'eda_5_delivery_delay_distribution.png')))\n"""))

cells.append(nbf.v4.new_markdown_cell("""### EDA 6: Payment Configuration vs Satisfaction
Dissatisfaction across payment methods and installment tiers.
"""))
cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(FIGURES_DIR, 'eda_6_payment_patterns_vs_satisfaction.png')))\n"""))

# Section 7: Key Insights Table
cells.append(nbf.v4.new_markdown_cell("""## 7. Synthesis of Empirical Findings & Viva Takeaways
"""))

cells.append(nbf.v4.new_code_cell("""eda_insights = pd.read_csv(os.path.join(TABLES_DIR, 'eda_insights.csv'))
display(eda_insights)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Key Methodological Conclusions for Academic Defense
1. **Target Leakage Strict Avoidance**: Features derived post-review (ratings, review comments, survey timestamps) are strictly quarantined from the ML feature matrix. All predictive features reflect operational fulfillment state at delivery time.
2. **Class Imbalance Readiness**: Dissatisfaction is observed in 13.96% of the binary classification sample ($12,272$ vs $75,636$). Conventional accuracy is an inappropriate metric; Task 3 must prioritize PR-AUC, F1-Score, and Precision-Recall trade-offs.
3. **Logistics as the Primary Friction Channel**: Delivery delay relative to estimated SLA exhibits the highest individual feature correlation ($r = +0.3927$). Late delivery increases the odds of customer dissatisfaction by nearly six-fold.
"""))

nb['cells'] = cells

with open(os.path.join('notebooks', 'Customer360_Task2.ipynb'), 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at notebooks/Customer360_Task2.ipynb")

