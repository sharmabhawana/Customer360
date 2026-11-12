# Customer360 AI: Enterprise Power BI Desktop Build Guide
**Module:** COM-702 (C) Applied Data Science — MIET Jammu  
**Project:** Customer360 AI: E-Commerce Customer Experience Analytics  
**Target User:** Power BI Developer & Academic Evaluation Panel  
**Author:** Bhawana Sharma (Roll No: 2023A1R076)

---

## 1. Design System & Theme Specifications

This build guide implements a modern, clean enterprise business intelligence dashboard inspired by Tier-1 corporate analytics platforms:
* **Background:** Canvas background is light slate `#F8FAFC`, card background is pure white `#FFFFFF`.
* **Primary Typography:** Segoe UI / Segoe UI Semibold.
* **Palette:**
  * **Deep Navy (`#0F172A`):** Used for report headers, primary metric values, and prominent axis titles.
  * **Neutral Teal & Cool Blue (`#0F766E` / `#0284C7` / `#1D4ED8`):** Used for satisfied cohorts, baseline punctuality, and model benchmarks.
  * **Controlled Crimson Red (`#DC2626`):** Reserved **strictly** for customer dissatisfaction, courier delivery delay, and operational risk.
  * **Neutral Slate (`#94A3B8` / `#64748B`):** Used for neutral score 3 reviews, metric category labels, and gridlines.
* **Currency Formatting:** Monetary figures retain the Brazilian Real symbol (`R$`) with two decimal places.

---

## 2. Setting Up Power BI Desktop

### Step 2.1: Launch Power BI Desktop
1. Open **Microsoft Power BI Desktop** from your Windows Start Menu (executable located at `C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe`).
2. Close the initial splash dialog.

### Step 2.2: Load Import-Ready Data Tables
1. On the **Home** ribbon, click **Get Data** $\rightarrow$ **Text/CSV**.
2. Navigate to your project directory:  
   `c:\Users\BHAWANA SHARMA\OneDrive\Desktop\Customer360_AI_COM702\outputs\powerbi\`
3. Import **`Fact_Orders.csv`** (95,824 rows). Confirm delimiter is **Comma** and encoding is **UTF-8**, then click **Load**.
4. Repeat this step for the 7 supporting dimension tables:
   * `Dim_Categories.csv` (74 product categories)
   * `Dim_Customer_Geo.csv` (27 Brazilian states)
   * `Dim_Model_Benchmark.csv` (3 evaluated algorithms)
   * `Dim_Confusion_Matrix.csv` (Exact $2\times 2$ matrix, 17,582 cases)
   * `Dim_Feature_Importance.csv` (Top 30 permutation importance rankings)
   * `Dim_NLP_Terms.csv` (Top 30 Portuguese TF-IDF n-grams)
   * `Dim_Structured_vs_Text.csv` (4-theme multimodal triangulation table)

---

### Step 2.3: Establish Star Schema Relationships
1. Switch to **Model view** on the left navigation bar.
2. Link the core dimension tables to `Fact_Orders`:
   * **Categories:** Drag `primary_product_category` from `Dim_Categories` onto `primary_product_category` in `Fact_Orders` (Cardinality: **1 to Many**, Cross-filter: **Single**).
   * **Geography:** Drag `customer_state` from `Dim_Customer_Geo` onto `customer_state` in `Fact_Orders` (Cardinality: **1 to Many**, Cross-filter: **Single**).
3. The remaining reference tables serve as evaluation dimension tables for Page 3.

---

### Step 2.4: Apply the Corporate Theme
1. On the top ribbon, select the **View** tab.
2. In the **Themes** dropdown, click **Browse for themes...**.
3. Select `outputs\powerbi\Customer360_Theme.json` and click **Open**.  
   *(The palette, visual margins, and card typography will automatically synchronize).*

---

## 3. Page-by-Page Visual Construction Guide

### Page 1: Customer Overview
* **Canvas Settings:** Set Page Size to $16:9$, Canvas Background to `#F8FAFC`.
* **Top Header:** Insert a **Text Box** across the top:
  * Title: `Customer360 AI: E-Commerce Customer Experience Analytics` (Font: 16 pt Bold, `#F8FAFC`, background: `#0F172A`).
  * Subtitle: `COM-702 (C) Applied Data Science | Model Institute of Engineering and Technology (MIET), Jammu` (Font: 10 pt, `#94A3B8`).

* **Row 1: KPI Cards (4 Cards across top row):**
  1. **Card 1 (Total Orders):** Field $\rightarrow$ `Fact_Orders[order_id]` (Count Distinct). Title: `TOTAL DELIVERED & REVIEWED ORDERS`. Formatted: `95,824`. Subtext: `Delivered Order Population (100%)`.
  2. **Card 2 (Satisfied Orders):** Measure $\rightarrow$ `[Satisfied Orders (Scores 4-5)]`. Title: `SATISFIED ORDERS (SCORES 4-5)`. Formatted: `75,636` (Teal `#0F766E`). Subtext: `78.93% of Total Population`.
  3. **Card 3 (Dissatisfied Orders):** Measure $\rightarrow$ `[Dissatisfied Orders (Scores 1-2)]`. Title: `DISSATISFIED ORDERS (SCORES 1-2)`. Formatted: `12,272` (Crimson `#DC2626`). Subtext: `12.81% of Total Population`.
  4. **Card 4 (Dissatisfaction Rate):** Measure $\rightarrow$ `[Dissatisfaction Rate (Total Delivered)]`. Title: `OVERALL DISSATISFACTION RATE`. Formatted: `12.81%` (Crimson `#DC2626`). Subtext: `Baseline E-Commerce Dissatisfaction`.

* **Row 2: Review Score Distribution & Donut Chart:**
  1. **Visual 1 (Column Chart):** Clustered Column Chart.
     * X-axis: `Fact_Orders[review_score]`.
     * Y-axis: `Fact_Orders[order_id]` (Count).
     * Data Labels: On.
     * Bar Colors: Score 1 (`#DC2626`), Score 2 (`#EF4444`), Score 3 (`#94A3B8`), Score 4 (`#0284C7`), Score 5 (`#0F766E`).
  2. **Visual 2 (Donut Chart):** Donut Chart.
     * Legend: `Fact_Orders[satisfaction_status]`.
     * Values: `Fact_Orders[order_id]` (Count).
     * Slice Colors: Satisfied (`#0F766E`), Neutral (`#94A3B8`), Dissatisfied (`#DC2626`).

* **Row 3: Product Category & State Distribution:**
  1. **Visual 3 (Horizontal Bar Chart - Top 15 Categories by Dissatisfaction):**
     * Visual Type: **Clustered Bar Chart** (Horizontal).
     * Y-axis: `Dim_Categories[primary_product_category]` (or `Fact_Orders[primary_product_category]`).
     * X-axis: Measure $\rightarrow$ **`[Category Dissatisfaction Rate]`** (Dynamic DAX Measure, NOT the static dimension column).
       * *Why this is essential:* Using the dynamic `[Category Dissatisfaction Rate]` measure guarantees that when a user selects a state (e.g., `RJ` or `SP`) or a delivery status (`Delayed`), the bar chart immediately recalculates the exact dissatisfaction rate for that filtered subset.
     * Filter: In the Filters pane under `primary_product_category`, select **Filter Type: Top N** $\rightarrow$ Show **Top 15** $\rightarrow$ By Value: **`[Category Dissatisfaction Rate]`**.
     * Data Labels: On.
     * Bar Color: Crimson Red (`#DC2626`).
  2. **Visual 4 (Horizontal Bar Chart - Customer Order Distribution by State):**
     * Visual Type: **Clustered Bar Chart** (Horizontal).
     * Y-axis: `Dim_Customer_Geo[customer_state]` (or `Fact_Orders[customer_state]`).
     * X-axis: Measure $\rightarrow$ **`[Total Delivered Orders]`** (`COUNTROWS('Fact_Orders')`).
     * Tooltips: Drag **`[Overall Dissatisfaction Rate]`** and `Dim_Customer_Geo[state_name]`.
     * Filter: Filter Type: **Top N** $\rightarrow$ Show **Top 10** by `[Total Delivered Orders]`.
     * Color: Navy `#1E293B`.
     * **Cross-Filtering Interactive Behavior:**
       * In Power BI, clicking any state bar (e.g., `SP` or `RJ`) automatically cross-filters all other visuals on the canvas.
       * Because Visual 3 uses the dynamic measure `[Category Dissatisfaction Rate]`, the category bars will dynamically recalculate and re-sort to show category vulnerability specifically within that clicked state.
       * The top KPI cards will also immediately slice to show the state's exact order volume and dissatisfaction rate. Click the bar again (or white space) to reset the filter.

---

### Page 2: Delivery Performance
* **Top Header:** Section heading indicating logistics SLA dynamics.
* **Row 1: KPI Cards (5 Cards across top row):**
  1. **Card 1:** `[On-Time Delivered Orders]` $\rightarrow$ `88,163` (Teal `#0F766E`). Label: `ON-TIME ORDERS`.
  2. **Card 2:** `[Delayed Delivered Orders]` $\rightarrow$ `7,661` (Crimson `#DC2626`). Label: `DELAYED ORDERS`.
  3. **Card 3:** `[Delivery Delay Rate %]` $\rightarrow$ `8.00%`. Label: `DELAY RATE`.
  4. **Card 4:** `[On-Time Dissatisfaction Rate]` $\rightarrow$ `9.22%` (Teal `#0F766E`). Label: `ON-TIME DISSATISFACTION`.
  5. **Card 5:** `[Delayed Dissatisfaction Rate]` $\rightarrow$ `54.07%` (Crimson `#DC2626`). Label: `DELAYED DISSATISFACTION (5.86x SURGE)`.

* **Methodological Notice Box:**
  Insert a text box with light blue background (`#EFF6FF`) and border:
  > *"Denominator Notice: The on-time (9.22%) and delayed (54.07%) dissatisfaction rates strictly utilize the full delivered and reviewed population (N = 95,824) as denominator. In the binary ML modeling cohort (N = 87,908, which excludes 7,916 neutral score 3 reviews), the corresponding rates are 10.02% for on-time and 61.01% for delayed orders. Both statistics are mathematically valid for their defined populations."*

* **Row 2: Delivery Comparison & Delay Distribution:**
  1. **Visual 1 (100% Stacked Bar Chart):**
     * Y-axis: `Fact_Orders[delivery_status]`.
     * X-axis: `Fact_Orders[order_id]` (Count).
     * Legend: `Fact_Orders[satisfaction_status]`.
     * Colors: Satisfied (`#0F766E`), Dissatisfied (`#DC2626`).
  2. **Visual 2 (Column / Area Chart):**
     * X-axis: `Fact_Orders[delivery_delay_days]`.
     * Y-axis: Count of orders.
     * Add Constant Line on X-axis at `0`, dashed red line, labeled `Promised Delivery SLA (0 Days)`.

* **Row 3: Freight Ratio & Financing Commitments:**
  1. **Visual 3 (Boxplot / Clustered Bar):**
     * X-axis: `review_score`.
     * Y-axis: `freight_ratio` (Average).
  2. **Visual 4 (Column Chart):**
     * X-axis: `Fact_Orders[installment_tier]`.
     * Y-axis: Average of `Fact_Orders[dissatisfied]`.
     * Format as Percentage.

---

### Page 3: Machine Learning and NLP
* **Row 1: Champion Model KPI Cards (4 Cards across top row):**
  1. **Card 1:** Champion Algorithm $\rightarrow$ `HistGradientBoosting (Balanced)`.
  2. **Card 2:** Primary Metric $\rightarrow$ `0.5291 PR-AUC` (Blue `#1D4ED8`, 3.79x baseline).
  3. **Card 3:** Proactive Recall $\rightarrow$ `56.15%` (Teal `#0F766E`, 1,378 at-risk orders caught).
  4. **Card 4:** ROC-AUC Benchmark $\rightarrow$ `0.7746` (RF: 0.7661 | LR: 0.7653).

* **Row 2: Model Comparison & Confusion Matrix:**
  1. **Visual 1 (Clustered Column Chart):**
     * X-axis: `Dim_Model_Benchmark[Model Name]`.
     * Y-axis: `Precision`, `Recall`, `F1-Score`, `ROC-AUC`, `PR-AUC`.
  2. **Visual 2 (Matrix Visual - Confusion Matrix):**
     * Rows: `Dim_Confusion_Matrix[Actual Class]`.
     * Columns: `Dim_Confusion_Matrix[Predicted Class]`.
     * Values: `Dim_Confusion_Matrix[Count]` (Sum).
     * Shows exact integer cells: $\text{TN} = 13,187$, $\text{FP} = 1,941$, $\text{FN} = 1,076$, $\text{TP} = 1,378$.

* **Row 3: Feature Importance & NLP Terms:**
  1. **Visual 3 (Horizontal Bar Chart):**
     * Y-axis: `Dim_Feature_Importance[Feature]`.
     * X-axis: `Dim_Feature_Importance[Importance Mean (ROC-AUC drop)]`.
     * Color: Teal `#0F766E`.
  2. **Visual 4 (Horizontal Clustered Bar Chart):**
     * Y-axis: `Dim_NLP_Terms[N-Gram]`.
     * X-axis: `Dim_NLP_Terms[TF-IDF Sum Weight]`.
     * Legend: `Dim_NLP_Terms[Cohort]`.
     * Colors: Dissatisfied (`#DC2626`), Satisfied (`#0F766E`).

* **Row 4: Multimodal Synthesis & Actionable Recommendations:**
  1. **Visual 5 (Table Visual):**
     * Columns: `Friction Dimension`, `Structured Data Findings`, `NLP Text Feedback Findings`, `Relationship`, `Business Interpretation`.
  2. **Strategic Recommendation Card:**
     * White container with green accent border (`#0F766E`):
     * *Key Insights:* (1) Delivery delay is the single strongest operational predictor of defection (5.86x escalation). (2) Text feedback validates logistics failure (`não recebi`) and product defects (`veio defeito`).
     * *Practical Recommendation:* Deploy an automated **Proactive Customer Retention Engine** at delivery confirmation. When courier telemetry marks an order as delayed, automatically dispatch compensatory store credit (R$ 15–25) and route customer inquiries to priority tier-1 support before the customer submits a review.

---

## 4. Saving the Official PBIX File

1. In Power BI Desktop, click **File** $\rightarrow$ **Save As**.
2. Select your folder:  
   `c:\Users\BHAWANA SHARMA\OneDrive\Desktop\Customer360_AI_COM702\outputs\powerbi\`
3. Name the file:  
   **`Customer360_AI_Executive_Dashboard.pbix`**
4. Click **Save**. Your official Power BI file is now saved and ready for academic submission and viva demonstration!
