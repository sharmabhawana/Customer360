"""
Dataset Inspection Script for Olist Brazilian E-Commerce
Customer360 AI - COM-702 Applied Data Science
"""

import os
import json
import pandas as pd
import numpy as np

RAW_DATA_DIR = os.path.join("data", "raw")
TABLES_DIR = os.path.join("outputs", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

CSV_FILES = [
    "olist_customers_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv"
]

def inspect_datasets():
    summary_list = []
    col_details = {}
    dfs = {}

    print("=" * 80)
    print("1. DATASET INVENTORY AND DIMENSIONS")
    print("=" * 80)

    for fname in CSV_FILES:
        fpath = os.path.join(RAW_DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"MISSING: {fname}")
            continue

        file_size_mb = os.path.getsize(fpath) / (1024 * 1024)
        df = pd.read_csv(fpath)
        dfs[fname] = df

        num_rows, num_cols = df.shape
        null_count = df.isnull().sum().sum()
        total_cells = num_rows * num_cols
        null_pct = (null_count / total_cells * 100) if total_cells > 0 else 0

        summary_list.append({
            "File Name": fname,
            "Rows": num_rows,
            "Columns": num_cols,
            "Size (MB)": round(file_size_mb, 2),
            "Total Nulls": null_count,
            "Null Cell %": round(null_pct, 2)
        })

        # Columns breakdown
        col_info = []
        for col in df.columns:
            n_null = int(df[col].isnull().sum())
            pct_null = round((n_null / num_rows * 100), 2)
            n_unique = int(df[col].nunique())
            dtype = str(df[col].dtype)
            col_info.append({
                "column": col,
                "dtype": dtype,
                "null_count": n_null,
                "null_pct": pct_null,
                "unique_count": n_unique
            })
        col_details[fname] = col_info

    summary_df = pd.DataFrame(summary_list)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(TABLES_DIR, "dataset_inventory.csv"), index=False)

    print("\n" + "=" * 80)
    print("2. PRIMARY KEYS AND UNIQUENESS CHECKS")
    print("=" * 80)
    
    # Check customers
    cust = dfs.get("olist_customers_dataset.csv")
    if cust is not None:
        print(f"Customers: len={len(cust)}")
        print(f"  customer_id unique: {cust['customer_id'].nunique()} (is PK? {cust['customer_id'].nunique() == len(cust)})")
        print(f"  customer_unique_id unique: {cust['customer_unique_id'].nunique()} (repeating customers: {len(cust) - cust['customer_unique_id'].nunique()})")

    # Check orders
    orders = dfs.get("olist_orders_dataset.csv")
    if orders is not None:
        print(f"Orders: len={len(orders)}")
        print(f"  order_id unique: {orders['order_id'].nunique()} (is PK? {orders['order_id'].nunique() == len(orders)})")
        print(f"  customer_id unique: {orders['customer_id'].nunique()} (matches len? {orders['customer_id'].nunique() == len(orders)})")
        print("  order_status distribution:")
        print(orders["order_status"].value_counts().to_string())

    # Check order_items
    items = dfs.get("olist_order_items_dataset.csv")
    if items is not None:
        print(f"Order Items: len={len(items)}")
        print(f"  unique order_id: {items['order_id'].nunique()}")
        print(f"  order_item_id max: {items['order_item_id'].max()}")
        pk_test = items.duplicated(subset=['order_id', 'order_item_id']).sum()
        print(f"  composite (order_id, order_item_id) duplicates: {pk_test} (is composite PK? {pk_test == 0})")
        print(f"  unique product_id: {items['product_id'].nunique()}")
        print(f"  unique seller_id: {items['seller_id'].nunique()}")

    # Check payments
    payments = dfs.get("olist_order_payments_dataset.csv")
    if payments is not None:
        print(f"Order Payments: len={len(payments)}")
        print(f"  unique order_id: {payments['order_id'].nunique()}")
        pk_test_p = payments.duplicated(subset=['order_id', 'payment_sequential']).sum()
        print(f"  composite (order_id, payment_sequential) duplicates: {pk_test_p} (is composite PK? {pk_test_p == 0})")
        print("  payment_type distribution:")
        print(payments["payment_type"].value_counts().to_string())

    # Check reviews
    reviews = dfs.get("olist_order_reviews_dataset.csv")
    if reviews is not None:
        print(f"Order Reviews: len={len(reviews)}")
        print(f"  review_id unique: {reviews['review_id'].nunique()} (duplicates: {reviews['review_id'].duplicated().sum()})")
        print(f"  order_id unique: {reviews['order_id'].nunique()} (orders with >1 review: {reviews['order_id'].duplicated().sum()})")
        print("  review_score distribution:")
        print(reviews["review_score"].value_counts().sort_index().to_string())
        print(f"  review_comment_title null count: {reviews['review_comment_title'].isnull().sum()} ({reviews['review_comment_title'].isnull().mean()*100:.1f}%)")
        print(f"  review_comment_message null count: {reviews['review_comment_message'].isnull().sum()} ({reviews['review_comment_message'].isnull().mean()*100:.1f}%)")
        print(f"  At least one comment field present: {((~reviews['review_comment_title'].isnull()) | (~reviews['review_comment_message'].isnull())).sum()}")

    # Check products
    products = dfs.get("olist_products_dataset.csv")
    if products is not None:
        print(f"Products: len={len(products)}")
        print(f"  product_id unique: {products['product_id'].nunique()} (is PK? {products['product_id'].nunique() == len(products)})")

    # Check sellers
    sellers = dfs.get("olist_sellers_dataset.csv")
    if sellers is not None:
        print(f"Sellers: len={len(sellers)}")
        print(f"  seller_id unique: {sellers['seller_id'].nunique()} (is PK? {sellers['seller_id'].nunique() == len(sellers)})")

    print("\n" + "=" * 80)
    print("3. JOIN CARDINALITY & INTEGRITY CHECKS")
    print("=" * 80)
    
    # Orders to customers
    if orders is not None and cust is not None:
        unmatched_cust = set(orders["customer_id"]) - set(cust["customer_id"])
        print(f"Orders -> Customers unmatched customer_ids: {len(unmatched_cust)}")

    # Orders to reviews
    if orders is not None and reviews is not None:
        orders_with_reviews = set(orders["order_id"]).intersection(set(reviews["order_id"]))
        orders_without_reviews = set(orders["order_id"]) - set(reviews["order_id"])
        reviews_without_orders = set(reviews["order_id"]) - set(orders["order_id"])
        print(f"Orders with reviews: {len(orders_with_reviews)} / {len(orders)}")
        print(f"Orders without reviews: {len(orders_without_reviews)}")
        print(f"Reviews pointing to non-existent orders: {len(reviews_without_orders)}")

    # Orders to items
    if orders is not None and items is not None:
        orders_with_items = set(orders["order_id"]).intersection(set(items["order_id"]))
        orders_without_items = set(orders["order_id"]) - set(items["order_id"])
        print(f"Orders with items: {len(orders_with_items)} / {len(orders)}")
        print(f"Orders without items (e.g. cancelled/unavailable): {len(orders_without_items)}")

    # Orders to payments
    if orders is not None and payments is not None:
        orders_with_payments = set(orders["order_id"]).intersection(set(payments["order_id"]))
        orders_without_payments = set(orders["order_id"]) - set(payments["order_id"])
        print(f"Orders with payments: {len(orders_with_payments)} / {len(orders)}")
        print(f"Orders without payments: {len(orders_without_payments)}")

    # Items to products
    if items is not None and products is not None:
        unmatched_products = set(items["product_id"]) - set(products["product_id"])
        print(f"Items -> Products unmatched product_ids: {len(unmatched_products)}")

    # Items to sellers
    if items is not None and sellers is not None:
        unmatched_sellers = set(items["seller_id"]) - set(sellers["seller_id"])
        print(f"Items -> Sellers unmatched seller_ids: {len(unmatched_sellers)}")

    # Save detailed column metadata to JSON
    with open(os.path.join(TABLES_DIR, "dataset_columns_metadata.json"), "w") as f:
        json.dump(col_details, f, indent=2)

    print("\nDetailed column metadata saved to outputs/tables/dataset_columns_metadata.json")

if __name__ == "__main__":
    inspect_datasets()

