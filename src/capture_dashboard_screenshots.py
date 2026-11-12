"""
Capture genuine high-resolution screenshots from running Streamlit dashboard
using local Microsoft Edge browser via Playwright.
Automatically copies to artifact directory for immediate UI viewing.
"""

import time
import os
import shutil
from playwright.sync_api import sync_playwright

output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(output_dir, exist_ok=True)

art_dir = r"C:\Users\BHAWANA SHARMA\.gemini\antigravity\brain\152b387e-3cfe-44d3-972c-6d18cca1efa6"

url = "http://localhost:8501"

print(f"Connecting to Streamlit dashboard at {url}...")

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    # 1600x1050 viewport with 1.5 device scale for crisp presentation resolution
    context = browser.new_context(
        viewport={"width": 1600, "height": 1050},
        device_scale_factor=1.5
    )
    page = context.new_page()
    
    # 1. Load dashboard
    page.goto(url, wait_until="networkidle", timeout=45000)
    page.wait_for_selector(".top-header-strip", timeout=30000)
    time.sleep(3) # allow Plotly figures to render animations
    
    # Check Title
    print("Page Title:", page.title())
    
    # Screenshot Tab 1
    tab1_path = os.path.join(output_dir, "dashboard_page1_customer_overview.png")
    page.screenshot(path=tab1_path, full_page=True)
    print(f"Captured Tab 1: {tab1_path} ({os.path.getsize(tab1_path):,} bytes)")
    shutil.copy2(tab1_path, os.path.join(art_dir, "dashboard_page1_customer_overview.png"))
    
    # 2. Switch to Tab 2
    tabs = page.locator('button[data-baseweb="tab"]')
    print(f"Found {tabs.count()} tabs.")
    
    # Click Tab 2
    tabs.nth(1).click()
    time.sleep(3)
    tab2_path = os.path.join(output_dir, "dashboard_page2_delivery_performance.png")
    page.screenshot(path=tab2_path, full_page=True)
    print(f"Captured Tab 2: {tab2_path} ({os.path.getsize(tab2_path):,} bytes)")
    shutil.copy2(tab2_path, os.path.join(art_dir, "dashboard_page2_delivery_performance.png"))
    
    # 3. Switch to Tab 3
    tabs.nth(2).click()
    time.sleep(3)
    tab3_path = os.path.join(output_dir, "dashboard_page3_ml_and_nlp.png")
    page.screenshot(path=tab3_path, full_page=True)
    print(f"Captured Tab 3: {tab3_path} ({os.path.getsize(tab3_path):,} bytes)")
    shutil.copy2(tab3_path, os.path.join(art_dir, "dashboard_page3_ml_and_nlp.png"))
    
    browser.close()

print("All updated dashboard screenshots captured and copied to artifact directory successfully!")
