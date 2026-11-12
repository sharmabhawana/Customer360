import time
from playwright.sync_api import sync_playwright

url = 'http://localhost:8501'
print('Testing Streamlit filters at', url)

with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1600, 'height': 1100})
    page.goto(url, wait_until='networkidle', timeout=30000)
    page.wait_for_selector('.main-header', timeout=15000)
    time.sleep(3)
    
    # 1. Check baseline KPI card values
    metric_values = page.locator('.metric-value').all_text_contents()
    print('Baseline KPI Metric Values:', metric_values[:4])
    assert '95,824' in metric_values[0], f'Expected 95,824 in top card, got {metric_values[0]}'
    assert '75,636' in metric_values[1], f'Expected 75,636 in satisfied card, got {metric_values[1]}'
    assert '12,272' in metric_values[2], f'Expected 12,272 in dissat card, got {metric_values[2]}'
    print('-> Baseline KPI verification PASSED.')
    
    # 2. Test Delivery Status Filter in Sidebar
    print('Testing Delivery Status filter in sidebar...')
    selectboxes = page.locator('div[data-testid="stSelectbox"]')
    print(f'Found {selectboxes.count()} selectboxes.')
    
    # Selectbox 1 is Delivery status (Selectbox 0 is category group)
    delivery_select = selectboxes.nth(1)
    delivery_select.click()
    time.sleep(1)
    
    # Click Delayed option
    page.keyboard.press('ArrowDown')
    page.keyboard.press('ArrowDown')
    page.keyboard.press('Enter')
    time.sleep(3)
    
    filtered_metrics = page.locator('.metric-value').all_text_contents()
    print('Filtered Metrics (Delayed Orders):', filtered_metrics[:4])
    assert '7,661' in filtered_metrics[0], f'Expected 7,661 delayed orders, got {filtered_metrics[0]}'
    assert '54.07%' in filtered_metrics[3], f'Expected 54.07% dissat rate, got {filtered_metrics[3]}'
    print('-> Delivery filter verification PASSED: 7,661 orders, 54.07% dissatisfaction rate!')
    
    # 3. Test Tab 2 under filter
    tabs = page.locator('button[data-baseweb="tab"]')
    tabs.nth(1).click()
    time.sleep(2)
    tab2_metrics = page.locator('.metric-value').all_text_contents()
    print('Tab 2 Metrics under Delayed Filter:', tab2_metrics[4:8])
    assert '7,661' in tab2_metrics[5], f'Expected 7,661 delayed orders on Tab 2, got {tab2_metrics[5]}'
    print('-> Tab 2 dynamic update PASSED.')
    
    # 4. Test Tab 3
    tabs.nth(2).click()
    time.sleep(2)
    tab3_metrics = page.locator('.metric-value').all_text_contents()
    print('Tab 3 Metrics:', tab3_metrics[8:12])
    assert 'HistGradientBoosting' in tab3_metrics[8]
    assert '0.5291' in tab3_metrics[9]
    assert '56.15%' in tab3_metrics[10]
    print('-> Tab 3 ML & NLP verification PASSED.')
    
    browser.close()

print('ALL STREAMLIT FILTER TESTS COMPLETED SUCCESSFULLY!')

