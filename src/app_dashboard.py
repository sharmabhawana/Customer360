"""
Customer360 AI: Enterprise Customer Experience Analytics Dashboard
Framework: Streamlit 1.57 + Plotly 7.1 Enterprise Visual System
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ENTERPRISE DESIGN SYSTEM
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer360 AI | Customer Experience Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Modern Enterprise CSS (White / Cool Slate / Deep Navy / Controlled Red)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1E293B;
        background-color: #F8FAFC;
    }
    
    /* Top Header Strip */
    .top-header-strip {
        background-color: #0F172A;
        border-bottom: 2px solid #1E293B;
        padding: 16px 24px;
        margin-bottom: 20px;
        border-radius: 6px;
    }
    
    .top-header-title {
        font-size: 20px;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.3px;
    }
    
    .top-header-subtitle {
        font-size: 12px;
        color: #94A3B8;
        margin: 4px 0 0 0;
        font-weight: 500;
    }
    
    /* Enterprise KPI Metric Card */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 12px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.1;
    }
    
    .kpi-value-danger {
        color: #DC2626 !important;
    }
    
    .kpi-value-success {
        color: #0F766E !important;
    }
    
    .kpi-value-primary {
        color: #1D4ED8 !important;
    }
    
    .kpi-subtext {
        font-size: 11px;
        color: #64748B;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Section Container & Notice Box */
    .section-notice {
        background-color: #EFF6FF;
        border-left: 3px solid #2563EB;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 12px;
        color: #1E3A8A;
        margin-bottom: 16px;
        line-height: 1.4;
    }
    
    .recommendation-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0F766E;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 14px;
    }
    
    .recommendation-title {
        font-size: 13px;
        font-weight: 700;
        color: #0F766E;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .recommendation-text {
        font-size: 12px;
        color: #334155;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Clean Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #FFFFFF;
        border-bottom: 1px solid #E2E8F0;
        padding: 0 4px;
        gap: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 16px;
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
        border: none;
        background: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        color: #0F172A !important;
        border-bottom: 2px solid #0F172A !important;
    }
    
    /* Filter Sidebar Compact Styling */
    .sidebar-section-title {
        font-size: 11px;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 16px;
        margin-bottom: 8px;
    }
    
    /* Plotly Chart Container */
    .chart-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. DATA LOAD & CACHE
# ------------------------------------------------------------------------------
@st.cache_data
def load_verified_data():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "powerbi")
    
    fact_orders = pd.read_csv(os.path.join(base_dir, "Fact_Orders.csv"))
    dim_categories = pd.read_csv(os.path.join(base_dir, "Dim_Categories.csv"))
    dim_geo = pd.read_csv(os.path.join(base_dir, "Dim_Customer_Geo.csv"))
    dim_model = pd.read_csv(os.path.join(base_dir, "Dim_Model_Benchmark.csv"))
    dim_cm = pd.read_csv(os.path.join(base_dir, "Dim_Confusion_Matrix.csv"))
    dim_fi = pd.read_csv(os.path.join(base_dir, "Dim_Feature_Importance.csv"))
    dim_nlp = pd.read_csv(os.path.join(base_dir, "Dim_NLP_Terms.csv"))
    dim_svt = pd.read_csv(os.path.join(base_dir, "Dim_Structured_vs_Text.csv"))
    
    return fact_orders, dim_categories, dim_geo, dim_model, dim_cm, dim_fi, dim_nlp, dim_svt

fact_orders, dim_categories, dim_geo, dim_model, dim_cm, dim_fi, dim_nlp, dim_svt = load_verified_data()


# ------------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & FILTERING
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-size:14px; font-weight:700; color:#0F172A;'>Customer360 AI Filters</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px; color:#64748B;'>Multi-Dimensional Customer Experience Filters</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("<div class='sidebar-section-title'>Geographic Scope</div>", unsafe_allow_html=True)
    all_states = sorted(fact_orders['customer_state'].unique().tolist())
    selected_states = st.multiselect(
        "Filter by Brazilian State:",
        options=all_states,
        default=[],
        help="Select one or more states to cross-filter orders."
    )
    
    st.markdown("<div class='sidebar-section-title'>Product Department</div>", unsafe_allow_html=True)
    cat_groups = ['All Departments'] + sorted(dim_categories['category_group'].unique().tolist())
    selected_group = st.selectbox(
        "Product Department:",
        options=cat_groups,
        index=0
    )
    
    st.markdown("<div class='sidebar-section-title'>Logistics Fulfillment</div>", unsafe_allow_html=True)
    deliv_opts = ['All Deliveries', 'On-Time (<= SLA)', 'Delayed (> SLA)']
    selected_delivery = st.selectbox(
        "SLA Adherence Status:",
        options=deliv_opts,
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#64748B; line-height:1.4;'>
        <b>Data Integrity Note:</b><br>
        All figures derived from verified Olist e-commerce cohort (2016–2018). Zero synthetic fabrication.
    </div>
    """, unsafe_allow_html=True)

# Slicing Engine
filtered_orders = fact_orders.copy()
if selected_states:
    filtered_orders = filtered_orders[filtered_orders['customer_state'].isin(selected_states)]
if selected_group != 'All Departments':
    valid_cats = dim_categories[dim_categories['category_group'] == selected_group]['primary_product_category'].tolist()
    filtered_orders = filtered_orders[filtered_orders['primary_product_category'].isin(valid_cats)]
if selected_delivery != 'All Deliveries':
    filtered_orders = filtered_orders[filtered_orders['delivery_status'] == selected_delivery]


# ------------------------------------------------------------------------------
# 4. TOP EXECUTIVE HEADER
# ------------------------------------------------------------------------------
st.markdown("""
<div class="top-header-strip">
    <div class="top-header-title">Customer360 AI: Predicting E-Commerce Customer Dissatisfaction & Feedback Analysis</div>
    <div class="top-header-subtitle">Predictive Machine Learning & Natural Language Processing on E-Commerce Customer Experience</div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 5. TABBED INTERFACE (PAGES 1, 2, 3)
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Page 1: Customer Overview",
    "Page 2: Delivery Performance",
    "Page 3: Machine Learning and NLP"
])


# ==============================================================================
# PAGE 1: CUSTOMER OVERVIEW
# ==============================================================================
with tab1:
    tot_orders = len(filtered_orders)
    sat_orders = int((filtered_orders['review_score'] >= 4).sum())
    dissat_orders = int((filtered_orders['review_score'] <= 2).sum())
    dissat_pct = (dissat_orders / tot_orders * 100) if tot_orders > 0 else 0.0
    
    # Row 1: KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Delivered & Reviewed Orders</div>
            <div class="kpi-value">{tot_orders:,}</div>
            <div class="kpi-subtext">Delivered Order Population (100%)</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Satisfied Orders (Scores 4-5)</div>
            <div class="kpi-value kpi-value-success">{sat_orders:,}</div>
            <div class="kpi-subtext">{sat_orders/tot_orders*100:.2f}% of Total Population</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Dissatisfied Orders (Scores 1-2)</div>
            <div class="kpi-value kpi-value-danger">{dissat_orders:,}</div>
            <div class="kpi-subtext">{dissat_orders/tot_orders*100:.2f}% of Total Population</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Overall Dissatisfaction Rate</div>
            <div class="kpi-value kpi-value-danger">{dissat_pct:.2f}%</div>
            <div class="kpi-subtext">Baseline E-Commerce Dissatisfaction</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Row 2: Review Score Distribution & Donut Chart
    col_c1, col_c2 = st.columns([3, 2])
    
    with col_c1:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Review Score Distribution (1 to 5 Stars)</div>", unsafe_allow_html=True)
        score_df = filtered_orders['review_score'].value_counts().sort_index().reset_index()
        score_df.columns = ['Score', 'Orders']
        score_df['Percentage'] = (score_df['Orders'] / tot_orders * 100).round(1)
        
        # Color palette: Scores 1-2 crimson red, Score 3 neutral slate, Scores 4-5 calm blue/teal
        bar_colors = ['#DC2626', '#EF4444', '#94A3B8', '#0284C7', '#0F766E']
        
        fig_scores = px.bar(
            score_df,
            x='Score',
            y='Orders',
            text=score_df.apply(lambda r: f"{r['Orders']:,} ({r['Percentage']}%)", axis=1),
            color='Score',
            color_discrete_sequence=bar_colors
        )
        fig_scores.update_traces(
            textposition='outside',
            textfont=dict(size=11, color='#1E293B')
        )
        fig_scores.update_layout(
            template='plotly_white',
            showlegend=False,
            height=320,
            margin=dict(l=20, r=20, t=20, b=30),
            xaxis=dict(title=dict(text="Review Star Rating", font=dict(size=12, color='#475569')), tickmode='linear', dtick=1),
            yaxis=dict(title=dict(text="Order Count", font=dict(size=12, color='#475569')), showgrid=True, gridcolor='#F1F5F9')
        )
        st.plotly_chart(fig_scores, use_container_width=True)
        
    with col_c2:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Customer Satisfaction Split</div>", unsafe_allow_html=True)
        neut_orders = int((filtered_orders['review_score'] == 3).sum())
        donut_df = pd.DataFrame({
            'Cohort': ['Satisfied (Scores 4-5)', 'Neutral (Score 3)', 'Dissatisfied (Scores 1-2)'],
            'Orders': [sat_orders, neut_orders, dissat_orders]
        })
        fig_donut = px.pie(
            donut_df,
            names='Cohort',
            values='Orders',
            hole=0.60,
            color='Cohort',
            color_discrete_map={
                'Satisfied (Scores 4-5)': '#0F766E',
                'Neutral (Score 3)': '#94A3B8',
                'Dissatisfied (Scores 1-2)': '#DC2626'
            }
        )
        fig_donut.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=11, color='#FFFFFF')
        )
        fig_donut.update_layout(
            template='plotly_white',
            showlegend=False,
            height=320,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Row 3: Product Category Dissatisfaction & State Order Distribution
    col_c3, col_c4 = st.columns([3, 2])
    
    with col_c3:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Dissatisfaction Rate by Product Category (Top 15 Vulnerable)</div>", unsafe_allow_html=True)
        cat_agg = filtered_orders.groupby('primary_product_category').agg(
            total=('order_id', 'count'),
            dissat=('dissatisfied', 'sum')
        ).reset_index()
        cat_agg = cat_agg[cat_agg['total'] >= 50] # minimum statistical support
        cat_agg['dissat_rate'] = (cat_agg['dissat'] / cat_agg['total'] * 100).round(2)
        top_cats = cat_agg.sort_values('dissat_rate', ascending=True).tail(15)
        
        fig_cats = px.bar(
            top_cats,
            y='primary_product_category',
            x='dissat_rate',
            orientation='h',
            text=top_cats['dissat_rate'].apply(lambda v: f"{v:.1f}%"),
            color='dissat_rate',
            color_continuous_scale=['#FCA5A5', '#DC2626']
        )
        fig_cats.update_traces(
            textposition='outside',
            textfont=dict(size=10, color='#1E293B')
        )
        fig_cats.update_layout(
            template='plotly_white',
            coloraxis_showscale=False,
            height=380,
            margin=dict(l=20, r=30, t=10, b=20),
            xaxis=dict(title=dict(text="Dissatisfaction Rate (%)", font=dict(size=11, color='#475569')), showgrid=True, gridcolor='#F1F5F9'),
            yaxis=dict(title="", tickfont=dict(size=10, color='#1E293B'))
        )
        st.plotly_chart(fig_cats, use_container_width=True)
        
    with col_c4:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Customer Order Distribution by Brazilian State (Top 10)</div>", unsafe_allow_html=True)
        state_agg = filtered_orders.groupby('customer_state').agg(
            total=('order_id', 'count'),
            dissat=('dissatisfied', 'sum')
        ).reset_index()
        state_agg['dissat_pct'] = (state_agg['dissat'] / state_agg['total'] * 100).round(1)
        top_states = state_agg.sort_values('total', ascending=True).tail(10)
        
        fig_states = px.bar(
            top_states,
            y='customer_state',
            x='total',
            orientation='h',
            text=top_states.apply(lambda r: f"{r['total']:,} ({r['dissat_pct']}% dissat)", axis=1),
            color_discrete_sequence=['#1E293B']
        )
        fig_states.update_traces(
            textposition='outside',
            textfont=dict(size=10, color='#1E293B')
        )
        fig_states.update_layout(
            template='plotly_white',
            height=380,
            margin=dict(l=20, r=40, t=10, b=20),
            xaxis=dict(title=dict(text="Order Volume", font=dict(size=11, color='#475569')), showgrid=True, gridcolor='#F1F5F9'),
            yaxis=dict(title="", tickfont=dict(size=11, color='#1E293B'))
        )
        st.plotly_chart(fig_states, use_container_width=True)


# ==============================================================================
# PAGE 2: DELIVERY PERFORMANCE
# ==============================================================================
with tab2:
    ontime_orders = int((filtered_orders['is_delayed'] == 0).sum())
    delayed_orders = int((filtered_orders['is_delayed'] == 1).sum())
    delay_rate = (delayed_orders / tot_orders * 100) if tot_orders > 0 else 0.0
    ontime_dissat = (filtered_orders[filtered_orders['is_delayed'] == 0]['dissatisfied'].mean() * 100) if ontime_orders > 0 else 0.0
    delayed_dissat = (filtered_orders[filtered_orders['is_delayed'] == 1]['dissatisfied'].mean() * 100) if delayed_orders > 0 else 0.0
    
    # Row 1: KPI Cards (5 Columns)
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">On-Time Orders</div>
            <div class="kpi-value kpi-value-success">{ontime_orders:,}</div>
            <div class="kpi-subtext">{100-delay_rate:.2f}% Punctual Fulfillment</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Delayed Orders</div>
            <div class="kpi-value kpi-value-danger">{delayed_orders:,}</div>
            <div class="kpi-subtext">{delay_rate:.2f}% Delivery SLA Breach</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Delay Rate</div>
            <div class="kpi-value">{delay_rate:.2f}%</div>
            <div class="kpi-subtext">Orders Delivered After SLA</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">On-Time Dissatisfaction</div>
            <div class="kpi-value kpi-value-success">{ontime_dissat:.2f}%</div>
            <div class="kpi-subtext">Baseline Dissatisfaction Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with p5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Delayed Dissatisfaction</div>
            <div class="kpi-value kpi-value-danger">{delayed_dissat:.2f}%</div>
            <div class="kpi-subtext">5.86x Escalation Surge</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class="section-notice">
        <b>Methodological Denominator Note:</b> The on-time (9.22%) and delayed (54.07%) dissatisfaction rates strictly utilize the full delivered and reviewed population (N = 95,824) as denominator. In the binary ML modeling cohort (N = 87,908, which excludes 7,916 neutral score 3 reviews), the corresponding rates are 10.02% for on-time and 61.01% for delayed orders. Both statistics are mathematically valid for their defined populations.
    </div>
    """, unsafe_allow_html=True)
    
    # Row 2: Satisfaction Comparison by Delivery Status & Delay Distribution
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Satisfaction Comparison by Delivery Status (The 5.86x Escalation)</div>", unsafe_allow_html=True)
        status_df = pd.DataFrame({
            'Delivery Status': ['On-Time (<= SLA)', 'Delayed (> SLA)'],
            'Satisfied (%)': [100 - ontime_dissat, 100 - delayed_dissat],
            'Dissatisfied (%)': [ontime_dissat, delayed_dissat]
        })
        
        fig_status = go.Figure()
        fig_status.add_trace(go.Bar(
            y=status_df['Delivery Status'],
            x=status_df['Satisfied (%)'],
            name='Satisfied (Scores 4-5)',
            orientation='h',
            marker=dict(color='#0F766E'),
            text=status_df['Satisfied (%)'].apply(lambda v: f"{v:.1f}%"),
            textposition='inside',
            textfont=dict(color='#FFFFFF', size=11)
        ))
        fig_status.add_trace(go.Bar(
            y=status_df['Delivery Status'],
            x=status_df['Dissatisfied (%)'],
            name='Dissatisfied (Scores 1-2)',
            orientation='h',
            marker=dict(color='#DC2626'),
            text=status_df['Dissatisfied (%)'].apply(lambda v: f"{v:.1f}%"),
            textposition='inside',
            textfont=dict(color='#FFFFFF', size=11)
        ))
        fig_status.update_layout(
            barmode='stack',
            template='plotly_white',
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)),
            xaxis=dict(title=dict(text="Percentage of Orders (%)", font=dict(size=11, color='#475569')), range=[0, 100]),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_status, use_container_width=True)
        
    with col_p2:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Delivery Delay Distribution (Relative to Promised SLA)</div>", unsafe_allow_html=True)
        deliv_clip = filtered_orders[['delivery_delay_days']].dropna()
        deliv_clip = deliv_clip[(deliv_clip['delivery_delay_days'] >= -25) & (deliv_clip['delivery_delay_days'] <= 25)]
        
        fig_delay_dist = px.histogram(
            deliv_clip,
            x='delivery_delay_days',
            nbins=50,
            color_discrete_sequence=['#1E293B']
        )
        fig_delay_dist.add_vline(
            x=0,
            line_width=2,
            line_dash='dash',
            line_color='#DC2626',
            annotation_text="Promised Delivery SLA (0 Days)",
            annotation_font=dict(size=10, color='#DC2626')
        )
        fig_delay_dist.update_layout(
            template='plotly_white',
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(title=dict(text="Days from Promised Delivery Date (Negative = Early, Positive = Late)", font=dict(size=11, color='#475569')), showgrid=True, gridcolor='#F1F5F9'),
            yaxis=dict(title=dict(text="Order Count", font=dict(size=11, color='#475569')), showgrid=True, gridcolor='#F1F5F9')
        )
        st.plotly_chart(fig_delay_dist, use_container_width=True)
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Row 3: Freight Ratio & Financing Installments
    col_p3, col_p4 = st.columns(2)
    
    with col_p3:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Freight-to-Order Ratio Comparison by Review Score</div>", unsafe_allow_html=True)
        freight_clean = filtered_orders[filtered_orders['freight_ratio'] <= 0.5]
        
        fig_freight = px.box(
            freight_clean,
            x='review_score',
            y='freight_ratio',
            color='review_score',
            color_discrete_sequence=['#DC2626', '#EF4444', '#94A3B8', '#0284C7', '#0F766E']
        )
        fig_freight.update_layout(
            template='plotly_white',
            showlegend=False,
            height=320,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(title=dict(text="Review Star Rating", font=dict(size=11, color='#475569'))),
            yaxis=dict(title=dict(text="Freight Ratio (Freight / Total Order Value)", font=dict(size=11, color='#475569')), showgrid=True, gridcolor='#F1F5F9')
        )
        st.plotly_chart(fig_freight, use_container_width=True)
        
    with col_p4:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Dissatisfaction Rate by Payment Installment Tier</div>", unsafe_allow_html=True)
        tier_order = ['1 Installment (Single)', '2-3 Installments', '4-6 Installments', '7-10 Installments', '11-24 Installments']
        inst_agg = filtered_orders.groupby('installment_tier').agg(
            total=('order_id', 'count'),
            dissat=('dissatisfied', 'sum')
        ).reset_index()
        inst_agg['sort_key'] = inst_agg['installment_tier'].apply(lambda x: tier_order.index(x) if x in tier_order else 99)
        inst_agg = inst_agg.sort_values('sort_key')
        inst_agg['dissat_pct'] = (inst_agg['dissat'] / inst_agg['total'] * 100).round(2)
        
        fig_inst = px.bar(
            inst_agg,
            x='installment_tier',
            y='dissat_pct',
            text=inst_agg['dissat_pct'].apply(lambda v: f"{v:.1f}%"),
            color='dissat_pct',
            color_continuous_scale=['#94A3B8', '#DC2626']
        )
        fig_inst.update_traces(
            textposition='outside',
            textfont=dict(size=11, color='#1E293B')
        )
        fig_inst.update_layout(
            template='plotly_white',
            coloraxis_showscale=False,
            height=320,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(title=dict(text="Credit Installment Commitment", font=dict(size=11, color='#475569'))),
            yaxis=dict(title=dict(text="Dissatisfaction Rate (%)", font=dict(size=11, color='#475569')), showgrid=True, gridcolor='#F1F5F9')
        )
        st.plotly_chart(fig_inst, use_container_width=True)


# ==============================================================================
# PAGE 3: MACHINE LEARNING AND NLP
# ==============================================================================
with tab3:
    # Row 1: KPI Cards (4 Columns)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Selected Champion Model</div>
            <div class="kpi-value" style="font-size:20px;">HistGradientBoosting</div>
            <div class="kpi-subtext">Class-Balanced Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">PR-AUC (Average Precision)</div>
            <div class="kpi-value kpi-value-primary">0.5291</div>
            <div class="kpi-subtext">3.79x Over Random Baseline (0.1396)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Proactive Recall (Dissatisfied)</div>
            <div class="kpi-value kpi-value-success">56.15%</div>
            <div class="kpi-subtext">1,378 of 2,454 At-Risk Orders Caught</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">ROC-AUC Benchmark</div>
            <div class="kpi-value">0.7746</div>
            <div class="kpi-subtext">RF: 0.7661 | Logistic Reg: 0.7653</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Row 2: Model Comparison & Confusion Matrix
    col_m1, col_m2 = st.columns([3, 2])
    
    with col_m1:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Algorithm Performance Comparison (Test Cohort N = 17,582)</div>", unsafe_allow_html=True)
        melted_model = dim_model.melt(
            id_vars=['Model Name'],
            value_vars=['Precision (Dissatisfied)', 'Recall (Dissatisfied)', 'F1-Score (Dissatisfied)', 'ROC-AUC', 'PR-AUC (Avg Precision)'],
            var_name='Metric',
            value_name='Score'
        )
        
        fig_comp = px.bar(
            melted_model,
            x='Model Name',
            y='Score',
            color='Metric',
            barmode='group',
            color_discrete_sequence=['#64748B', '#0284C7', '#0F766E', '#1D4ED8', '#DC2626']
        )
        fig_comp.update_layout(
            template='plotly_white',
            height=320,
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
            xaxis=dict(title=""),
            yaxis=dict(title=dict(text="Evaluation Metric Score (0 to 1)", font=dict(size=11, color='#475569')), range=[0, 0.9], showgrid=True, gridcolor='#F1F5F9')
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
    with col_m2:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Champion Confusion Matrix (Exact Integer Counts)</div>", unsafe_allow_html=True)
        cm_matrix = np.array([[13187, 1941], [1076, 1378]])
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Customer Class", y="Actual Customer Class", color="Cases"),
            x=['Predicted Satisfied', 'Predicted Dissatisfied'],
            y=['True Satisfied', 'True Dissatisfied'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm.update_traces(
            textfont=dict(size=14, color='#0F172A')
        )
        fig_cm.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Row 3: Feature Importance & NLP Terms
    col_m3, col_m4 = st.columns(2)
    
    with col_m3:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Top 10 Permutation Feature Importance Rankings</div>", unsafe_allow_html=True)
        top_fi = dim_fi.head(10).sort_values('Importance Mean (ROC-AUC drop)', ascending=True)
        
        fig_fi = px.bar(
            top_fi,
            y='Feature',
            x='Importance Mean (ROC-AUC drop)',
            orientation='h',
            text=top_fi['Importance Mean (ROC-AUC drop)'].apply(lambda v: f"{v:.4f}"),
            color_discrete_sequence=['#0F766E']
        )
        fig_fi.update_traces(
            textposition='outside',
            textfont=dict(size=10, color='#1E293B')
        )
        fig_fi.update_layout(
            template='plotly_white',
            height=340,
            margin=dict(l=20, r=30, t=10, b=20),
            xaxis=dict(title=dict(text="ROC-AUC Degradation Upon Feature Permutation", font=dict(size=11, color='#475569')), showgrid=True, gridcolor='#F1F5F9'),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        
    with col_m4:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Portuguese Customer Feedback Terms (TF-IDF Salience)</div>", unsafe_allow_html=True)
        top_dissat = dim_nlp[dim_nlp['Cohort'].str.contains('Dissat')].head(6)
        top_sat = dim_nlp[dim_nlp['Cohort'].str.contains('Satisf')].head(6)
        top_nlp = pd.concat([top_dissat, top_sat])
        
        fig_nlp = px.bar(
            top_nlp,
            x='TF-IDF Sum Weight',
            y='N-Gram',
            color='Cohort',
            orientation='h',
            color_discrete_map={
                'Dissatisfied (Scores 1-2)': '#DC2626',
                'Satisfied (Scores 4-5)': '#0F766E'
            }
        )
        fig_nlp.update_layout(
            template='plotly_white',
            height=340,
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
            xaxis=dict(title=dict(text="Cumulative TF-IDF Corpus Weight", font=dict(size=11, color='#475569')), showgrid=True, gridcolor='#F1F5F9'),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_nlp, use_container_width=True)
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Row 4: Multimodal Summary Table & Strategy
    st.markdown("<div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:6px;'>Structured Operational Data vs. Unstructured Customer Text Synthesis</div>", unsafe_allow_html=True)
    st.dataframe(
        dim_svt[['Friction Dimension', 'Structured Data Findings', 'NLP Text Feedback Findings', 'Relationship (Support / Complement / Diverge)', 'Business Interpretation']],
        use_container_width=True,
        hide_index=True
    )
    
    # Concise Academic Insight & Recommendation Box
    st.markdown("""
    <div class="recommendation-card">
        <div class="recommendation-title">Key Insights & Practical Recommendation</div>
        <p class="recommendation-text">
            <b>1. Operational Root Cause:</b> Courier delivery delay beyond promised SLA is the single most decisive operational driver of customer dissatisfaction, surging negative ratings from 9.22% to 54.07% (a 5.86x escalation).<br>
            <b>2. Text Feedback Validation:</b> Customer comments triangulate logistics failure with terms like <i>'não recebi'</i> (did not receive) and mechanical product failure with terms like <i>'veio defeito'</i> (arrived defective).<br>
            <b>3. Practical Recommendation:</b> Implement an automated <b>Proactive Customer Retention Engine</b> triggered at delivery confirmation. When courier telemetry marks an order as delayed, automatically dispatch compensatory store credit (R$ 15–25) and route customer inquiries to priority tier-1 support before the customer submits a negative review.
        </p>
    </div>
    """, unsafe_allow_html=True)
