import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import shap
import os

# --- 1. Page Configuration & Theme ---
st.set_page_config(page_title="Enehano Lead Intelligence", layout="wide")

# Enehano Corporate Palette: Green: #a6ce39, Dark: #111418
st.markdown(f"""
    <style>
    /* Global Glassmorphism */
    .stApp {{
        background: radial-gradient(circle at top right, #1a1f25, #0e1117);
        color: #e0e0e0;
    }}
    
    /* Glass Cards */
    .glass-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
    }}
    
    /* Enehano Green Accents */
    .stButton>button {{
        background-color: #a6ce39 !important;
        color: #111418 !important;
        border-radius: 8px;
        font-weight: bold;
    }}
    
    /* Metrics Visibility Fix */
    [data-testid="stMetric"] {{
        background: rgba(166, 206, 57, 0.05);
        border: 1px solid rgba(166, 206, 57, 0.2);
        border-radius: 10px;
        padding: 10px;
    }}

    /* Score Banner Logic */
    .score-box {{
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 15px;
    }}
    .score-diff {{ font-size: 16px; opacity: 0.7; font-weight: 400; }}
    </style>
""", unsafe_allow_html=True)

# --- 2. Load Assets ---
@st.cache_resource
def load_ml_assets():
    try:
        return (joblib.load("model.pkl"), 
                joblib.load("preprocessor.pkl"), 
                joblib.load("feature_names.pkl"))
    except: return None, None, None

@st.cache_data
def load_directory():
    return pd.read_csv("lead_train.csv")

model, preprocessor, feature_names = load_ml_assets()
master_df = load_directory()

# --- 3. Session State for Navigation & Simulation ---
if 'view' not in st.session_state: st.session_state.view = 'directory'
if 'selected_ico' not in st.session_state: st.session_state.selected_ico = None
if 'original_score' not in st.session_state: st.session_state.original_score = None

# --- 4. Logic Functions ---
def get_business_impact(val):
    """Converts raw SHAP values to business labels."""
    if val > 0.15: return "🔥 Critical Driver"
    if val > 0.05: return "✅ Positive Influence"
    if val < -0.15: return "⚠️ Critical Risk"
    if val < -0.05: return "📉 Negative Factor"
    return "Neutral"

def reset_simulation():
    """Resets simulator sliders to default values."""
    if 'ttfr_val' in st.session_state: del st.session_state.ttfr_val
    if 'web_val' in st.session_state: del st.session_state.web_val
    st.rerun()

# --- 5. Global Top Header (Logo Centered) ---
st.columns([1, 1, 1])[1].image("enehano_logo.svg" if os.path.exists("enehano_logo.svg") else "https://www.enehano.cz/hubfs/Enehano_Logo_White.png", use_container_width=True)

search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
with search_col2:
    search_ico = st.text_input("🔍 Global IČO Search", placeholder="Lookup company by ID...", label_visibility="collapsed")
    if len(search_ico) == 8:
        st.session_state.selected_ico = search_ico
        st.session_state.view = 'profile'
        st.rerun()

st.divider()

# --- 6. View: Directory (Modern Card-Table) ---
if st.session_state.view == 'directory':
    st.subheader("Sales Pipeline Directory")
    
    f1, f2, f3 = st.columns(3)
    with f1: ind_f = st.multiselect("Industry Sector", master_df['CZ_NACE_Section'].unique())
    with f2: rat_f = st.multiselect("Lead Status", master_df['Rating'].unique())
    with f3: src_f = st.multiselect("Lead Source", master_df['LeadSource'].unique())

    df = master_df.copy()
    if ind_f: df = df[df['CZ_NACE_Section'].isin(ind_f)]
    if rat_f: df = df[df['Rating'].isin(rat_f)]
    if src_f: df = df[df['LeadSource'].isin(src_f)]

    # Modern Dataframe Selection
    selection = st.dataframe(
        df[['IČO', 'Company_Name', 'Legal_Form_Label', 'Rating', 'LeadSource']],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row" # Cleaner selection
    )

    if selection.selection.rows:
        st.session_state.selected_ico = df.iloc[selection.selection.rows[0]]['IČO']
        st.session_state.view = 'profile'
        st.session_state.original_score = None # Reset for new company
        st.rerun()

# --- 7. View: Profile & Simulation ---
elif st.session_state.view == 'profile':
    ico = st.session_state.selected_ico
    row = master_df[master_df['IČO'] == int(ico)].iloc[0]
    
    # Header for Profile
    p_head1, p_head2 = st.columns([3, 1])
    with p_head1:
        st.title(f"🏢 {row['Company_Name']}")
        st.caption(f"Registered IČO: {ico} | {row['Legal_Form_Label']}")
    with p_head2:
        if st.button("⬅ Back to Directory"):
            st.session_state.view = 'directory'
            st.rerun()

    # Layout: Sidebar Simulator | Main Content
    with st.sidebar:
        st.markdown("### 🧪 Simulation Panel")
        st.write("Adjust parameters to optimize the deal.")
        
        sim_ttfr = st.slider("Response Delay (hrs)", 0.1, 48.0, 2.0, key="ttfr_val")
        sim_web = st.slider("Website Activity", 0, 100, 45, key="web_val")
        sim_demo = st.checkbox("Demo Conducted", True)
        sim_linkedin = st.checkbox("Social Interaction", False)
        
        if st.button("🔄 Reset to Original"):
            reset_simulation()

    # --- Core Logic ---
    # Prepare Input
    input_row = pd.DataFrame([{
        "Legal_Form_Label": row['Legal_Form_Label'], "CZ_NACE_Section": row['CZ_NACE_Section'], 
        "Industry": "Technology", "NumberOfEmployees": 150, "Annual_Revenue_MCZK__c": 200,
        "Time_to_First_Response_h__c": sim_ttfr, "Web_Interactions__c": sim_web,
        "Email_Opens__c": 5, "Email_Clicks__c": 2, "Emails_Sent__c": 10,
        "Email_Open_Rate__c": 0.5, "Form_Submissions__c": 1, "Calls_Made__c": 2,
        "Meetings_Held__c": 0, "Content_Downloads__c": 1, "Chatbot_Interactions__c": 2,
        "Days_Since_Last_Activity__c": 1, "Days_in_Pipeline__c": 10, "Company_Age_Years": 10,
        "Demo_Requested__c": 1 if sim_demo else 0, "Proposal_Sent__c": 0,
        "LinkedIn_Viewed__c": 1 if sim_linkedin else 0, "LeadSource": row['LeadSource'],
        "Size_Band": "Medium", "Rating": "Warm"
    }])

    X_proc = preprocessor.transform(input_row)
    prob = model.predict_proba(X_proc)[0][1]
    current_score = int(prob * 100)

    # Store first calculated score as original reference
    if st.session_state.original_score is None:
        st.session_state.original_score = current_score

    # UI Content
    col_vis, col_expl = st.columns([1, 1.5], gap="large")
    
    with col_vis:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Score Banner with Comparison
        diff_text = f'<span class="score-diff">(was {st.session_state.original_score}%)</span>' if current_score != st.session_state.original_score else ""
        
        color = "#a6ce39" if current_score >= 70 else "#ffc107" if current_score >= 40 else "#ff4b4b"
        st.markdown(f"""
            <div style="background: {color}22; border: 2px solid {color};" class="score-box">
                AI CONVERSION SCORE<br>
                <span style="font-size: 54px; color: {color};">{current_score}%</span><br>
                {diff_text}
            </div>
        """, unsafe_allow_html=True)
        
        st.metric("Engagement Goal", f"{sim_web} visits", delta=f"{sim_web - 45} vs avg")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_expl:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Business Impact Analysis")
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_proc)
            raw_impacts = shap_vals[1].flatten() if isinstance(shap_vals, list) else shap_vals.flatten()
            
            min_len = min(len(raw_impacts), len(feature_names))
            shap_df = pd.DataFrame({"Feature": feature_names[:min_len], "Impact": raw_impacts[:min_len]}).sort_values("Impact", ascending=False)
            
            # Show top 5 in business terms
            for _, r in shap_df.head(5).iterrows():
                if abs(r['Impact']) > 0.02:
                    impact_label = get_business_impact(r['Impact'])
                    color = "#a6ce39" if r['Impact'] > 0 else "#ff4b4b"
                    st.write(f"**{r['Feature'].split('__')[0]}**")
                    st.caption(f"{impact_label}")
                    st.progress(min(abs(r['Impact']) * 2, 1.0))
        except:
            st.write("Generating business insights...")
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Enehano Solutions · Sales Intelligence Platform · 2026")