import streamlit as slt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report
import streamlit.components.v1 as _stc, json as _json, textwrap, hashlib

# ── Load external CSS (JS injection bypasses Streamlit 1.36+ DOMPurify) ────
# Spacing, prediction card alignment & tower size update v8
with open('style.css', encoding='utf-8-sig') as _f:
    _css = _f.read()
_css_hash = hashlib.md5(_css.encode('utf-8')).hexdigest()[:8]
_stc.html(
    f'<script>'
    f'/* hash: {_css_hash} */\n'
    f'var _el = parent.document.getElementById("custom-theme-style");'
    f'if (!_el) {{'
    f'    _el = parent.document.createElement("style");'
    f'    _el.id = "custom-theme-style";'
    f'    parent.document.head.appendChild(_el);'
    f'}}'
    f'_el.textContent = {_json.dumps(_css)};'
    f'</script>',
    height=0, scrolling=False
)
slt.markdown(f'<style>{_css}</style>', unsafe_allow_html=True)
del _stc, _json, _css, _css_hash

# ── Session State Defaults ──────────────────────────────────────────────────
defaults = {
    'PA': 'None', 'DE': 'None', 'TN': 0,
    'PH': 'None', 'ML': 'None', 'IN': 'None',
    'OS': 'None', 'OB': 'None', 'DP': 'None',
    'TS': 'None', 'STV': 'None', 'SM': 'None',
    'TE': 'None', 'PP': 'None',
    'PM': 'None', 'MC': 0.0, 'TC': 0.0,
    'result': None, 'proba': None
}
for k, v in defaults.items():
    if k not in slt.session_state:
        slt.session_state[k] = v

def reset_form():
    for k, v in defaults.items():
        slt.session_state[k] = v

# ── Train Model ─────────────────────────────────────────────────────────────
Telecom = pd.read_csv('Telco-Customer-Churn.csv')
Telecom['TotalCharges'] = pd.to_numeric(Telecom['TotalCharges'], errors='coerce')

group_cols = [col for col in Telecom.columns if col not in ['customerID', 'TotalCharges']]
Telecom['TotalCharges'] = Telecom['TotalCharges'].fillna(
    Telecom.groupby(group_cols)['TotalCharges'].transform('mean')
)
Telecom['TotalCharges'] = Telecom['TotalCharges'].fillna(Telecom['TotalCharges'].mean())

Telecom['Partner']          = Telecom['Partner'].map({'Yes': 1, 'No': 0})
Telecom['Dependents']       = Telecom['Dependents'].map({'Yes': 1, 'No': 0})
Telecom['PhoneService']     = Telecom['PhoneService'].map({'Yes': 1, 'No': 0})
Telecom['MultipleLines']    = Telecom['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 2})
Telecom['InternetService']  = Telecom['InternetService'].map({'Fiber optic': 1, 'DSL': 2, 'No': 0})
Telecom['OnlineSecurity']   = Telecom['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
Telecom['OnlineBackup']     = Telecom['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
Telecom['DeviceProtection'] = Telecom['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
Telecom['TechSupport']      = Telecom['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
Telecom['StreamingTV']      = Telecom['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
Telecom['StreamingMovies']  = Telecom['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
Telecom['Contract']         = Telecom['Contract'].map({'Month-to-month': 1, 'One year': 2, 'Two year': 3})
Telecom['PaperlessBilling'] = Telecom['PaperlessBilling'].map({'Yes': 1, 'No': 0})
Telecom['PaymentMethod']    = Telecom['PaymentMethod'].map({'Electronic check': 1, 'Mailed check': 2, 'Bank transfer (automatic)': 3, 'Credit card (automatic)': 4})
Telecom['Churn']            = Telecom['Churn'].map({'Yes': 1, 'No': 0})

Telecom['tenure']         = Telecom['tenure'] / 10
Telecom['MonthlyCharges'] = Telecom['MonthlyCharges'] / 10
Telecom['TotalCharges']   = Telecom['TotalCharges'] / 100

x = Telecom[['Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']]
y = Telecom['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
model = XGBClassifier(n_estimators=100, max_depth=3)
model.fit(x_train, y_train)


# ── Top Header with Gold Telecom Tower Icon ─────────────────────────────────
slt.markdown(textwrap.dedent("""
<div class="telecom-header">
    <div class="telecom-header-icon">
        <svg viewBox="0 0 24 24" width="44" height="44" fill="none" stroke="#F5C542" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2v20M4.93 4.93l14.14 14.14M19.07 4.93L4.93 19.07" stroke-opacity="0.35"/>
            <circle cx="12" cy="12" r="2" fill="#F5C542"/>
            <path d="M16.24 7.76a6 6 0 0 1 0 8.49M7.76 16.24a6 6 0 0 1 0-8.49"/>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 19.07a10 10 0 0 1 0-14.14"/>
        </svg>
    </div>
    <div class="telecom-title-box">
        <h1>Telecom <span class="highlight-green">Customer</span> <span class="highlight-gold">Churn Prediction</span></h1>
        <p class="telecom-subtitle">Fill in the customer details below, then press Predict.</p>
    </div>
</div>
"""), unsafe_allow_html=True)

# ── Main Two-Column Layout (Large Form Box Left + Stacked Sidebar Right) ───
form_col, right_col = slt.columns([2.25, 1.0])

with form_col:
    # ── Section 1: Customer (3 fields) ───────────────────────────────────────
    slt.markdown(textwrap.dedent("""
    <div class="section-header">
        <div class="badge-num">1</div>
        <div class="section-title">Customer</div>
        <div class="field-badge">3 fields</div>
    </div>
    """), unsafe_allow_html=True)

    c1, c2, c3 = slt.columns(3)
    PA = c1.selectbox("PARTNER",           ['None', 'Yes', 'No'],                                                key='PA')
    DE = c2.selectbox("DEPENDENTS",        ['None', 'Yes', 'No'],                                                key='DE')
    TN = c3.number_input("TENURE (MONTHS)", min_value=0,                                                         key='TN')

    # ── Section 2: Services (9 fields) ───────────────────────────────────────
    slt.markdown(textwrap.dedent("""
    <div class="section-header">
        <div class="badge-num">2</div>
        <div class="section-title">Services</div>
        <div class="field-badge">9 fields</div>
    </div>
    """), unsafe_allow_html=True)

    s1, s2, s3 = slt.columns(3)
    PH  = s1.selectbox("PHONESERVICE",     ['None', 'Yes', 'No'],                                                key='PH')
    ML  = s2.selectbox("MULTIPLELINES",    ['None', 'Yes', 'No', 'No phone service'],                            key='ML')
    IN  = s3.selectbox("INTERNETSERVICE",  ['None', 'Fiber optic', 'DSL', 'No'],                                 key='IN')

    s4, s5, s6 = slt.columns(3)
    OS  = s4.selectbox("ONLINESECURITY",   ['None', 'Yes', 'No', 'No internet service'],                         key='OS')
    OB  = s5.selectbox("ONLINEBACKUP",     ['None', 'Yes', 'No', 'No internet service'],                         key='OB')
    DP  = s6.selectbox("DEVICEPROTECTION", ['None', 'Yes', 'No', 'No internet service'],                         key='DP')

    s7, s8, s9 = slt.columns(3)
    TS  = s7.selectbox("TECHSUPPORT",      ['None', 'Yes', 'No', 'No internet service'],                         key='TS')
    STV = s8.selectbox("STREAMINGTV",      ['None', 'Yes', 'No', 'No internet service'],                         key='STV')
    SM  = s9.selectbox("STREAMINGMOVIES",  ['None', 'Yes', 'No', 'No internet service'],                         key='SM')

    # ── Section 3: Billing (5 fields) ────────────────────────────────────────
    slt.markdown(textwrap.dedent("""
    <div class="section-header">
        <div class="badge-num">3</div>
        <div class="section-title">Billing</div>
        <div class="field-badge">5 fields</div>
    </div>
    """), unsafe_allow_html=True)

    b1, b2, b3 = slt.columns(3)
    TE  = b1.selectbox("CONTRACT",         ['None', 'Month-to-month', 'One year', 'Two year'],                   key='TE')
    PP  = b2.selectbox("PAPERLESSBILLING", ['None', 'Yes', 'No'],                                                key='PP')
    PM  = b3.selectbox("PAYMENTMETHOD",    ['None', 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], key='PM')

    b4, _, b5 = slt.columns(3)
    MC  = b4.number_input("MONTHLY CHARGES", min_value=0.0,                                                      key='MC')
    TC  = b5.number_input("TOTAL CHARGES",   min_value=0.0,                                                      key='TC')

    # ── Buttons: Predict (Left Corner) & Reset (Right Corner) ───────────────
    b6, _, b7 = slt.columns(3)
    bt  = b6.button("🚀 Predict", type="primary", use_container_width=True)
    rst = b7.button("🔄 Reset",   type="secondary", use_container_width=True, on_click=reset_form)

# ── Standalone Right-Side Sidebar (Enlarged Tower Top + Prediction Card Directly Underneath) ──
with right_col:
    slt.markdown(textwrap.dedent("""
    <div class="right-stacked-sidebar">
        <div class="tower-standalone-wrapper">
            <svg viewBox="0 0 200 280" width="100%" height="510" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="100" cy="50" r="18" stroke="#00E676" stroke-width="1.8" stroke-dasharray="3 3" opacity="0.85"/>
                <circle cx="100" cy="50" r="32" stroke="#00E676" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.6"/>
                <circle cx="100" cy="46" r="46" stroke="#00E676" stroke-width="1.2" opacity="0.35"/>
                <circle cx="100" cy="50" r="60" stroke="#00E676" stroke-width="1" opacity="0.2"/>
                <line x1="100" y1="20" x2="100" y2="50" stroke="#F5C542" stroke-width="3" stroke-linecap="round"/>
                <circle cx="100" cy="20" r="4" fill="#00E676"/>
                <line x1="90" y1="50" x2="45" y2="250" stroke="#00E676" stroke-width="2.5"/>
                <line x1="110" y1="50" x2="155" y2="250" stroke="#00E676" stroke-width="2.5"/>
                <line x1="100" y1="50" x2="100" y2="250" stroke="#00C853" stroke-width="1.5" opacity="0.7"/>
                <line x1="83" y1="85" x2="117" y2="85" stroke="#F5C542" stroke-width="1.8"/>
                <line x1="75" y1="125" x2="125" y2="125" stroke="#F5C542" stroke-width="1.8"/>
                <line x1="65" y1="170" x2="135" y2="170" stroke="#F5C542" stroke-width="2"/>
                <line x1="55" y1="215" x2="145" y2="215" stroke="#F5C542" stroke-width="2"/>
                <line x1="83" y1="85" x2="125" y2="125" stroke="#00E676" stroke-width="1" opacity="0.7"/>
                <line x1="117" y1="85" x2="75" y2="125" stroke="#00E676" stroke-width="1" opacity="0.7"/>
                <line x1="75" y1="125" x2="135" y2="170" stroke="#00E676" stroke-width="1.2" opacity="0.7"/>
                <line x1="125" y1="125" x2="65" y2="170" stroke="#00E676" stroke-width="1.2" opacity="0.7"/>
                <line x1="65" y1="170" x2="145" y2="215" stroke="#00E676" stroke-width="1.4" opacity="0.7"/>
                <line x1="135" y1="170" x2="55" y2="215" stroke="#00E676" stroke-width="1.4" opacity="0.7"/>
                <ellipse cx="100" cy="255" rx="70" ry="12" fill="#00E676" opacity="0.15"/>
            </svg>
        </div>
        <div class="sidebar-info-card">
            <h4>Predict</h4>
            <div class="card-subtitle">Customer Churn</div>
            <div class="card-motto">Stay Connected<br>Build Better Business</div>
            <div class="sidebar-bar-chart">
                <div class="bar bar-gold bar-h-45"></div>
                <div class="bar bar-gold bar-h-75"></div>
                <div class="bar bar-green bar-h-60"></div>
                <div class="bar bar-green bar-h-95"></div>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

# ── Encode Inputs and Execute Prediction ────────────────────────────────────
try:
    if any(val == 'None' for val in [PA, DE, PH, ML, IN, OS, OB, DP, TS, STV, SM, TE, PP, PM]):
        _enc_ready = False
    else:
        PA_enc  = 1 if PA == 'Yes' else 0
        DE_enc  = 1 if DE == 'Yes' else 0
        TN_enc  = TN / 10
        PH_enc  = 1 if PH == 'Yes' else 0
        ML_enc  = {'Yes': 1, 'No': 0, 'No phone service': 2}[ML]
        IN_enc  = {'Fiber optic': 1, 'DSL': 2, 'No': 0}[IN]
        OS_enc  = {'Yes': 1, 'No': 0, 'No internet service': 2}[OS]
        OB_enc  = {'Yes': 1, 'No': 0, 'No internet service': 2}[OB]
        DP_enc  = {'Yes': 1, 'No': 0, 'No internet service': 2}[DP]
        TS_enc  = {'Yes': 1, 'No': 0, 'No internet service': 2}[TS]
        STV_enc = {'Yes': 1, 'No': 0, 'No internet service': 2}[STV]
        SM_enc  = {'Yes': 1, 'No': 0, 'No internet service': 2}[SM]
        TE_enc  = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}[TE]
        PP_enc  = 1 if PP == 'Yes' else 0
        PM_enc  = {'Electronic check': 1, 'Mailed check': 2, 'Bank transfer (automatic)': 3, 'Credit card (automatic)': 4}[PM]
        MC_enc  = MC / 10
        TC_enc  = TC / 100
        _enc_ready = True
except (KeyError, TypeError):
    _enc_ready = False

if bt and _enc_ready:
    input_features = [[PA_enc, DE_enc, TN_enc, PH_enc, ML_enc, IN_enc,
                      OS_enc, OB_enc, DP_enc, TS_enc, STV_enc, SM_enc,
                      TE_enc, PP_enc, PM_enc, MC_enc, TC_enc]]
    output = model.predict(input_features)
    slt.session_state['result'] = int(output[0])
    try:
        proba = model.predict_proba(input_features)
        slt.session_state['proba'] = float(proba[0][int(output[0])]) * 100
    except Exception:
        slt.session_state['proba'] = None
elif bt and not _enc_ready:
    slt.session_state['result'] = 'incomplete'

# ── Result Display ──────────────────────────────────────────────────────────
res = slt.session_state.get('result')
proba = slt.session_state.get('proba')
proba_txt = f" (Confidence: <strong>{proba:.1f}%</strong>)" if proba is not None else ""


if res == 1:
    slt.markdown(f"""
    <div class="churn-card-unhappy">
        <h3>⚠️ Customer Likely to Churn 😔</h3>
        <p>The machine learning model predicts this customer is at <strong>high risk of leaving</strong>{proba_txt}. Consider offering tailored contract discounts or retention perks.</p>
    </div>
    """, unsafe_allow_html=True)

elif res == 0:
    slt.markdown(f"""
    <div class="churn-card-happy">
        <h3>✅ Customer Likely to Stay 😊</h3>
        <p>The machine learning model predicts this customer is <strong>satisfied and likely to remain</strong>{proba_txt}. Keep up great service standards!</p>
    </div>
    """, unsafe_allow_html=True)

elif res == 'incomplete':
    slt.markdown("""
    <div class="churn-card-warning">
        <p>⚠️ <strong>Please fill in all fields before pressing Predict.</strong></p>
    </div>
    """, unsafe_allow_html=True)





    
    

    