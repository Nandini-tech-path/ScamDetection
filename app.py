"""
Streamlit Web Application for AI Scam Message Detection.
ScamShield AI — 3-Column Cybersecurity Dashboard.
"""

import streamlit as st
from src.predict import predict

# --- Page Configuration ---
st.set_page_config(
    page_title="ScamShield AI | Scam & Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Comprehensive CSS for 3-Column Dashboard ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Protect Material Icons */
    .material-icons, [class*="material-symbols"], [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }

    /* Typography – scoped, doesn't break icons */
    body, .stApp, p, h1, h2, h3, h4, h5, h6, li, span, button, input, textarea, label, div.stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Prevent horizontal overflow */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }

    /* Background */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }

    /* Hide Streamlit chrome */
    header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
    }
    #MainMenu, footer, .stDeployButton { display: none !important; }

    /* Main container: full width, tight padding */
    .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* === SHARED PANEL BASE === */
    .panel-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .panel-title {
        font-size: 0.68rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #f1f5f9;
    }

    /* === LEFT PANEL === */
    .left-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 0.65rem;
    }
    .left-brand-icon {
        width: 34px;
        height: 34px;
        background: #0f172a;
        color: #fff;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .left-brand-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .left-brand-sub {
        font-size: 0.68rem;
        color: #64748b;
    }

    .info-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.76rem;
        line-height: 1.4;
        color: #334155;
    }
    .info-item:last-child { margin-bottom: 0; }
    .info-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #0f172a;
        flex-shrink: 0;
        margin-top: 5px;
    }
    .info-label {
        font-weight: 600;
        color: #1e293b;
    }

    /* Detection Pipeline */
    .pipeline-wrap {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #334155;
        padding: 5px 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 5px;
        width: 100%;
    }
    .pipeline-icon {
        width: 20px;
        height: 20px;
        background: #0f172a;
        color: #fff;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.65rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .pipeline-arrow {
        font-size: 0.65rem;
        color: #94a3b8;
        margin-left: 14px;
        line-height: 1;
        padding: 1px 0;
    }

    /* === CENTER PANEL === */
    .center-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .center-brand-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .center-brand-icon {
        width: 38px;
        height: 38px;
        background: #0f172a;
        color: #fff;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    .center-brand-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .center-brand-sub {
        font-size: 0.76rem;
        color: #64748b;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background: #22c55e;
        border-radius: 50%;
    }

    /* Preset buttons */
    .section-label {
        font-size: 0.68rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    div.preset-btn-container button {
        background-color: #ffffff !important;
        color: #334155 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        transition: all 0.12s ease !important;
        height: 2rem !important;
    }
    div.preset-btn-container button:hover {
        background-color: #f8fafc !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }

    /* Textarea */
    [data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 7px !important;
        padding: 8px 10px !important;
        font-size: 0.88rem !important;
        line-height: 1.4 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        transition: border-color 0.12s ease !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #0f172a !important;
        box-shadow: 0 0 0 2px rgba(15,23,42,0.07) !important;
        outline: none !important;
    }
    [data-testid="stTextArea"] textarea::placeholder { color: #94a3b8 !important; }

    /* Action buttons */
    div.action-analyze button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: 1px solid #0f172a !important;
        border-radius: 6px !important;
        font-size: 0.83rem !important;
        font-weight: 600 !important;
        height: 2.1rem !important;
        box-shadow: 0 1px 2px rgba(15,23,42,0.1) !important;
        transition: background-color 0.12s ease !important;
    }
    div.action-analyze button:hover { background-color: #1e293b !important; }
    div.action-clear button {
        background-color: #ffffff !important;
        color: #475569 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        height: 2.1rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        transition: all 0.12s ease !important;
    }
    div.action-clear button:hover { background-color: #f8fafc !important; border-color: #94a3b8 !important; }

    /* Result banner */
    .result-banner-scam {
        background: #ffffff;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }
    .result-banner-safe {
        background: #ffffff;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }
    .result-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .result-title-group { display: flex; align-items: center; gap: 8px; }
    .result-badge-icon {
        width: 30px; height: 30px; border-radius: 6px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0;
    }
    .icon-scam { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .icon-safe { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
    .result-title-text { font-size: 1.05rem; font-weight: 700; margin: 0; }
    .title-scam { color: #991b1b; }
    .title-safe { color: #166534; }
    .result-desc-text { font-size: 0.75rem; color: #64748b; margin: 1px 0 0 0; }
    .result-score-group { display: flex; align-items: center; gap: 6px; }
    .score-badge { font-size: 0.78rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; }
    .score-badge-danger { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .score-badge-safe { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }

    .sev-pill {
        display: inline-block; font-size: 0.65rem; font-weight: 700;
        padding: 2px 6px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.04em;
    }
    .sev-critical { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .sev-high { background: #ffedd5; color: #c2410c; border: 1px solid #fed7aa; }
    .sev-medium { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .sev-low { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }

    /* Threat Meter */
    .risk-meter-wrapper { margin-top: 0.55rem; padding-top: 0.45rem; border-top: 1px solid #f1f5f9; }
    .risk-meter-header { display: flex; justify-content: space-between; font-size: 0.7rem; font-weight: 600; color: #475569; margin-bottom: 4px; }
    .custom-meter-track { width: 100%; height: 7px; background: #e2e8f0; border-radius: 9999px; overflow: hidden; margin: 3px 0; }
    .custom-meter-fill-scam { height: 100%; background: #dc2626; border-radius: 9999px; }
    .custom-meter-fill-safe { height: 100%; background: #16a34a; border-radius: 9999px; }
    .risk-meter-scale { display: flex; justify-content: space-between; font-size: 0.62rem; font-weight: 600; color: #94a3b8; margin-top: 2px; }

    /* Signals row */
    .result-signals-box {
        margin-top: 0.5rem; padding-top: 0.45rem; border-top: 1px solid #f1f5f9;
        display: flex; align-items: center; gap: 5px; flex-wrap: wrap;
    }
    .result-signals-label { font-size: 0.65rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; }
    .signal-chip { display: inline-flex; align-items: center; gap: 3px; font-size: 0.68rem; font-weight: 600; padding: 2px 6px; border-radius: 4px; }
    .signal-chip-danger { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .signal-chip-safe { background: #f0fdf4; color: #166634; border: 1px solid #bbf7d0; }

    /* White sub-panels */
    .white-panel { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 7px; padding: 0.75rem; margin-bottom: 0.6rem; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
    .panel-heading { font-size: 0.76rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.02em; }
    .message-highlight-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 5px; padding: 7px 9px; font-size: 0.82rem; line-height: 1.45; color: #334155; }

    /* Why-flagged list */
    .why-flagged-item { display: flex; align-items: flex-start; gap: 7px; font-size: 0.76rem; line-height: 1.4; color: #334155; margin-bottom: 5px; }
    .why-flagged-item:last-child { margin-bottom: 0; }
    .why-bullet-danger { color: #dc2626; font-weight: 700; line-height: 1.2; }
    .why-bullet-safe { color: #16a34a; font-weight: 700; line-height: 1.2; }
    .why-label { font-weight: 600; color: #0f172a; }

    /* Chips */
    .chips-wrapper { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 0.35rem; }
    .chip-item { display: inline-flex; align-items: center; gap: 3px; background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 1px 6px; border-radius: 3px; font-size: 0.7rem; font-weight: 500; }
    .chip-score { font-size: 0.62rem; color: #dc2626; font-weight: 700; background: rgba(220,38,38,0.08); padding: 1px 3px; border-radius: 2px; }

    /* === RIGHT PANEL === */
    .right-stat-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f1f5f9;
    }
    .right-stat-row:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
    .right-stat-label { font-size: 0.68rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
    .right-stat-value { font-size: 0.9rem; font-weight: 700; color: #0f172a; }
    .right-stat-value-danger { color: #dc2626; }
    .right-stat-value-safe { color: #16a34a; }
    .right-stat-value-muted { color: #64748b; font-size: 0.78rem; font-weight: 500; }

    .right-meter-track { width: 100%; height: 5px; background: #e2e8f0; border-radius: 9999px; overflow: hidden; margin: 3px 0; }
    .right-meter-fill-scam { height: 100%; background: #dc2626; border-radius: 9999px; }
    .right-meter-fill-safe { height: 100%; background: #16a34a; border-radius: 9999px; }

    .channel-row { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 3px; }
    .channel-tag { font-size: 0.68rem; font-weight: 600; padding: 2px 7px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; color: #475569; }

    .finding-row { background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid; border-radius: 0 5px 5px 0; padding: 6px 8px; margin-bottom: 5px; }
    .finding-row:last-child { margin-bottom: 0; }
    .finding-row-critical { border-left-color: #dc2626; }
    .finding-row-high { border-left-color: #ea580c; }
    .finding-row-medium { border-left-color: #d97706; }
    .finding-row-low { border-left-color: #16a34a; }
    .finding-title-text { font-size: 0.75rem; font-weight: 600; color: #1e293b; margin-left: 5px; }
    .finding-body-text { font-size: 0.7rem; color: #64748b; margin-top: 2px; line-height: 1.35; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Init ---
if "message_input" not in st.session_state:
    st.session_state["message_input"] = ""
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


def set_example_text(text: str):
    st.session_state["message_input"] = text
    st.session_state["last_result"] = None  # Reset result on new preset


def clear_input_text():
    st.session_state["message_input"] = ""
    st.session_state["last_result"] = None


# --- Preset messages ---
example_1 = "URGENT: Your SBI account is suspended due to incomplete KYC. Click http://bit.ly/sbi-kyc-update to update your PAN and Aadhaar immediately."
example_2 = "Electricity alert: Power supply will be disconnected tonight at 9:30 PM due to unpaid bill of Rs 4,850. Call electricity officer immediately at 9876543210."
example_3 = "Hi, your Swiggy delivery partner has picked up your food order #9021 and is arriving in 15 minutes."


# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────
def _signal_map(flag: dict):
    t = flag.get("title", "")
    d = flag.get("description", "")
    if "Urgency" in t:
        return ("Urgency", "Urgent language detected", "Creates pressure to act immediately.")
    elif "KYC" in t or "Identity" in t:
        return ("KYC Request", "KYC request detected", "Requests sensitive identity information.")
    elif "Account" in t or "Suspension" in t:
        return ("Account Threat", "Account threat detected", "Claims the account will be suspended.")
    elif "Link" in t or "URL" in t or "Shortener" in t:
        return ("Suspicious URL", "Suspicious URL detected", "Shortened link hides the real destination.")
    elif "OTP" in t or "Credential" in t:
        return ("OTP Request", "Credential demand detected", "Solicits private OTP or authentication pins.")
    elif "Advance-Fee" in t or "Financial" in t or "Bill" in t:
        return ("Financial Demand", "Payment coercion detected", "Demands immediate fee or bill clearance.")
    else:
        clean = t.replace(" Detection", "").replace(" Pattern", "").replace(" Threat", "")
        return (clean, t, d)


def _severity(risk_score_int, is_scam):
    if not is_scam:
        return "LOW", "sev-low"
    if risk_score_int >= 80:
        return "CRITICAL", "sev-critical"
    elif risk_score_int >= 50:
        return "HIGH", "sev-high"
    return "MEDIUM", "sev-medium"


# ───────────────────────────────────────────────
# 3-Column Layout
# ───────────────────────────────────────────────
col_left, col_center, col_right = st.columns([0.21, 0.58, 0.21], gap="small")

# ════════════════════════════════════════════
#  LEFT PANEL — Security Control Panel
# ════════════════════════════════════════════
with col_left:
    # Brand mark
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:0.75rem;margin-bottom:0.65rem;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
            <div class="left-brand">
                <div class="left-brand-icon">🛡</div>
                <div>
                    <div class="left-brand-name">ScamShield AI</div>
                    <div class="left-brand-sub">Security Control Panel</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # How It Works
    st.markdown(
        """
        <div class="panel-box">
            <div class="panel-title">How It Works</div>
            <div class="info-item">
                <div class="info-dot"></div>
                <div><span class="info-label">Message Analysis</span> — NLP bi-grams &amp; linguistic pattern scoring.</div>
            </div>
            <div class="info-item">
                <div class="info-dot"></div>
                <div><span class="info-label">Heuristic Engine</span> — Detects panic urgency, shorteners &amp; coercion.</div>
            </div>
            <div class="info-item">
                <div class="info-dot"></div>
                <div><span class="info-label">Risk Scoring</span> — Calibrated Bayesian threat probability.</div>
            </div>
            <div class="info-item">
                <div class="info-dot"></div>
                <div><span class="info-label">Explainable Signals</span> — Token attribution &amp; highlighted triggers.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Architecture
    st.markdown(
        """
        <div class="panel-box">
            <div class="panel-title">Model Architecture</div>
            <div class="info-item"><div class="info-dot"></div><div><span class="info-label">Pipeline:</span> Hybrid TF-IDF (1-2 grams) + Security Features</div></div>
            <div class="info-item"><div class="info-dot"></div><div><span class="info-label">Classifier:</span> Cost-Tuned Regularized Linear Model</div></div>
            <div class="info-item"><div class="info-dot"></div><div><span class="info-label">Dataset:</span> SMS Spam Collection + Phishing Corpus</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detection Pipeline
    st.markdown(
        """
        <div class="panel-box">
            <div class="panel-title">Detection Pipeline</div>
            <div class="pipeline-wrap">
                <div class="pipeline-step"><div class="pipeline-icon">①</div> Input Message</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step"><div class="pipeline-icon">②</div> NLP Analysis</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step"><div class="pipeline-icon">③</div> Security Features</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step"><div class="pipeline-icon">④</div> Risk Scoring</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step"><div class="pipeline-icon">⑤</div> Final Verdict</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Footer note
    st.markdown(
        """
        <div style="font-size:0.68rem;color:#94a3b8;line-height:1.4;padding:0.4rem 0.2rem;">
            Educational evaluation — BTech AI PBL.<br>Always verify via official banking channels.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════
#  CENTER PANEL — Analysis Workspace
# ════════════════════════════════════════════
with col_center:
    # Header
    st.markdown(
        """
        <div class="center-header">
            <div class="center-brand-left">
                <div class="center-brand-icon">🛡</div>
                <div>
                    <div class="center-brand-title">ScamShield AI</div>
                    <div class="center-brand-sub">Real-time Scam &amp; Phishing Threat Intelligence</div>
                </div>
            </div>
            <div class="status-badge">
                <span class="status-dot"></span>
                Detection Engine Active
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Try Presets
    st.markdown('<div class="section-label">Quick Try Presets</div>', unsafe_allow_html=True)
    ce1, ce2, ce3 = st.columns(3)
    with ce1:
        st.markdown('<div class="preset-btn-container">', unsafe_allow_html=True)
        st.button("KYC Phishing", on_click=set_example_text, args=(example_1,), use_container_width=True, key="btn_kyc")
        st.markdown('</div>', unsafe_allow_html=True)
    with ce2:
        st.markdown('<div class="preset-btn-container">', unsafe_allow_html=True)
        st.button("Bill Threat", on_click=set_example_text, args=(example_2,), use_container_width=True, key="btn_bill")
        st.markdown('</div>', unsafe_allow_html=True)
    with ce3:
        st.markdown('<div class="preset-btn-container">', unsafe_allow_html=True)
        st.button("Delivery Alert", on_click=set_example_text, args=(example_3,), use_container_width=True, key="btn_delivery")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:0.4rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Message to Analyze</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        label="Message",
        label_visibility="collapsed",
        key="message_input",
        height=90,
        placeholder="Paste an SMS, WhatsApp, or email alert here...",
    )

    ca1, ca2, _ = st.columns([1.3, 0.85, 4.5])
    with ca1:
        st.markdown('<div class="action-analyze">', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze Message", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ca2:
        st.markdown('<div class="action-clear">', unsafe_allow_html=True)
        st.button("Clear", on_click=clear_input_text, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Run Analysis & Persist to Session State ---
    if user_input and user_input.strip():
        if st.session_state.get("last_result") is None or st.session_state.get("last_analyzed_text") != user_input.strip():
            with st.spinner("Analyzing message..."):
                result = predict(user_input)
            st.session_state["last_result"] = result
            st.session_state["last_analyzed_text"] = user_input.strip()
    else:
        st.session_state["last_result"] = None

    result = st.session_state.get("last_result")

    if result:
        prob_val = result["scam_probability"] * 100.0
        risk_int = int(round(prob_val))
        severity_label, severity_pill = _severity(risk_int, result["is_scam"])

        # Build signals
        raw_signals, why_items = [], []
        for flag in result.get("red_flags", []):
            chip, why_title, why_desc = _signal_map(flag)
            if chip not in [s[0] for s in raw_signals]:
                raw_signals.append((chip, flag.get("severity", "HIGH").upper()))
                why_items.append((why_title, why_desc))

        if result["is_scam"]:
            banner_cls = "result-banner-scam"
            icon_cls = "icon-scam"
            title_cls = "title-scam"
            badge_cls = "score-badge-danger"
            title_txt = "SCAM DETECTED"
            icon_sym = "🛡"
            desc_txt = "This message exhibits characteristics of fraudulent social engineering or phishing."
            meter_cls = "custom-meter-fill-scam"
            bullet_cls = "why-bullet-danger"
        else:
            banner_cls = "result-banner-safe"
            icon_cls = "icon-safe"
            title_cls = "title-safe"
            badge_cls = "score-badge-safe"
            title_txt = "MESSAGE APPEARS SAFE"
            icon_sym = "✓"
            desc_txt = "No strong malicious indicators, credential harvesting, or coercion patterns were identified."
            meter_cls = "custom-meter-fill-safe"
            bullet_cls = "why-bullet-safe"
            if not why_items:
                why_items = [("Clean communication pattern", "No urgency coercion, masked domains, or identity demands identified.")]

        # Signals chips html
        signals_html = ""
        if raw_signals:
            for chip, _ in raw_signals:
                signals_html += f'<span class="signal-chip signal-chip-danger">{chip}</span>'
        else:
            signals_html = '<span class="signal-chip signal-chip-safe">✓ No Threat Indicators</span>'

        # Result banner
        st.markdown(
            f"""
            <div class="{banner_cls}">
                <div class="result-header-row">
                    <div class="result-title-group">
                        <div class="result-badge-icon {icon_cls}">{icon_sym}</div>
                        <div>
                            <div class="result-title-text {title_cls}">{title_txt}</div>
                            <div class="result-desc-text">{desc_txt}</div>
                        </div>
                    </div>
                    <div class="result-score-group">
                        <span class="score-badge {badge_cls}">Risk Score: {risk_int}/100</span>
                        <span class="sev-pill {severity_pill}">{severity_label}</span>
                    </div>
                </div>
                <div class="risk-meter-wrapper">
                    <div class="risk-meter-header">
                        <span>Threat Meter Index — {risk_int}/100 ({severity_label})</span>
                        <span>Model Confidence: {result['confidence_score']:.1f}%</span>
                    </div>
                    <div class="custom-meter-track">
                        <div class="{meter_cls}" style="width:{risk_int}%;"></div>
                    </div>
                    <div class="risk-meter-scale">
                        <span>0 LOW</span><span>25 MED</span><span>50 HIGH</span><span>75+ CRIT</span>
                    </div>
                </div>
                <div class="result-signals-box">
                    <span class="result-signals-label">Detected Signals:</span>
                    {signals_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Two sub-panels
        rc1, rc2 = st.columns([1, 1], gap="small")

        with rc1:
            # Why flagged
            flagged_rows = "".join(
                f'<div class="why-flagged-item"><span class="{bullet_cls}">•</span>'
                f'<div><span class="why-label">{t}</span> — {d}</div></div>'
                for t, d in why_items
            )
            st.markdown(
                f"""
                <div class="white-panel">
                    <div class="panel-heading">Why this message was {"flagged" if result["is_scam"] else "safe"}</div>
                    {flagged_rows}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with rc2:
            # Message attribution
            tokens_html = ""
            if result.get("top_suspicious_tokens"):
                tokens_html = '<div class="chips-wrapper">' + "".join(
                    f'<span class="chip-item">{tok} <span class="chip-score">+{sc:.2f}</span></span>'
                    for tok, sc in result["top_suspicious_tokens"]
                ) + "</div>"
            st.markdown(
                f"""
                <div class="white-panel">
                    <div class="panel-heading">Message Attribution Analysis</div>
                    <div class="message-highlight-box">{result["highlighted_html"]}</div>
                    {tokens_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════
#  RIGHT PANEL — Live Analysis
# ════════════════════════════════════════════
with col_right:
    result = st.session_state.get("last_result")

    if result:
        prob_val = result["scam_probability"] * 100.0
        risk_int = int(round(prob_val))
        severity_label, severity_pill = _severity(risk_int, result["is_scam"])
        conf = result["confidence_score"]
        is_scam = result["is_scam"]
        meter_fill = "right-meter-fill-scam" if is_scam else "right-meter-fill-safe"
        val_class = "right-stat-value-danger" if is_scam else "right-stat-value-safe"
        status_label = "SCAM DETECTED" if is_scam else "MESSAGE SAFE"
        status_style = "color:#991b1b;font-weight:700;" if is_scam else "color:#166534;font-weight:700;"

        # Build signals for right panel
        signal_chips_right = []
        for flag in result.get("red_flags", []):
            chip, _, _ = _signal_map(flag)
            if chip not in signal_chips_right:
                signal_chips_right.append(chip)

        signals_right_html = ""
        if signal_chips_right:
            for chip in signal_chips_right:
                signals_right_html += f'<span class="signal-chip signal-chip-danger">{chip}</span>'
        else:
            signals_right_html = '<span class="signal-chip signal-chip-safe">✓ None</span>'

        # Verdict
        st.markdown(
            f"""
            <div class="panel-box" style="border-left:4px solid {'#dc2626' if is_scam else '#16a34a'};">
                <div class="panel-title">Live Analysis Result</div>
                <div style="font-size:0.85rem;{status_style}margin-bottom:4px;">{status_label}</div>
                <div class="right-stat-row">
                    <span class="right-stat-label">Risk Score</span>
                    <span class="right-stat-value {val_class}">{risk_int}/100</span>
                </div>
                <div class="right-meter-track">
                    <div class="{meter_fill}" style="width:{risk_int}%;"></div>
                </div>
                <div class="right-stat-row" style="margin-top:6px;">
                    <span class="right-stat-label">Severity</span>
                    <span class="sev-pill {severity_pill}">{severity_label}</span>
                </div>
                <div class="right-stat-row">
                    <span class="right-stat-label">Model Confidence</span>
                    <span class="right-stat-value">{conf:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Detected Signals
        st.markdown(
            f"""
            <div class="panel-box">
                <div class="panel-title">Detected Signals</div>
                <div style="display:flex;flex-wrap:wrap;gap:5px;">{signals_right_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Red Flags
        if result.get("red_flags"):
            flags_html = ""
            for flag in result["red_flags"]:
                sev = flag.get("severity", "HIGH").upper()
                flags_html += (
                    f'<div class="finding-row finding-row-{sev.lower()}">'
                    f'<div><span class="sev-pill sev-{sev.lower()}">{sev}</span>'
                    f'<span class="finding-title-text">{flag["title"]}</span></div>'
                    f'<div class="finding-body-text">{flag["description"]}</div></div>'
                )
            st.markdown(
                f"""
                <div class="panel-box">
                    <div class="panel-title">Heuristic Findings</div>
                    {flags_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        # Idle state — awaiting analysis
        st.markdown(
            """
            <div class="panel-box">
                <div class="panel-title">Live Analysis</div>
                <div style="color:#94a3b8;font-size:0.78rem;line-height:1.5;text-align:center;padding:1rem 0.5rem;">
                    <div style="font-size:1.6rem;margin-bottom:8px;">🛡</div>
                    Awaiting message analysis.<br>
                    Enter or paste a message in the center panel and click <strong>Analyze Message</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="panel-box">
                <div class="panel-title">Analysis Status</div>
                <div class="right-stat-row">
                    <span class="right-stat-label">Engine</span>
                    <span class="status-badge"><span class="status-dot"></span>Active</span>
                </div>
                <div class="right-stat-row">
                    <span class="right-stat-label">Risk Score</span>
                    <span class="right-stat-value-muted">—</span>
                </div>
                <div class="right-stat-row">
                    <span class="right-stat-label">Severity</span>
                    <span class="right-stat-value-muted">—</span>
                </div>
                <div class="right-stat-row">
                    <span class="right-stat-label">Confidence</span>
                    <span class="right-stat-value-muted">—</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="panel-box">
                <div class="panel-title">Supported Channels</div>
                <div class="channel-row">
                    <span class="channel-tag">SMS</span>
                    <span class="channel-tag">WhatsApp</span>
                    <span class="channel-tag">Email</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
