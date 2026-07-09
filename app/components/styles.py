from __future__ import annotations

import streamlit as st


def apply_enterprise_theme() -> None:
    """Apply the shared WorkLens visual language."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700&display=swap');
        :root {
            --ink: #12211c;
            --muted: #62716b;
            --paper: #f5f7f4;
            --card: #ffffff;
            --accent: #19a974;
            --accent-soft: #dff6ec;
            --line: #dde5e0;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
        h1, h2, h3 { font-family: "Manrope", sans-serif !important; letter-spacing: -0.03em; }
        h1 { font-size: clamp(2.4rem, 5vw, 4.8rem) !important; max-width: 850px; }
        [data-testid="stSidebar"] { background: #10231c; }
        [data-testid="stSidebar"] * { color: #e8f2ed !important; }
        .brand-mark {
            width: 38px; height: 38px; display: grid; place-items: center;
            border-radius: 11px; background: #2bd39a; color: #10231c !important;
            font: 700 20px "Manrope";
        }
        .eyebrow {
            color: #16845d; font-size: .75rem; font-weight: 700;
            letter-spacing: .14em; margin-top: 1rem;
        }
        .feature-card {
            min-height: 190px; background: var(--card); border: 1px solid var(--line);
            border-radius: 18px; padding: 24px; margin: 20px 0;
            box-shadow: 0 8px 30px rgba(29, 55, 45, .05);
        }
        .feature-card span { color: var(--accent); font: 700 .78rem "Manrope"; }
        .feature-card h3 { margin: 26px 0 8px; }
        .feature-card p { color: var(--muted); line-height: 1.55; }
        .insight-box {
            background: #10231c; color: #eff8f3; border-radius: 18px;
            padding: 22px 24px; margin: 12px 0 20px;
        }
        .insight-box small {
            color: #67d9ae; font-size: .72rem; font-weight: 700; letter-spacing: .12em;
        }
        .insight-box p { margin: 10px 0 0; line-height: 1.55; }
        .risk-critical, .risk-high, .risk-medium, .risk-low {
            border-radius: 999px; padding: 4px 10px; font-size: .76rem; font-weight: 700;
        }
        .risk-critical { background: #fee2e2; color: #b42318; }
        .risk-high { background: #ffedd5; color: #c2410c; }
        .risk-medium { background: #fef3c7; color: #a16207; }
        .risk-low { background: #dcfce7; color: #15803d; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; }
        [data-testid="stPlotlyChart"] {
            background: white; border: 1px solid var(--line);
            border-radius: 16px; padding: 5px;
        }
        div[data-testid="stMetric"] {
            background: var(--card); border: 1px solid var(--line);
            border-radius: 16px; padding: 18px;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"] * {
            color: var(--ink) !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.9rem !important;
        }
        .stButton > button {
            border-radius: 10px; border: 0; background: var(--ink); color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
