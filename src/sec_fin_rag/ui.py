# ruff: noqa: E501
from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from .config import get_settings
from .manifest import load_issuers
from .pipeline import load_scorecards

if TYPE_CHECKING:
    from .runtime import Runtime

COLORS = {
    "background": "#f4f6f8",
    "surface": "#ffffff",
    "surface_alt": "#eef2f5",
    "border": "#dce2e8",
    "text": "#172033",
    "muted": "#667085",
    "primary": "#0f766e",
    "cobalt": "#2457d6",
    "positive": "#15803d",
    "negative": "#c2414b",
    "nav": "#111827",
}

METRIC_LABELS = {
    "revenue": "Revenue",
    "net_income": "Net income",
    "operating_cash_flow": "Operating cash flow",
    "free_cash_flow": "Free cash flow",
    "assets": "Total assets",
    "liabilities": "Total liabilities",
    "equity": "Shareholders’ equity",
    "eps_diluted": "Diluted EPS",
    "operating_income": "Operating income",
    "revenue_cagr_3y": "3-year revenue CAGR",
    "revenue_growth_yoy": "Revenue growth",
    "eps_growth": "EPS growth",
    "operating_margin": "Operating margin",
    "net_margin": "Net margin",
    "return_on_assets": "Return on assets",
    "fcf_margin": "Free-cash-flow margin",
    "cash_to_debt": "Cash / debt",
    "inverse_debt_to_assets": "Inverse debt / assets",
    "current_ratio": "Current ratio",
    "interest_coverage": "Interest coverage",
}


def _top_navigation() -> None:
    with st.container(key="finai_nav"):
        columns = st.columns([2.2, 1, 1.25, 1, 1.3, 1.15], vertical_alignment="center")
        with columns[0]:
            st.markdown(
                """<div class="brand-lockup"><div class="brand-mark">F</div><div>
                <div class="brand-name">Fin AI</div>
                <div class="brand-tagline">SEC INTELLIGENCE</div></div></div>""",
                unsafe_allow_html=True,
            )
        links = [
            ("streamlit_app.py", "Dashboard"),
            ("pages/1_Company_Intelligence.py", "Companies"),
            ("pages/2_Peer_Analysis.py", "Peers"),
            ("pages/3_Research_Assistant.py", "Ask Fin AI"),
        ]
        for column, (target, label) in zip(columns[1:5], links, strict=True):
            with column:
                st.page_link(target, label=label, width="stretch")
        with columns[5]:
            st.page_link("pages/4_About_and_Data.py", label="About", width="stretch")
            st.markdown(
                '<div class="nav-status"><span></span> SEC snapshot</div>', unsafe_allow_html=True
            )


def page_setup(title: str, description: str = "", *, eyebrow: str = "FIN AI RESEARCH") -> None:
    st.set_page_config(page_title=f"{title} | Fin AI", page_icon="📈", layout="wide")
    st.markdown(
        f"""<style>
        :root {{ color-scheme:light; }}
        .stApp {{ background:{COLORS["background"]};color:{COLORS["text"]}; }}
        .block-container {{ max-width:1440px;padding-top:1rem;padding-bottom:4rem; }}
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none; }}
        [data-testid="stHeader"] {{ background:transparent; }}
        #MainMenu, footer, [data-testid="stToolbar"] {{ visibility:hidden; }}
        .st-key-finai_nav {{ background:{COLORS["nav"]};border:1px solid #263244;border-radius:14px;padding:.55rem .75rem;margin-bottom:2.2rem;box-shadow:0 12px 30px rgba(15,23,42,.12); }}
        .st-key-finai_nav [data-testid="stPageLink"] a {{ color:#e5edf7 !important;background:transparent;border-radius:8px;text-align:center;font-size:.79rem;font-weight:650;padding:.55rem .45rem;justify-content:center; }}
        .st-key-finai_nav [data-testid="stPageLink"] a * {{ color:#e5edf7 !important; }}
        .st-key-finai_nav [data-testid="stPageLink"] a:hover {{ color:#fff !important;background:#263244;text-decoration:none; }}
        .st-key-finai_nav [data-testid="stPageLink"] a:hover * {{ color:#fff !important; }}
        .brand-lockup {{ display:flex;align-items:center;gap:.7rem;min-height:2.4rem; }}
        .brand-mark {{ width:2.25rem;height:2.25rem;display:grid;place-items:center;border-radius:8px;color:#fff;font-weight:850;font-size:1.2rem;background:{COLORS["primary"]};box-shadow:inset 0 0 0 1px rgba(255,255,255,.18); }}
        .brand-name {{ color:#fff;font-size:1.1rem;font-weight:780;letter-spacing:-.025em; }}
        .brand-tagline {{ color:#8fa0b6;font-size:.57rem;letter-spacing:.13em;font-weight:650; }}
        .nav-status {{ color:#91a0b4;font-size:.58rem;text-align:center;margin-top:-.3rem; }}
        .nav-status span {{ display:inline-block;width:.4rem;height:.4rem;border-radius:50%;background:#34d399;margin-right:.25rem; }}
        .page-eyebrow {{ color:{COLORS["primary"]};font-weight:760;font-size:.68rem;letter-spacing:.16em;margin-bottom:.5rem; }}
        .page-title {{ color:{COLORS["text"]};font-size:2.3rem;font-weight:780;letter-spacing:-.045em;line-height:1.08;margin:0;max-width:900px; }}
        .page-description {{ color:{COLORS["muted"]};font-size:1rem;max-width:780px;line-height:1.6;margin:.7rem 0 1.8rem; }}
        .metric-card {{ min-height:118px;padding:1.05rem 1.1rem;border-radius:12px;background:{COLORS["surface"]};border:1px solid {COLORS["border"]};box-shadow:0 5px 16px rgba(23,32,51,.045); }}
        .metric-label {{ color:{COLORS["muted"]};font-size:.69rem;font-weight:720;letter-spacing:.055em;text-transform:uppercase; }}
        .metric-value {{ color:{COLORS["text"]};font-size:1.55rem;font-weight:770;letter-spacing:-.03em;margin:.36rem 0 .2rem;font-variant-numeric:tabular-nums; }}
        .metric-detail {{ color:#7b8797;font-size:.73rem;line-height:1.35; }}
        .section-kicker {{ color:{COLORS["text"]};font-size:1.18rem;font-weight:740;margin:1.8rem 0 .2rem;letter-spacing:-.015em; }}
        .section-copy {{ color:{COLORS["muted"]};font-size:.86rem;margin-bottom:.95rem;max-width:900px; }}
        .pill {{ display:inline-block;border:1px solid {COLORS["border"]};border-radius:99px;padding:.27rem .6rem;color:{COLORS["muted"]};background:#fff;font-size:.69rem;font-weight:670;margin-right:.35rem; }}
        .pill.accent {{ color:#0b5f59;border-color:#9ed8d1;background:#e9f7f5; }}
        [data-testid="stMetric"] {{ background:{COLORS["surface"]};border:1px solid {COLORS["border"]};padding:15px 17px;border-radius:12px;box-shadow:0 5px 16px rgba(23,32,51,.04); }}
        [data-testid="stDataFrame"] {{ background:#fff;border:1px solid #cbd5df;border-radius:11px;overflow:hidden;box-shadow:0 4px 14px rgba(23,32,51,.035); }}
        [data-baseweb="select"] > div {{ background:#fff !important;border-color:#b9c4d0 !important; }}
        [data-baseweb="select"] span, [data-baseweb="select"] input {{ color:{COLORS["text"]} !important; }}
        [data-baseweb="popover"], [role="listbox"] {{ background:#fff !important;color:{COLORS["text"]} !important; }}
        [role="option"] {{ color:{COLORS["text"]} !important; }}
        [role="option"]:hover {{ background:#e7edf3 !important; }}
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {{ background:#dceeea !important;color:#0b5f59 !important; }}
        [data-testid="stExpander"] {{ background:#fff;border-color:#cbd5df !important;border-radius:10px; }}
        .stTabs [data-baseweb="tab-list"] {{ gap:.3rem;border-bottom:1px solid {COLORS["border"]}; }}
        .stTabs [data-baseweb="tab"] {{ color:#475467 !important;border-radius:8px 8px 0 0;padding:.58rem .95rem;font-weight:650; }}
        .stTabs [aria-selected="true"] {{ color:{COLORS["primary"]} !important;background:#e9f4f2 !important; }}
        .stButton>button, [data-testid="stLinkButton"] a {{ border-radius:8px;font-weight:680; }}
        [data-testid="stChatMessage"] {{ background:#fff;border:1px solid {COLORS["border"]};border-radius:12px;padding:.35rem .75rem;box-shadow:0 4px 14px rgba(23,32,51,.035); }}
        .answer-meta {{ color:#7b8797;font-size:.7rem;border-top:1px solid {COLORS["border"]};margin-top:.8rem;padding-top:.55rem; }}
        .ticker-shell {{ overflow:hidden;border:1px solid #263244;border-radius:9px;background:{COLORS["nav"]};margin:-.2rem 0 .45rem;white-space:nowrap;box-shadow:0 5px 16px rgba(15,23,42,.08); }}
        .ticker-track {{ display:inline-flex;width:max-content;animation:ticker-scroll 42s linear infinite; }}
        .ticker-shell:hover .ticker-track {{ animation-play-state:paused; }}
        .ticker-item {{ display:inline-flex;align-items:center;gap:.5rem;padding:.62rem 1.1rem;border-right:1px solid #2c394c;font-size:.77rem; }}
        .ticker-symbol {{ color:#fff;font-weight:760;letter-spacing:.045em; }}
        .ticker-price {{ color:#c8d2df;font-variant-numeric:tabular-nums; }}
        .ticker-up {{ color:#4ade80;font-weight:700; }}
        .ticker-down {{ color:#fb7185;font-weight:700; }}
        @keyframes ticker-scroll {{ from{{transform:translateX(0)}}to{{transform:translateX(-50%)}} }}
        @media(max-width:900px) {{ .block-container{{padding-top:.5rem}}.page-title{{font-size:1.75rem}}.brand-tagline,.nav-status{{display:none}}.st-key-finai_nav{{padding:.4rem}} }}
        </style>""",
        unsafe_allow_html=True,
    )
    _top_navigation()
    st.markdown(
        f'<div class="page-eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<div class="page-description">{html.escape(description)}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="section-kicker">{html.escape(title)}</div>'
        f'<div class="section-copy">{html.escape(description)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-detail">{html.escape(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def pill(text: str, *, accent: bool = False) -> str:
    return f'<span class="pill{" accent" if accent else ""}">{html.escape(text)}</span>'


def market_ticker(rows: list[dict[str, Any]]) -> None:
    available = [row for row in rows if row.get("status") == "available"]
    if not available:
        return
    items = []
    for row in available:
        change = row.get("change_percent")
        direction = "ticker-up" if change is not None and float(change) >= 0 else "ticker-down"
        arrow = "▲" if change is not None and float(change) >= 0 else "▼"
        change_text = f"{arrow} {abs(float(change)):.2f}%" if change is not None else "—"
        items.append(
            '<span class="ticker-item">'
            f'<span class="ticker-symbol">{html.escape(str(row["ticker"]))}</span>'
            f'<span class="ticker-price">${float(row["price"]):,.2f}</span>'
            f'<span class="{direction}">{change_text}</span></span>'
        )
    tape = "".join(items)
    st.markdown(
        f'<div class="ticker-shell"><div class="ticker-track">{tape}{tape}</div></div>',
        unsafe_allow_html=True,
    )


def issuers() -> list[Any]:
    return load_issuers(get_settings().manifest_path)


@st.cache_resource(show_spinner="Connecting to research services…")  # type: ignore[untyped-decorator]
def runtime() -> Runtime:
    from .runtime import Runtime

    return Runtime(get_settings())


@st.cache_data(ttl=900)  # type: ignore[untyped-decorator]
def scorecards() -> pd.DataFrame:
    return load_scorecards(get_settings())


def artifact_notice() -> None:
    st.info("Research artifacts are not available. Build and index the offline dataset first.")


def metric_label(value: str) -> str:
    return METRIC_LABELS.get(value, value.replace("_", " ").title())


def format_large(value: float | None, unit: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "—"
    prefix = "$" if unit == "USD" else ""
    for suffix, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(value) >= divisor:
            return f"{prefix}{value / divisor:,.2f}{suffix}"
    return f"{prefix}{value:,.2f}"


def format_percent(value: float | None, *, already_percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = value if already_percent else value * 100
    return f"{number:+.1f}%"


def plotly_layout(figure: Any, *, height: int = 430) -> Any:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["muted"], "family": "Inter, sans-serif"},
        title_font={"color": COLORS["text"], "size": 16},
        legend_title_text="",
        margin={"l": 25, "r": 20, "t": 55, "b": 35},
        hoverlabel={"bgcolor": COLORS["nav"], "font_color": "#ffffff"},
    )
    figure.update_xaxes(gridcolor="rgba(102,112,133,.14)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(102,112,133,.14)", zeroline=False)
    return figure
