from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .config import get_settings
from .manifest import load_issuers
from .pipeline import load_scorecards
from .runtime import Runtime


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} | SEC Research Terminal", page_icon="📈", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #07111f 0%, #0d1828 55%, #101c2e 100%); }
        [data-testid="stMetric"] {
          background:#132238; border:1px solid #243b58;
          padding:16px; border-radius:12px;
        }
        .evidence { border-left:3px solid #37c9a5; padding-left:12px; color:#b9c9dc; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(title)
    st.caption(
        "SEC-grounded research • Market data may be delayed • Not personalized investment advice"
    )


def issuers() -> list[Any]:
    return load_issuers(get_settings().manifest_path)


@st.cache_resource(show_spinner="Connecting to research services…")  # type: ignore[untyped-decorator]
def runtime() -> Runtime:
    return Runtime(get_settings())


@st.cache_data(ttl=900)  # type: ignore[untyped-decorator]
def scorecards() -> pd.DataFrame:
    return load_scorecards(get_settings())


def artifact_notice() -> None:
    st.info(
        "Offline artifacts are not present yet. Run `uv run finrag download`, `parse`, "
        "`build-artifacts`, and `index` before using data-backed views."
    )


def format_large(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    for suffix, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(value) >= divisor:
            return f"${value / divisor:,.2f}{suffix}"
    return f"${value:,.2f}"
