"""Smoke-test every dashboard page in PUBLIC mode and LIVE mode without secrets (must fall back cleanly)."""
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "results" / "metrics.csv").exists(), reason="no results yet")


@pytest.mark.parametrize("mode", ["PUBLIC", "LIVE"])
@pytest.mark.parametrize("page", ["Today's Orders", "Backtest", "What-if", "About"])
def test_page_renders(mode, page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300)
    # unreachable sheet: LIVE must show a clear error and fall back (independent of any local secrets.toml)
    at.secrets["live"] = {"csv_url": "http://127.0.0.1:9/missing.csv"}
    at.run()
    at.sidebar.radio[0].set_value(mode)
    at.sidebar.radio[1].set_value(page)
    at.run()
    assert not at.exception, at.exception
    assert any("StockPilot" in m.value for m in at.markdown)
    if mode == "LIVE":
        assert any("LIVE mode" in e.value for e in at.error)   # clear error, then PUBLIC fallback
