from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exceptions():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
