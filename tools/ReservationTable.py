from langchain_core.tools import tool # type: ignore
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
DOCUMENTS_PATH = ROOT / "documents"

@tool(parse_docstring=True)
def record_reservation(title: str, date, location):
    pass