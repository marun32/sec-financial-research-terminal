from sec_fin_rag.models import Filing
from sec_fin_rag.parsing import parse_filing_html


def filing() -> Filing:
    return Filing(
        issuer="Example Corp",
        ticker="EX",
        cik=1,
        form="10-K",
        filing_date="2026-01-01",
        report_date="2025-12-31",
        accession="0001-26-000001",
        primary_document="example.htm",
        source_url="https://www.sec.gov/example",
    )


def test_table_is_converted_to_markdown_with_context() -> None:
    html = """
    <html><body><h2>Consolidated Statements</h2>
    <table><tr><th>Year</th><th>Revenue</th></tr>
    <tr><td>2026</td><td>$100</td></tr></table></body></html>
    """
    blocks = parse_filing_html(html, filing())
    assert len(blocks) == 1
    assert blocks[0].block_type == "table"
    assert blocks[0].section == "Consolidated Statements"
    assert "| Year | Revenue |" in blocks[0].text
    assert blocks[0].metadata["source_url"].startswith("https://www.sec.gov")
