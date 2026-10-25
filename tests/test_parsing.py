from sec_fin_rag.models import Filing
from sec_fin_rag.parsing import parse_filing_html, section_distribution_audit


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


def test_short_styled_inline_xbrl_item_heading_is_detected_before_length_filter() -> None:
    html = """
    <html><body>
    <p><b>DOCUMENTS INCORPORATED BY REFERENCE</b></p>
    <div style="font-weight:700"><span>Item 1A. Ris</span>
      <ix:nonNumeric name="dei:DocumentType">k Factors</ix:nonNumeric></div>
    <div>Our concentrated supply chain and reliance on suppliers could materially harm
    operations.</div>
    </body></html>
    """
    blocks = parse_filing_html(html, filing())
    assert len(blocks) == 1
    assert blocks[0].section == "Item 1A. Risk Factors"


def test_item_context_survives_uppercase_subsection_heading() -> None:
    html = """
    <html><body>
    <div><span>Item 1A. Risk Factors</span></div>
    <h3>RISKS RELATED TO OUR SUPPLY CHAIN</h3>
    <p>Our suppliers may be unable to provide enough components during periods of high demand.</p>
    </body></html>
    """
    blocks = parse_filing_html(html, filing())
    assert blocks[0].section == "Item 1A. Risk Factors — RISKS RELATED TO OUR SUPPLY CHAIN"


def test_item_only_heading_uses_form_specific_canonical_title() -> None:
    html = """
    <html><body>
    <div><span>Item 1.</span></div>
    <p>We design and sell products to customers throughout the world and across many industries.</p>
    </body></html>
    """
    blocks = parse_filing_html(html, filing())
    assert blocks[0].section == "Item 1. Business"


def test_fragmented_business_description_does_not_inherit_front_matter() -> None:
    html = """
    <html><body>
    <p>DOCUMENTS INCORPORATED BY REFERENCE</p>
    <p><span>Item 1. Busines</span><span>s Description</span></p>
    <p>We operate insurance and other businesses through subsidiaries around the world.</p>
    </body></html>
    """
    blocks = parse_filing_html(html, filing())
    assert blocks[0].section == "Item 1. Business Description"


def test_styled_item_7_and_item_8_headings_are_preserved() -> None:
    html = """
    <html><body>
    <div><b>Item 7. Management's Discussion and Analysis of Financial Condition and
    Results of Operations</b></div>
    <p>Revenue increased because demand for the company's products was substantially higher.</p>
    <div><b>Item 8. Financial Statements and Supplementary Data</b></div>
    <table><tr><th>Year</th><th>Revenue</th></tr><tr><td>2026</td><td>100</td></tr></table>
    </body></html>
    """
    blocks = parse_filing_html(html, filing())
    assert blocks[0].section.startswith("Item 7. Management's Discussion")
    assert blocks[1].section == "Item 8. Financial Statements and Supplementary Data"


def test_table_of_contents_does_not_become_item_heading() -> None:
    html = """
    <html><body>
    <table><tr><td>Item 1.</td><td>Business</td></tr>
      <tr><td>Item 1A.</td><td>Risk Factors</td></tr></table>
    <p>This front-matter paragraph is long enough to become a document block for testing.</p>
    </body></html>
    """
    blocks = parse_filing_html(html, filing())
    assert blocks[-1].section == "Document"


def test_section_distribution_audit_reports_missing_risk_labels() -> None:
    good = parse_filing_html(
        "<html><body><p>Item 1A. Risk Factors</p>"
        "<p>Supply chain disruption could materially harm our business and financial results.</p>"
        "</body></html>",
        filing(),
    )
    other_filing = Filing(**{**filing().__dict__, "accession": "0001-26-000002"})
    missing = parse_filing_html(
        "<html><body><p>Item 1. Business</p>"
        "<p>We provide products and services to customers throughout the world.</p></body></html>",
        other_filing,
    )
    audit = section_distribution_audit(good + missing)
    assert audit["summary"]["risk_factor_block_count"] == 1
    assert audit["summary"]["missing_10k_risk_label_accessions"] == ["0001-26-000002"]
