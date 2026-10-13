from finhallu.corpus import Retriever, html_to_text

HTML = """<html><head><title>x</title></head><body>
<div style="display:none"><ix:header>HIDDEN XBRL 999999</ix:header></div>
<p>Item 8. Financial Statements</p>
<table><tr><td>Total net sales</td><td>$</td><td><ix:nonFraction>416,161</ix:nonFraction></td><td>$</td><td>391,035</td></tr>
<tr><td>Net income</td><td>$</td><td>112,010</td><td>$</td><td>93,736</td></tr></table>
<p>Risk factors discuss supply chain and tariffs.</p></body></html>"""


def test_html_to_text_keeps_rows_drops_hidden():
    t = html_to_text(HTML)
    assert "Total net sales | 416,161 | 391,035" in t
    assert "HIDDEN XBRL" not in t


def test_retriever_finds_table():
    r = Retriever(html_to_text(HTML) + "\n" + ("filler text about operations. " * 400), k=1)
    assert "416,161" in r.search("total net sales revenue")[0]
