from paper_provenance import numbered_blocks


def test_numbered_blocks_extracts_multiline_table_caption():
    text = """**Table 6. Long caption
(continued)**

| A | B |
|---|---|
| 1 | 2 |

After.
"""
    blocks = numbered_blocks(text, "Table")

    assert set(blocks) == {"6"}
    assert "(continued)**" in blocks["6"]
    assert "| 1 | 2 |" in blocks["6"]
    assert "After." not in blocks["6"]


def test_numbered_blocks_includes_figure_image_and_caption():
    text = """![Alt](figure.png)

**Figure 1. Caption.** Text.

Next.
"""
    blocks = numbered_blocks(text, "Figure")

    assert blocks["1"].startswith("![Alt](figure.png)")
    assert "**Figure 1." in blocks["1"]
