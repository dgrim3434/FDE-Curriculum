from metadata.structure import detect_headings
from retrieval.qasper import render_paper

PAPER = {
    "title": "Zero-Shot Translation via Bridge Pretraining",
    "abstract": "We study zero-shot translation.",
    "full_text": [
        {"section_name": "Introduction", "paragraphs": ["Para one.", "Para two."]},
        {"section_name": "Experiments ::: Setup ::: Datasets.", "paragraphs": ["We use Europarl."]},
        {"section_name": "Dogmatism in the Reddit Community ", "paragraphs": ["", "   ", "Reddit text."]},
        {"section_name": "", "paragraphs": ["Orphan paragraph continues."]},
        {"section_name": "Conclusion", "paragraphs": ["  Leading spaces kept.  "]},
    ],
    "qas": [],
}

EXPECTED_TEXT = (
    "# Abstract\n\n"
    "We study zero-shot translation.\n\n"
    "# Introduction\n\n"
    "Para one.\n\n"
    "Para two.\n\n"
    "# Experiments > Setup > Datasets\n\n"
    "We use Europarl.\n\n"
    "# Dogmatism in the Reddit Community\n\n"
    "Reddit text.\n\n"
    "Orphan paragraph continues.\n\n"
    "# Conclusion\n\n"
    "  Leading spaces kept.  "
)

EXPECTED_RENDERED = [
    (1, "Abstract"),
    (1, "Introduction"),
    (1, "Experiments > Setup > Datasets"),
    (1, "Dogmatism in the Reddit Community"),
    (1, "Conclusion"),
]


def test_doc_text_exact():
    doc, _ = render_paper(PAPER)
    assert doc == EXPECTED_TEXT


def test_rendered_list_exact_and_tuples():
    _, rendered = render_paper(PAPER)
    assert rendered == EXPECTED_RENDERED
    assert all(isinstance(r, tuple) for r in rendered)


def test_paragraphs_appear_verbatim():
    doc, _ = render_paper(PAPER)
    for sec in PAPER["full_text"]:
        for p in sec["paragraphs"]:
            if p.strip():
                assert p in doc, f"paragraph changed or missing: {p!r}"


def test_blank_paragraphs_skipped_no_double_blank_lines():
    doc, _ = render_paper(PAPER)
    assert "\n\n\n" not in doc


def test_empty_section_name_keeps_paragraphs_without_heading():
    doc, rendered = render_paper(PAPER)
    assert "Orphan paragraph continues." in doc
    assert "# \n" not in doc and not doc.rstrip().endswith("#")
    heading_lines = [ln for ln in doc.split("\n") if ln.startswith("# ")]
    assert len(heading_lines) == len(rendered)


def test_no_abstract_heading_when_abstract_blank():
    for blank in ["", "   "]:
        doc, rendered = render_paper({**PAPER, "abstract": blank})
        assert doc.startswith("# Introduction")
        assert (1, "Abstract") not in rendered


def test_title_is_not_rendered():
    doc, _ = render_paper(PAPER)
    assert PAPER["title"] not in doc


def test_detect_headings_reproduces_rendered():
    doc, rendered = render_paper(PAPER)
    assert [(h.level, h.title) for h in detect_headings(doc)] == rendered