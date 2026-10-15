import pytest

from metadata.structure import Heading, build_sections, detect_headings, section_path_at


def found(doc):
    """detect_headings as (level, title, start) tuples for easy comparison."""
    return [(h.level, h.title, h.start) for h in detect_headings(doc)]


def is_heading(line):
    return len(detect_headings(line)) == 1


# ---------------------------------------------------------------- detect_headings: each rule
def test_markdown_levels_and_positions():
    doc = "# Intro\n\nSome text here.\n\n## Background\nMore text.\n### Deep Dive\n"
    assert found(doc) == [
        (1, "Intro", 0),
        (2, "Background", doc.index("## Background")),
        (3, "Deep Dive", doc.index("### Deep Dive")),
    ]


def test_markdown_title_is_stripped():
    assert found("##   Results   ") == [(2, "Results", 0)]


@pytest.mark.parametrize("line", ["#NoSpace", "####### Seven hashes"])
def test_markdown_rejects_malformed(line):
    assert not is_heading(line)


@pytest.mark.parametrize("line,level,title", [
    ("2 Methods", 1, "Methods"),
    ("3. Results", 1, "Results"),
    ("2.1 Participants", 2, "Participants"),
    ("2.1.3 Inclusion criteria", 3, "Inclusion criteria"),
])
def test_numbered_headings(line, level, title):
    assert found(line) == [(level, title, 0)]


@pytest.mark.parametrize("line,level", [
    ("CHAPTER I.", 2),
    ("Chapter 12", 2),
    ("BOOK ONE: 1805", 1),
    ("PART II", 1),
])
def test_chapter_headings(line, level):
    assert found(line) == [(level, line, 0)]              # title = whole stripped line


@pytest.mark.parametrize("line", ["Results", "Abstract", "Related Work", "Acknowledgements.", "  Conclusions:  "])
def test_known_section_names(line):
    hs = detect_headings(line)
    assert len(hs) == 1 and hs[0].level == 1


def test_markdown_wins_over_numbered():
    assert found("# 2 Methods") == [(1, "2 Methods", 0)]


# ---------------------------------------------------------------- detect_headings: false positives
@pytest.mark.parametrize("line", [
    "1. Patients were randomized to two groups.",                                         # numbered sentence
    "4 patients in the treatment arm and 2 in the control arm withdrew before follow up", # > 10 words
    "Figure 2. Survival by treatment arm",                                                # caption
    "Table 3 Baseline characteristics",                                                   # caption
    "Results were consistent across sites.",                                              # starts with a known name
    "Part of our approach is described in the next section of this paper",                # chapter word, > 8 words
    "Book reviews were collected from several online sources in the spring",              # chapter word, > 8 words
])
def test_not_headings(line):
    assert not is_heading(line)


def test_windows_line_endings_positions_and_titles():
    doc = "Intro text\r\n\r\n## Methods\r\nBody line\r\n2.1 Setup\r\n"
    assert found(doc) == [
        (2, "Methods", doc.index("## Methods")),
        (2, "Setup", doc.index("2.1 Setup")),
    ]


def test_empty_document():
    assert detect_headings("") == []


# ---------------------------------------------------------------- build_sections (your Section 6 checkpoint)
CHECKPOINT = [
    Heading(level=1, title="Abstract", start=0),
    Heading(level=1, title="1 Introduction", start=400),
    Heading(level=1, title="2 Methods", start=900),
    Heading(level=2, title="2.1 Cohort", start=960),
    Heading(level=3, title="2.1.1 Inclusion", start=1100),
    Heading(level=2, title="2.2 Statistics", start=1600),
    Heading(level=1, title="3 Results", start=2100),
]


def test_build_sections_spans_and_paths():
    secs = build_sections(CHECKPOINT, doc_len=3000)
    got = [(s.title, s.start, s.end, s.path) for s in secs]
    assert got == [
        ("Abstract", 0, 400, ["Abstract"]),
        ("1 Introduction", 400, 900, ["1 Introduction"]),
        ("2 Methods", 900, 2100, ["2 Methods"]),
        ("2.1 Cohort", 960, 1600, ["2 Methods", "2.1 Cohort"]),
        ("2.1.1 Inclusion", 1100, 1600, ["2 Methods", "2.1 Cohort", "2.1.1 Inclusion"]),
        ("2.2 Statistics", 1600, 2100, ["2 Methods", "2.2 Statistics"]),      # sibling replaced 2.1
        ("3 Results", 2100, 3000, ["3 Results"]),
    ]


def test_build_sections_one_per_heading_in_document_order():
    secs = build_sections(CHECKPOINT, doc_len=3000)
    assert len(secs) == len(CHECKPOINT)
    assert [s.start for s in secs] == sorted(s.start for s in secs)


def test_build_sections_skipped_level():
    secs = build_sections([Heading(1, "A", 0), Heading(3, "C", 50), Heading(1, "B", 100)], doc_len=200)
    by_title = {s.title: s for s in secs}
    assert by_title["C"].path == ["A", "C"]
    assert (by_title["A"].end, by_title["C"].end, by_title["B"].end) == (100, 100, 200)


def test_build_sections_paths_are_independent_lists():
    secs = build_sections(CHECKPOINT, doc_len=3000)
    secs[2].path.append("MUTATED")
    assert secs[3].path == ["2 Methods", "2.1 Cohort"], "sections must not share one path list"


def test_build_sections_single_and_empty():
    assert build_sections([], doc_len=100) == []
    (only,) = build_sections([Heading(2, "Only", 10)], doc_len=100)
    assert (only.start, only.end, only.path) == (10, 100, ["Only"])


# ---------------------------------------------------------------- section_path_at
@pytest.fixture
def checkpoint_sections():
    secs = build_sections(CHECKPOINT, doc_len=3000)
    secs_shifted = build_sections(
        [Heading(h.level, h.title, h.start + 50) for h in CHECKPOINT], doc_len=3050)
    return secs, secs_shifted


@pytest.mark.parametrize("pos,expected", [
    (1200, ["2 Methods", "2.1 Cohort", "2.1.1 Inclusion"]),
    (1650, ["2 Methods", "2.2 Statistics"]),
    (950, ["2 Methods"]),
    (2500, ["3 Results"]),
    (960, ["2 Methods", "2.1 Cohort"]),          # exactly at a heading's start
    (2999, ["3 Results"]),
])
def test_section_path_at_checkpoint_answers(checkpoint_sections, pos, expected):
    secs, _ = checkpoint_sections
    assert section_path_at(secs, pos) == expected


def test_section_path_at_before_first_heading(checkpoint_sections):
    _, shifted = checkpoint_sections              # first heading now starts at 50
    assert section_path_at(shifted, 10) == []


# ---------------------------------------------------------------- end to end
def test_detect_build_lookup_end_to_end():
    doc = "Preamble line.\n\n# Methods\n\n## Cohort\n\nPatients were enrolled.\n\n# Results\n\nIt worked.\n"
    secs = build_sections(detect_headings(doc), doc_len=len(doc))
    assert section_path_at(secs, doc.index("Patients")) == ["Methods", "Cohort"]
    assert section_path_at(secs, doc.index("It worked")) == ["Results"]
    assert section_path_at(secs, doc.index("Preamble")) == []