import pytest

from metadata.fields import extract_fields, front_matter, parse_date, parse_fields

LICENSE = (
    "This eBook is for the use of anyone anywhere in the United States and\n"
    "most other parts of the world at no cost and with almost no restrictions\n"
    "whatsoever. You may copy it, give it away or re-use it under the terms\n"
    "of the Project Gutenberg License included with this eBook or online\n"
    "at www.gutenberg.org. If you are not located in the United States,\n"
    "you will have to check the laws of the country where you are located\n"
    "before using this eBook.\n"
)

WAR_AND_PEACE_HEADER = (
    "The Project Gutenberg eBook of War and Peace\n"
    "    \n"
    + LICENSE +
    "\n"
    "Title: War and Peace\n"
    "\n"
    "Author: graf Leo Tolstoy\n"
    "\n"
    "Translator: Aylmer Maude\n"
    "        Louise Maude\n"
    "\n"
    "\n"
    "        \n"
    "Release date: April 1, 2001 [eBook #2600]\n"
    "                Most recently updated: June 14, 2022\n"
    "\n"
    "Language: English\n"
    "\n"
    "Other information and formats: www.gutenberg.org/ebooks/2600\n"
    "\n"
)

BODY_WITH_DISTRACTORS = (
    "*** START OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***\n"
    "\n"
    "BOOK ONE: 1805\n"
    "Author: Somebody Else\n"
    "Release date: January 1, 1900\n"
    "\"Well, Prince, so Genoa and Lucca are now just family estates.\"\n"
)

WAR_AND_PEACE = WAR_AND_PEACE_HEADER + BODY_WITH_DISTRACTORS


# ---------------------------------------------------------------- front_matter
def test_front_matter_stops_at_start_marker():
    front = front_matter(WAR_AND_PEACE)
    assert "Title: War and Peace" in front
    assert "START OF" not in front and "Somebody Else" not in front


def test_front_matter_older_marker_spelling():
    text = "Title: X\n*** START OF THIS PROJECT GUTENBERG EBOOK X ***\nAuthor: Body"
    assert "Body" not in front_matter(text)


def test_front_matter_marker_is_case_insensitive():
    text = "Title: X\n*** start of the project gutenberg ebook x ***\nAuthor: Body"
    assert "Body" not in front_matter(text)


def test_front_matter_without_marker_is_first_100_lines():
    text = "\n".join(f"line {i}" for i in range(150))
    front = front_matter(text)
    assert len(front.splitlines()) == 100
    assert "line 99" in front and "line 100" not in front


# ---------------------------------------------------------------- parse_fields
def test_parse_fields_war_and_peace_layout():
    f = parse_fields(front_matter(WAR_AND_PEACE))
    assert f["title"] == ["War and Peace"]
    assert f["author"] == ["graf Leo Tolstoy"]
    assert f["translator"] == ["Aylmer Maude", "Louise Maude"]              # continuation line
    assert f["release date"] == ["April 1, 2001 [eBook #2600]"]             # update NOT glued on
    assert f["most recently updated"] == ["June 14, 2022"]                  # its own field
    assert f["language"] == ["English"]


def test_parse_fields_prose_resets_last_label():
    front = "Credits: Some Team\nThis line is prose, not a field\n    indented after prose\n"
    assert parse_fields(front)["credits"] == ["Some Team"]


def test_parse_fields_blank_lines_do_not_reset_last_label():
    assert parse_fields("Translator: A\n\n    B\n")["translator"] == ["A", "B"]


def test_parse_fields_duplicate_label_keeps_the_first():
    f = parse_fields("Title: First\nTitle: Second\n    more\n")
    assert f["title"] == ["First"]


def test_parse_fields_lowercases_labels_and_strips_values():
    assert parse_fields("Release Date:    June 1, 1998   \n") == {"release date": ["June 1, 1998"]}


# ---------------------------------------------------------------- parse_date
@pytest.mark.parametrize("raw,expected", [
    ("June 1, 1998 [eBook #1342]", "1998-06-01"),
    ("August 17, 2005 [EBook #5200]", "2005-08-17"),
    ("Jun 1, 1998", "1998-06-01"),
    ("1998-06-01", "1998-06-01"),
    ("1 June 1998", "1998-06-01"),
    ("June 1998", "1998-06"),                 # partial: never invent a day
    ("June 1998 [EBook #99]", "1998-06"),
    ("sometime in spring", None),
    ("", None),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


# ---------------------------------------------------------------- extract_fields
def test_extract_fields_war_and_peace_exact():
    assert extract_fields(WAR_AND_PEACE) == {
        "title": "War and Peace",
        "authors": ["graf Leo Tolstoy"],
        "date": "2001-04-01",
        "date_raw": "April 1, 2001 [eBook #2600]",
        "provenance": {"title": "label:title", "authors": "label:author", "date": "label:release date"},
    }


def test_translator_is_never_an_author():
    text = "Title: Metamorphosis\n\nAuthor: Franz Kafka\n\nTranslator: David Wyllie\n"
    assert extract_fields(text)["authors"] == ["Franz Kafka"]


def test_authors_split_on_and():
    out = extract_fields("Title: The Elements of Style\nAuthor: William Strunk and E. B. White\n")
    assert out["authors"] == ["William Strunk", "E. B. White"]


def test_multi_line_author_field():
    out = extract_fields("Title: X\nAuthor: First Person\n        Second Person\n")
    assert out["authors"] == ["First Person", "Second Person"]


def test_multi_line_title_is_joined():
    out = extract_fields("Title: The Adventures of\n       Sherlock Holmes\nAuthor: Arthur Conan Doyle\n")
    assert out["title"] == "The Adventures of Sherlock Holmes"


def test_title_falls_back_to_header_line_and_drops_by_tail():
    text = "\n\n   \nThe Project Gutenberg EBook of Emma, by Jane Austen\n\nRelease Date: August, 1994\n"
    out = extract_fields(text)
    assert out["title"] == "Emma"
    assert out["provenance"]["title"] == "header_line"


def test_release_date_wins_even_when_update_comes_first():
    text = "Most recently updated: June 9, 2026\nRelease date: August 17, 2005 [eBook #5200]\n"
    out = extract_fields(text)
    assert out["date"] == "2005-08-17"
    assert out["provenance"]["date"] == "label:release date"


def test_posting_date_used_when_no_release_date():
    out = extract_fields("Title: A Modest Proposal\nPosting Date: July 27, 2008 [EBook #1080]\n")
    assert out["date"] == "2008-07-27"
    assert out["date_raw"] == "July 27, 2008 [EBook #1080]"
    assert out["provenance"]["date"] == "label:posting date"


def test_unparseable_date_keeps_raw():
    out = extract_fields("Title: X\nRelease date: unknown\n")
    assert out["date"] is None
    assert out["date_raw"] == "unknown"
    assert out["provenance"]["date"] == "label:release date"


def test_missing_everything_is_none_with_not_found():
    out = extract_fields("Just some text.\nNothing labeled here.\n")
    assert out == {
        "title": None, "authors": None, "date": None, "date_raw": None,
        "provenance": {"title": "not_found", "authors": "not_found", "date": "not_found"},
    }


def test_body_fields_after_start_marker_are_ignored():
    out = extract_fields(WAR_AND_PEACE)
    assert "Somebody Else" not in (out["authors"] or [])
    assert out["date"] != "1900-01-01"