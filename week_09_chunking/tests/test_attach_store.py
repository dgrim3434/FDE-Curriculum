import json

import numpy as np
import pytest

from chunking.base import Chunk
from metadata.attach import attach_metadata, build_embed_text, set_embed_text
from metadata.store import check_aligned, load_chunks, load_vectors, save_chunks, save_vectors
from metadata.structure import build_sections, detect_headings

DOC = "Preamble.\n\n# Methods\n\nPatients were enrolled.\n\n# Results\n\nIt worked."

DOC_FIELDS = {
    "title": "A Trial",
    "authors": ["Ann Lee"],
    "date": "2001-04-01",
    "date_raw": "April 1, 2001",
    "provenance": {"title": "label:title", "authors": "label:author", "date": "label:release date"},
}


def ck(doc, idx, sub, doc_id="d1"):
    s = doc.index(sub)
    return Chunk(doc_id, idx, s, s + len(sub), sub, len(sub.split()))


def make_chunks():
    return [ck(DOC, 0, "Preamble."), ck(DOC, 1, "Patients were enrolled."), ck(DOC, 2, "It worked.")]


def attached():
    chunks = make_chunks()
    sections = build_sections(detect_headings(DOC), len(DOC))
    return attach_metadata(chunks, DOC_FIELDS, sections)


# ---------------------------------------------------------------- attach_metadata
def test_attach_returns_the_same_list():
    chunks = make_chunks()
    out = attach_metadata(chunks, DOC_FIELDS, build_sections(detect_headings(DOC), len(DOC)))
    assert out is chunks


def test_section_paths_by_position():
    paths = [c.metadata["section_path"] for c in attached()]
    assert paths == [[], ["Methods"], ["Results"]]          # preamble has no section


def test_doc_fields_copied():
    for c in attached():
        assert c.metadata["title"] == "A Trial"
        assert c.metadata["authors"] == ["Ann Lee"]
        assert c.metadata["date"] == "2001-04-01"
        assert c.metadata["provenance"] == DOC_FIELDS["provenance"]
        assert c.metadata["n_chunks"] == 3


def test_prev_and_next_are_chunk_ids():
    c0, c1, c2 = attached()
    assert (c0.metadata["prev_id"], c0.metadata["next_id"]) == (None, "d1::1")
    assert (c1.metadata["prev_id"], c1.metadata["next_id"]) == ("d1::0", "d1::2")
    assert (c2.metadata["prev_id"], c2.metadata["next_id"]) == ("d1::1", None)


def test_single_chunk_has_no_neighbors():
    (c,) = attach_metadata([ck(DOC, 0, "It worked.")], DOC_FIELDS, [])
    assert c.metadata["prev_id"] is None and c.metadata["next_id"] is None
    assert c.metadata["section_path"] == []


def test_text_and_spans_untouched():
    before = [(c.text, c.start, c.end) for c in make_chunks()]
    after = [(c.text, c.start, c.end) for c in attached()]
    assert before == after


# ---------------------------------------------------------------- build_embed_text / set_embed_text
def test_embed_mode_none():
    c = attached()[1]
    assert build_embed_text(c, "none") == "Patients were enrolled."


def test_embed_mode_title():
    c = attached()[1]
    assert build_embed_text(c, "title") == "Title: A Trial\nPatients were enrolled."


def test_embed_mode_title_section():
    c = attached()[1]
    assert build_embed_text(c, "title_section") == "Title: A Trial\nSection: Methods\nPatients were enrolled."


def test_embed_nested_section_path_joined():
    c = attached()[1]
    c.metadata["section_path"] = ["2 Methods", "2.1 Cohort"]
    assert build_embed_text(c, "title_section") == \
        "Title: A Trial\nSection: 2 Methods > 2.1 Cohort\nPatients were enrolled."


def test_embed_missing_parts_are_left_out():
    c = attached()[1]
    c.metadata["title"] = None
    assert build_embed_text(c, "title") == "Patients were enrolled."
    assert build_embed_text(c, "title_section") == "Section: Methods\nPatients were enrolled."

    c = attached()[0]                                        # preamble: empty section path
    assert build_embed_text(c, "title_section") == "Title: A Trial\nPreamble."

    c.metadata["title"] = None
    assert build_embed_text(c, "title_section") == "Preamble."


def test_embed_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_embed_text(attached()[0], "everything")


def test_embed_does_not_modify_text():
    c = attached()[1]
    build_embed_text(c, "title_section")
    assert c.text == "Patients were enrolled."


def test_set_embed_text_writes_both_keys():
    chunks = attached()
    set_embed_text(chunks, "title")
    for c in chunks:
        assert c.metadata["embed_mode"] == "title"
        assert c.metadata["embed_text"] == "Title: A Trial\n" + c.text


# ---------------------------------------------------------------- save_chunks / load_chunks
def test_chunks_round_trip_equal(tmp_path):
    chunks = attached()
    chunks[1].metadata["note"] = "Café — naïve résumé"
    path = tmp_path / "sub" / "chunks.jsonl"                 # parent folder does not exist yet
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks


def test_jsonl_one_object_per_line_with_chunk_id(tmp_path):
    path = tmp_path / "chunks.jsonl"
    save_chunks(attached(), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(ln)["chunk_id"] for ln in lines] == ["d1::0", "d1::1", "d1::2"]


def test_empty_list_round_trip(tmp_path):
    path = tmp_path / "empty.jsonl"
    save_chunks([], path)
    assert load_chunks(path) == []


# ---------------------------------------------------------------- vectors and alignment
def test_vectors_round_trip(tmp_path):
    vecs = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    ids = ["d1::0", "d1::1", "d1::2"]
    path = tmp_path / "sub" / "vectors.npz"
    save_vectors(path, vecs, ids)
    v2, ids2 = load_vectors(path)
    assert np.array_equal(v2, vecs)
    assert ids2 == ids and isinstance(ids2, list)


def test_check_aligned_passes_on_matching_order():
    chunks = attached()
    check_aligned(chunks, ["d1::0", "d1::1", "d1::2"])
    check_aligned(chunks, np.array(["d1::0", "d1::1", "d1::2"]))      # array input works too


@pytest.mark.parametrize("ids", [
    ["d1::1", "d1::0", "d1::2"],          # reordered
    ["d1::0", "d1::1"],                   # missing one
])
def test_check_aligned_raises_on_mismatch(ids):
    with pytest.raises(ValueError):
        check_aligned(attached(), ids)