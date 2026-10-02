"""AI generated Test Cases for my parser functions"""

import pytest

from prompting.parsing import (
    ParseResults,
    extract_tag,
    normalize_numbers,
    numbers_equal,
    normalize_answer,
    exact_match,
    f1,
    normalize_label,
    make_number_parser,
    make_label_parser,
    make_text_parser,
    Action,
    Finish,
    ReActParserError,
    parse_react,
    ACTION_EX,
)


# ─────────────────────────────── extract_tag ───────────────────────────────

class TestExtractTag:
    def test_basic(self):
        assert extract_tag("<answer>18</answer>", "answer") == "18"

    def test_other_tag_name(self):
        assert extract_tag("reasoning... <type>bridge</type>", "type") == "bridge"

    def test_takes_last_when_placeholder_mentioned_first(self):
        text = "I'll put it in <answer></answer> at the end. So 6*3=18. <answer>18</answer>"
        assert extract_tag(text, "answer") == "18"

    def test_takes_last_on_self_correction(self):
        text = "<answer>12</answer> wait, no. <answer>18</answer>"
        assert extract_tag(text, "answer") == "18"

    def test_multiline_content_returned_raw(self):
        assert extract_tag("<answer>\n 18 \n</answer>", "answer") == "\n 18 \n"

    def test_case_insensitive(self):
        assert extract_tag("<Answer>18</Answer>", "answer") == "18"

    def test_unclosed_returns_none(self):
        assert extract_tag("so the result is <answer>18", "answer") is None

    def test_whitespace_only_returned_raw(self):
        assert extract_tag("<answer>   </answer>", "answer") == "   "

    def test_empty_returned_raw(self):
        assert extract_tag("<answer></answer>", "answer") == ""

    def test_no_tag_returns_none(self):
        assert extract_tag("The answer is 18.", "answer") is None

    def test_does_not_match_a_different_tag(self):
        assert extract_tag("<answers>18</answers>", "answer") is None


# ──────────────────────────── normalize_numbers ────────────────────────────

class TestNormalizeNumbers:
    @pytest.mark.parametrize("s", ["18", "18.0", "18.", "  18  "])
    def test_plain_forms(self, s):
        assert normalize_numbers(s) == pytest.approx(18.0)

    def test_currency_and_commas(self):
        assert normalize_numbers("$1,080.00") == pytest.approx(1080.0)

    @pytest.mark.parametrize("s", ["18 dollars", "The answer is 18", "18 apples."])
    def test_number_inside_words(self, s):
        assert normalize_numbers(s) == pytest.approx(18.0)

    def test_negative(self):
        assert normalize_numbers("-5") == pytest.approx(-5.0)

    def test_decimal(self):
        assert normalize_numbers("0.5") == pytest.approx(0.5)

    @pytest.mark.parametrize("s", ["", "   ", "eighteen", "no number here"])
    def test_no_number_returns_none(self, s):
        assert normalize_numbers(s) is None

    def test_percent_pinned(self):
        assert normalize_numbers("50%") == pytest.approx(50.0)

    def test_fraction(self):
        assert normalize_numbers("3/4") == pytest.approx(0.75)

    def test_fraction_with_zero_denominator_is_none(self):
        assert normalize_numbers("1/0") is None

    def test_first_number_wins(self):
        assert normalize_numbers("18 (it takes 3 hours)") == pytest.approx(18.0)

    def test_first_number_wins_even_if_a_fraction_comes_later(self):
        assert normalize_numbers("18 (she ate 1/2 of them)") == pytest.approx(18.0)


class TestNumbersEqual:
    def test_equal(self):
        assert numbers_equal(18.0, 18.0)

    def test_float_rounding(self):
        assert numbers_equal(0.1 + 0.2, 0.3)

    def test_different(self):
        assert not numbers_equal(18.0, 18.5)

    @pytest.mark.parametrize("a,b", [(None, 18.0), (18.0, None), (None, None)])
    def test_none_is_never_equal(self, a, b):
        assert numbers_equal(a, b) is False


# ─────────────────────── normalize_answer / EM / F1 ────────────────────────

class TestNormalizeAnswer:
    def test_article_and_punctuation(self):
        assert normalize_answer("The Scott Derrickson.") == "scott derrickson"

    def test_whitespace_collapsed(self):
        assert normalize_answer("  Ed   Wood ") == "ed wood"

    def test_leading_an(self):
        assert normalize_answer("an American") == "american"

    def test_article_mid_phrase(self):
        assert normalize_answer("A Theory of the Universe") == "theory of universe"

    def test_articles_only_removed_as_whole_words(self):
        assert normalize_answer("theater") == "theater"
        assert normalize_answer("Anatomy") == "anatomy"

    def test_internal_punctuation(self):
        assert normalize_answer("U.S.A.") == "usa"


class TestExactMatch:
    def test_match_after_normalization(self):
        assert exact_match("The Ed Wood", "ed wood") is True

    def test_extra_word_fails(self):
        assert exact_match("Ed Wood film", "Ed Wood") is False


class TestF1:
    def test_identical(self):
        assert f1("Scott Derrickson", "Scott Derrickson") == pytest.approx(1.0)

    def test_partial_overlap_is_token_level(self):
        # pred tokens [scott, derrickson, film], gold [scott, derrickson]
        # P = 2/3, R = 1 -> F1 = 0.8
        assert f1("the Scott Derrickson film", "Scott Derrickson") == pytest.approx(0.8)

    def test_repeated_tokens_counted_once_per_gold(self):
        # pred [wood, wood], gold [wood] -> common 1, P = 1/2, R = 1 -> 2/3
        assert f1("wood wood", "wood") == pytest.approx(2 / 3)

    def test_no_overlap_is_zero(self):
        assert f1("Ed Wood", "Scott Derrickson") == pytest.approx(0.0)

    def test_empty_prediction_is_zero_not_crash(self):
        assert f1("", "Scott Derrickson") == pytest.approx(0.0)

    def test_prediction_that_normalizes_to_empty_is_zero(self):
        assert f1("the.", "Scott Derrickson") == pytest.approx(0.0)


# ───────────────────────────── normalize_label ─────────────────────────────

ALLOWED = {"comparison", "bridge"}


class TestNormalizeLabel:
    @pytest.mark.parametrize("s", ["comparison", "Comparison", " comparison. ", "COMPARISON"])
    def test_variants_of_comparison(self, s):
        assert normalize_label(s, ALLOWED) == "comparison"

    @pytest.mark.parametrize("s", ["**bridge**", '"bridge"', "'bridge'"])
    def test_decorated_bridge(self, s):
        assert normalize_label(s, ALLOWED) == "bridge"

    def test_not_in_set(self):
        assert normalize_label("both", ALLOWED) is None

    def test_returns_label_exactly_as_in_allowed_set(self):
        assert normalize_label("yes", {"Yes", "No"}) == "Yes"


# ──────────────────────────────── factories ────────────────────────────────

class TestNumberParser:
    def test_ok(self):
        r = make_number_parser()("so 6 * 3 = 18. <answer>$18</answer>")
        assert isinstance(r, ParseResults)
        assert r.ok is True and r.value == pytest.approx(18.0) and r.error is None

    def test_missing_tag(self):
        r = make_number_parser()("The answer is 18.")
        assert r.ok is False and r.value is None
        assert isinstance(r.error, str) and "<answer>" in r.error

    def test_not_a_number(self):
        r = make_number_parser()("<answer>eighteen</answer>")
        assert r.ok is False and r.value is None
        assert isinstance(r.error, str) and r.error

    @pytest.mark.parametrize("text", ["<answer></answer>", "<answer>   </answer>", "<answer>\n</answer>"])
    def test_empty_or_whitespace_tag_is_error(self, text):
        r = make_number_parser()(text)
        assert r.ok is False and r.value is None and r.error

    def test_whitespace_around_number_ok(self):
        r = make_number_parser()("<answer>\n  18  \n</answer>")
        assert r.ok and r.value == pytest.approx(18.0)

    def test_missing_tag_and_bad_number_give_different_errors(self):
        p = make_number_parser()
        assert p("no tag").error != p("<answer>eighteen</answer>").error

    def test_custom_tag(self):
        r = make_number_parser("result")("<result>42</result>")
        assert r.ok and r.value == pytest.approx(42.0)

    def test_custom_tag_error_names_that_tag(self):
        r = make_number_parser("result")("<answer>42</answer>")
        assert not r.ok and "<result>" in r.error

    def test_factories_are_independent(self):
        a, b = make_number_parser("a"), make_number_parser("b")
        assert a("<a>1</a>").ok and not a("<b>1</b>").ok
        assert b("<b>1</b>").ok and not b("<a>1</a>").ok


class TestLabelParser:
    def test_clean(self):
        r = make_label_parser("type", ALLOWED)("<type>bridge</type>")
        assert r.ok and r.value == "bridge" and r.error is None

    def test_checkpoint_1_case_label_inside_reasoning(self):
        text = "I think this is <type>Comparison</type> since it compares two people."
        r = make_label_parser("type", ALLOWED)(text)
        assert r.ok and r.value == "comparison"

    def test_label_not_allowed_lists_every_option(self):
        r = make_label_parser("type", ALLOWED)("<type>both</type>")
        assert not r.ok and r.value is None
        assert "comparison" in r.error and "bridge" in r.error

    def test_missing_tag(self):
        r = make_label_parser("type", ALLOWED)("comparison")
        assert not r.ok and "<type>" in r.error

    @pytest.mark.parametrize("text", ["<type></type>", "<type>   </type>"])
    def test_empty_or_whitespace_tag_is_error(self, text):
        r = make_label_parser("type", ALLOWED)(text)
        assert r.ok is False and r.value is None and r.error

    def test_whitespace_around_label_ok(self):
        r = make_label_parser("type", ALLOWED)("<type>\n  Bridge  \n</type>")
        assert r.ok and r.value == "bridge"


class TestTextParser:
    def test_stripped_case_preserved(self):
        r = make_text_parser()("<answer>  Scott Derrickson </answer>")
        assert r.ok and r.value == "Scott Derrickson"

    def test_multiline_prompt(self):
        text = "<diagnosis>drops units</diagnosis><prompt>Solve:\n{{ question }}\nEnd with <answer>N</answer></prompt>"
        r = make_text_parser("prompt")(text)
        assert r.ok and "{{ question }}" in r.value and r.value.startswith("Solve:")

    @pytest.mark.parametrize("text", ["<answer></answer>", "<answer> </answer>", "<answer>\n\t</answer>"])
    def test_empty_or_whitespace_tag_is_error(self, text):
        r = make_text_parser()(text)
        assert not r.ok and r.value is None and r.error

    def test_last_tag_used(self):
        r = make_text_parser()("<answer>Ed Wood</answer> actually <answer>Scott Derrickson</answer>")
        assert r.ok and r.value == "Scott Derrickson"

    def test_missing_tag(self):
        r = make_text_parser("prompt")("here is a prompt")
        assert not r.ok and "<prompt>" in r.error


# ──────────────────────────────── parse_react ──────────────────────────────

CLEAN = (
    "<thought>I need Ed Wood's nationality.</thought>\n"
    "<action>search</action>\n"
    "<input>Ed Wood</input>"
)


class TestReActAction:
    def test_clean_action(self):
        r = parse_react(CLEAN)
        assert isinstance(r, Action)
        assert r.tool == "search"
        assert r.arg == "Ed Wood"
        assert r.thought == "I need Ed Wood's nationality."

    def test_case_and_whitespace_normalized(self):
        r = parse_react("<Thought>x</Thought><Action> Search </Action><Input>  Ed Wood </Input>")
        assert isinstance(r, Action)
        assert r.tool == "Search" and r.arg == "Ed Wood"

    def test_unknown_tool_still_parses(self):
        r = parse_react("<action>google</action><input>Ed Wood</input>")
        assert isinstance(r, Action) and r.tool == "google"

    def test_multiline_input_kept(self):
        r = parse_react("<action>calculate</action><input>2 + 3 *\n4</input>")
        assert isinstance(r, Action) and r.arg == "2 + 3 *\n4"

    def test_second_complete_pair_is_ignored(self):
        text = (
            "<action>search</action><input>A</input>"
            "<action>lookup</action><input>B</input>"
        )
        r = parse_react(text)
        assert isinstance(r, Action) and r.tool == "search" and r.arg == "A"


class TestReActFinish:
    def test_finish_tag(self):
        r = parse_react("<thought>Both are American.</thought>\n<finish>yes</finish>")
        assert isinstance(r, Finish)
        assert r.answer == "yes" and r.thought == "Both are American."

    def test_finish_answer_stripped(self):
        r = parse_react("<finish>  Scott Derrickson  </finish>")
        assert isinstance(r, Finish) and r.answer == "Scott Derrickson"

    def test_finish_before_action_wins(self):
        r = parse_react("<finish>yes</finish>\n<action>search</action><input>x</input>")
        assert isinstance(r, Finish) and r.answer == "yes"

    def test_action_before_finish_wins(self):
        r = parse_react("<action>search</action><input>x</input>\n<finish>yes</finish>")
        assert isinstance(r, Action) and r.tool == "search"


class TestReActThought:
    def test_unclosed_thought(self):
        r = parse_react("<thought>I need X.\n<action>search</action><input>X</input>")
        assert isinstance(r, Action) and r.thought == "I need X."

    def test_no_thought_tags_uses_text_before_action(self):
        r = parse_react("I need X.\n<action>search</action><input>X</input>")
        assert isinstance(r, Action) and r.thought == "I need X."

    def test_nothing_before_action_is_none(self):
        r = parse_react("<action>search</action><input>X</input>")
        assert isinstance(r, Action) and r.thought is None

    def test_whitespace_before_action_is_none(self):
        r = parse_react("   \n<action>search</action><input>X</input>")
        assert isinstance(r, Action) and r.thought is None


class TestReActErrors:
    @pytest.mark.parametrize("text", [
        "The answer is yes.",                                              # nothing at all
        "<thought>hmm</thought><action>search</action>",                   # no input
        "<input>Ed Wood</input><action>search</action>",                   # input only BEFORE action
        "<action>search</action><action>lookup</action><input>x</input>",  # action between action and input
        "<action>search</action><input>   </input>",                       # empty input
        "<finish>   </finish>",                                            # empty finish
        "<action>finish</action><input> </input>",                         # empty finish via action
    ])
    def test_error_type_reason_and_raw(self, text):
        r = parse_react(text)
        assert isinstance(r, ReActParserError)
        assert isinstance(r.reason, str) and r.reason.strip()
        assert r.raw == text

    def test_no_action_error_teaches_the_format(self):
        r = parse_react("The answer is yes.")
        assert "<action>" in r.reason and "<finish>" in r.reason

    def test_missing_input_error_mentions_input_tag(self):
        r = parse_react("<action>search</action>")
        assert "<input>" in r.reason


class TestReActKept:
    def test_kept_ends_at_input_for_clean_action(self):
        r = parse_react(CLEAN)
        assert r.kept.rstrip().endswith("</input>")
        assert "I need Ed Wood's nationality." in r.kept

    def test_hallucinated_observation_is_cut(self):
        text = (
            CLEAN
            + "\n<observation>Ed Wood was a Canadian filmmaker.</observation>"
            + "\n<thought>They differ.</thought><action>finish</action><input>no</input>"
        )
        r = parse_react(text)
        assert isinstance(r, Action) and r.arg == "Ed Wood"
        assert "Canadian" not in r.kept
        assert "They differ" not in r.kept

    def test_plain_text_observation_is_cut(self):
        r = parse_react(CLEAN + "\nObservation: Ed Wood was Canadian.")
        assert "Canadian" not in r.kept

    def test_kept_ends_at_finish(self):
        r = parse_react("<thought>ok</thought><finish>yes</finish>\nObservation: extra")
        assert isinstance(r, Finish)
        assert r.kept.rstrip().endswith("</finish>") and "extra" not in r.kept

    def test_kept_on_missing_input_ends_at_action(self):
        r = parse_react("<thought>t</thought><action>search</action> and then I rambled on")
        assert isinstance(r, ReActParserError)
        assert r.kept.rstrip().endswith("</action>") and "rambled" not in r.kept

    def test_kept_on_total_failure_is_the_raw_reply(self):
        text = "The answer is yes."
        r = parse_react(text)
        assert r.kept == text


# ─────────────────── the example you show the model must parse ─────────────

def test_action_example_in_error_messages_is_itself_valid():
    """Small models copy examples. If the example in your error message doesn't parse,
    every repair teaches the model the wrong format."""
    r = parse_react(ACTION_EX)
    assert isinstance(r, Action)
    assert r.tool == "search"
    assert r.arg == "John Doe"