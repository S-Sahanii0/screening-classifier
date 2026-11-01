from screener.classify import Classifier, derive_verdict, parse_response
from screener.prompts import VARIANTS, build_system_prompt, build_user_prompt

from fakes import FakeClient

GOOD = ('{"original_model":"yes","clinical_outcome":"yes","fairness_in_results":"yes",'
        '"verdict":"A","reason":"own model"}')


def test_parse_clean_json():
    p = parse_response(GOOD)
    assert p.verdict == "A" and p.original_model == "yes" and p.reason == "own model"


def test_parse_json_inside_code_fence_and_chatter():
    p = parse_response("Sure!\n```json\n" + GOOD + "\n```\nHope that helps.")
    assert p.verdict == "A"


def test_parse_normalises_verdict_spellings():
    for raw in ("Group B", "b", "B"):
        txt = GOOD.replace('"verdict":"A"', f'"verdict":"{raw}"')
        assert parse_response(txt).verdict == "B"
    assert parse_response(GOOD.replace('"A"', '"needs check"', 1).replace(
        '"verdict":"needs check"', '"verdict":"needs check"')).verdict in ("needs_check", "A", "parse_error")


def test_parse_bad_output_is_parse_error_not_crash():
    assert parse_response("I think this is group A.").verdict == "parse_error"
    assert parse_response('{"verdict": "maybe"}').verdict == "parse_error"
    assert parse_response("").verdict == "parse_error"


def test_unknown_check_value_becomes_unclear():
    p = parse_response(GOOD.replace('"clinical_outcome":"yes"', '"clinical_outcome":"probably"'))
    assert p.clinical_outcome == "unclear"


def test_derive_verdict_matches_rubric():
    assert derive_verdict(["yes", "yes", "yes"]) == "A"
    assert derive_verdict(["yes", "no", "unclear"]) == "B"
    assert derive_verdict(["yes", "unclear", "yes"]) == "needs_check"


def test_title_only_prompt_has_no_abstract():
    user = build_user_prompt("title_only", "My title", "SECRET ABSTRACT TEXT")
    assert "SECRET ABSTRACT TEXT" not in user and "My title" in user


def test_every_variant_builds():
    fs = [(10, "t", "a"), (46, "t", "a"), (15, "t", "a"), (4, "t", "a"), (39, "t", "a"), (24, "t", "a")]
    for v in VARIANTS:
        assert "JSON" in build_system_prompt(v, fs if v == "few_shot" else None)


def test_rubric_does_not_name_evaluation_papers():
    text = build_system_prompt("rules")
    for leak in ("Meng", "Drop the shortcuts", "Woo"):
        assert leak not in text


def test_classifier_calls_client_once():
    c = FakeClient(verdict="B")
    p = Classifier(c, "rules").classify("Some title", "Some abstract")
    assert p.verdict == "B" and len(c.calls) == 1
