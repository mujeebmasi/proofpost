from proofpost.checks.grounding import check_grounding, numbers_in
from proofpost.models import Claim, Draft, DraftLine


def claim(cid="C1", text="Classification took about 11 minutes.", quote="about eleven minutes per shipment"):
    return Claim(id=cid, text=text, quote=quote, passage_id="p-1", source="call.md")


def draft(*lines):
    return Draft(lines=[DraftLine(text=t, kind=k, claim_ids=ids) for t, k, ids in lines])


def codes(issues):
    return [i.code for i in issues]


def test_numbers_normalise_digits_commas_and_words():
    assert numbers_in("1,200 shipments, eleven people, 3x, 40%") == {"1200", "11", "3", "40"}
    assert numbers_in("B2B and no one cares") == set()  # "B2B" and "one" are not numbers


def test_number_from_the_cited_claim_is_allowed():
    d = draft(("it took about 11 minutes.", "claim", ["C1"]))
    assert check_grounding(d, [claim()], [], "Maya") == []


def test_invented_number_is_an_error():
    d = draft(("it took 11 minutes, an 80% cut.", "claim", ["C1"]))
    issues = check_grounding(d, [claim()], [], "Maya")
    assert codes(issues) == ["unsupported_number"]
    assert "80" in issues[0].message


def test_claim_line_without_citation_is_an_error():
    d = draft(("customers love it.", "claim", []))
    assert codes(check_grounding(d, [claim()], [], "Maya")) == ["uncited_claim"]


def test_number_without_citation_is_an_error_even_in_an_opinion():
    d = draft(("i think 90 percent of this is boring.", "opinion", []))
    assert "uncited_claim" in codes(check_grounding(d, [claim()], [], "Maya"))


def test_citing_a_claim_that_does_not_exist_is_an_error():
    d = draft(("it took 11 minutes.", "claim", ["C9"]))
    assert "unknown_claim" in codes(check_grounding(d, [claim()], [], "Maya"))


def test_opinion_and_question_need_no_citation():
    d = draft(("i think boring work matters.", "opinion", []), ("what do you think?", "question", []))
    assert check_grounding(d, [claim()], [], "Maya") == []


def test_unknown_company_name_is_a_warning():
    from proofpost.models import Passage

    passages = [Passage(id="p-1", source="call.md", text="we spoke to Harbor Freight about filing")]
    d = draft(("we spoke to Acme about filing.", "opinion", []))
    issues = check_grounding(d, [claim()], passages, "Maya")
    assert [(i.severity, i.code) for i in issues] == [("warning", "unknown_name")]
    d_ok = draft(("we spoke to Harbor about filing.", "opinion", []))
    assert check_grounding(d_ok, [claim()], passages, "Maya") == []
