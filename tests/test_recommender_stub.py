from app.services.recommender_stub import get_recommendations


def test_phone_request_does_not_match_headphones():
    recs = get_recommendations(1, "recommend me a phone")
    assert recs, "expected at least one recommendation"
    assert not any("headphone" in r["name"].lower() for r in recs)
    assert any("phone" in r["name"].lower() for r in recs)


def test_plural_keyword_matches_singular_product():
    recs = get_recommendations(1, "recommend me phones")
    assert any("iphone" in r["name"].lower() for r in recs)


def test_every_recommendation_is_a_catalog_product():
    for r in get_recommendations(1, "recommend me a phone"):
        assert r["id"] in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        assert r["currency"] == "AED"
        assert r["reason"].startswith("Matches your request for")


def test_short_keyword_does_not_match_as_a_prefix():
    # "son" is a prefix of "Sony": a gift for a son must not return Sony products.
    assert get_recommendations(1, "gift for my son") == []


def test_short_keyword_still_matches_a_whole_word():
    # "ps5" (3 characters) equals a product word, so it still matches.
    recs = get_recommendations(1, "recommend a ps5 controller")
    assert [r["id"] for r in recs][:1] == [2]


def test_perfume_example_still_returns_both_perfumes():
    recs = get_recommendations(1, "Can you recommend a perfume?")
    assert sorted(r["id"] for r in recs) == [4, 7]


def test_bad_catalog_does_not_crash(monkeypatch):
    from app.services import recommender_stub

    monkeypatch.setattr(recommender_stub, "_load_products", lambda: [{"id": "bad"}])
    assert get_recommendations(1, "recommend me a phone") == []