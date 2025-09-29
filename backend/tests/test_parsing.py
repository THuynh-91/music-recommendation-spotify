import pytest

from app.spotify.parsing import parse_spotify_url


@pytest.mark.parametrize(
    "value, expected_type",
    [
        ("https://open.spotify.com/track/2TpxZ7JUBn3uw46aR7qd6V", "track"),
        ("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M", "playlist"),
        ("spotify:track:2TpxZ7JUBn3uw46aR7qd6V", "track"),
        ("spotify:playlist:37i9dQZF1DXcBWIGoYBM5M", "playlist"),
    ],
)
def test_parse_spotify_url(value, expected_type):
    entity = parse_spotify_url(value)
    assert entity.kind == expected_type


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com",
        "spotify:album:123",
        "",
    ],
)
def test_parse_invalid_url(value):
    with pytest.raises(ValueError):
        parse_spotify_url(value)
