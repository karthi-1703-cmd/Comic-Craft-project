from fastapi.testclient import TestClient

from app.main import app
from app.schemas import PromptRequest


def test_home_and_health():

    client = TestClient(app)

    home = client.get("/")

    assert home.status_code == 200
    assert 'name="tone"' in home.text
    assert 'name="art_style"' in home.text
    assert 'maxlength="300"' in home.text

    health = client.get("/health")

    assert health.status_code == 200

    assert (
        health.json()["status"]
        == "ok"
    )


def test_prompt_request_accepts_descriptive_art_style():

    request = PromptRequest(
        story_prompt="A hero crosses a stormy sea.",
        character_name="Miriam",
        setting=(
            "A magical forest at night, with ancient trees and glowing flowers "
            "surrounding a small stone cottage overlooking the forest"
        ),
        tone="Epic",
        art_style=(
            "Cinematic biblical comic illustration, dramatic inking, "
            "rich colors, realistic stormy sea"
        ),
    )

    assert len(request.art_style) > 80
    assert len(request.setting) > 120