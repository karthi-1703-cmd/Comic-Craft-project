import uuid
from pathlib import Path

from huggingface_hub import InferenceClient


class ImageGenerator:
    """Text-to-image generation using Hugging Face."""

    def __init__(
        self,
        token: str,
        model: str,
        output_dir: Path,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
    ):

        self.token = token
        self.model = model
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance_scale = guidance_scale

        self.client = None

        if token:

            self.client = InferenceClient(
                api_key=token,
                provider="auto",
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _require_client(self):

        if self.client is None:

            raise RuntimeError(
                "HF_TOKEN is missing. "
                "Add your Hugging Face token to .env "
                "and restart the server."
            )

        return self.client

    def generate_image(
        self,
        prompt: str,
        panel_number: int,
    ):

        client = self._require_client()

        final_prompt = (
            f"{prompt}. "
            "Professional sequential comic illustration, "
            "expressive character, cinematic composition, "
            "detailed environment, polished linework, rich color, "
            "consistent character design, no written words, "
            "no watermark, no logo."
        )

        image = client.text_to_image(
            final_prompt,
            model=self.model,
            width=self.width,
            height=self.height,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            negative_prompt=(
                "blurry, distorted anatomy, duplicate characters, "
                "text, watermark, logo"
            ),
        )

        filename = (
            f"panel_{panel_number}_"
            f"{uuid.uuid4().hex[:10]}.png"
        )

        path = self.output_dir / filename

        image.save(path)

        return f"/static/panels/{filename}"