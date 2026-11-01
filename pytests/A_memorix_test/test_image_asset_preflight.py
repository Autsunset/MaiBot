from PIL import Image

import io

import pytest

from src.A_memorix.core.image.asset_store import ImageAssetStore, ImageAssetValidationError


def payload(fmt="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (2, 2)).save(output, format=fmt)
    return output.getvalue()


@pytest.mark.parametrize(
    ("data", "limits", "code"),
    [(b"", {}, "empty_image"), (b"invalid", {}, "invalid_image"),
     (payload("GIF"), {}, "unsupported_format"), (payload(), {"max_bytes": 1}, "byte_limit"),
     (payload(), {"max_pixels": 1}, "pixel_limit")],
)
def test_preflight_reports_typed_permanent_errors(data, limits, code):
    settings = {"max_bytes": 1024, "max_pixels": 1024, **limits}
    with pytest.raises(ImageAssetValidationError) as raised:
        ImageAssetStore.inspect_payload(data, **settings)
    assert raised.value.code == code


def test_shared_preflight_does_not_publish_or_convert_original(tmp_path):
    original = payload()
    inspected = ImageAssetStore.inspect_payload(original, max_bytes=1024, max_pixels=1024)
    assert (inspected.width, inspected.height, inspected.mime_type) == (2, 2, "image/png")
    assert list(tmp_path.iterdir()) == []
    assert original == payload()
