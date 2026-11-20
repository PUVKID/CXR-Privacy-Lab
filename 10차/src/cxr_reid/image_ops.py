"""흉부 X선 이미지에 원본·블러·모자이크 변환을 적용한다."""

from PIL import Image, ImageFilter


def process_image(image: Image.Image, mode: str, strength: int | float) -> Image.Image:
    """원본을 변경하지 않고 지정한 개인정보 보호 변환을 반환한다.

    블러 수준은 PIL GaussianBlur의 반지름(픽셀)이며, 모자이크 수준은
    원본 좌표계에서 하나의 타일이 차지하는 가로·세로 길이(픽셀)다.
    """
    if mode == "original" or (mode == "blur" and float(strength) == 0):
        return image.copy()
    if mode == "blur":
        if strength < 0:
            raise ValueError("블러 강도는 0 이상이어야 합니다.")
        return image.filter(ImageFilter.GaussianBlur(radius=float(strength)))
    if mode == "mosaic":
        tile_size = int(strength)
        if tile_size < 1:
            raise ValueError("모자이크 블록 크기는 1 이상이어야 합니다.")
        if tile_size == 1:
            return image.copy()
        w, h = image.size
        new_size = (max(1, (w + tile_size - 1) // tile_size),
                    max(1, (h + tile_size - 1) // tile_size))
        return image.resize(new_size, Image.Resampling.BOX).resize((w, h), Image.Resampling.NEAREST)
    raise ValueError(f"알 수 없는 실험 종류: {mode}")
