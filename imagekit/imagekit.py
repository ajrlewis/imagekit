from collections import Counter
import colorsys
from io import BytesIO
import numpy as np
from PIL import ImageDraw
from PIL.Image import Image
from pillow_heif import register_heif_opener
import pytesseract
import scipy.ndimage as ndimage

register_heif_opener()

Color = tuple[int]
Colors = list[Color]


def resize(img: Image, width: int, height: int) -> Image:
    resized_image = img.resize((width, height))
    return resized_image


def size(img: Image) -> tuple[int]:
    return img.size


def favicon(img: Image) -> Image:
    resized_img = resize(img, 16, 16)
    return resized_img


def shrink(img: Image, factor: float = 0.5) -> Image:
    width, height = size(image)
    new_width, new_height = int(round(factor * width)), int(round(factor * height))
    resized_image = resize(img, new_width, new_height)
    return resized_image


def compress(img: Image, max_size: int = 1024) -> Image:
    compressed_img = img.copy()
    compressed_img.thumbnail((max_size, max_size))
    return compressed_img


def add_whitespace(img: Image, padding: int = 100) -> Image:
    padded_size = (img.size[0] + 2 * padding, img.size[1] + 2 * padding)
    padded_img = Image.new("RGBA", padded_size, (255, 255, 255))
    position = (padding, padding)
    padded_img.paste(img, position)
    return padded_img


def cirularize(img: Image) -> Image:
    width, height = img.size
    alpha = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([0, 0, height, width], 0, 360, fill=255)
    np_alpha = np.array(alpha)
    np_img = np.array(img)
    np_img = np.dstack((np_img, np_alpha))
    return Image.fromarray(np_img)


def most_frequent_colors(img: Image, top: int = 5) -> Colors:
    width, height = img.size
    colors = img.getcolors(width * height)
    frequencies = sorted(colors, key=lambda x: x[0], reverse=True)
    return [f[1] for f in frequencies][:top]


def make_most_common_color_transparent(img):
    color_count = Counter(img.getdata())
    most_common_color = color_count.most_common(1)[0][0]
    transparent_img = img.convert("RGBA")
    data = transparent_img.getdata()
    new_data = []
    for item in data:
        if item == most_common_color:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)
    transparent_img.putdata(new_data)
    return transparent_img


def convert_heic_to_jpg_or_png(input_path, output_path, output_format):
    image = Image.open(input_path)
    image.save(output_path)
    return output_path


def replace_colors(img: Image, source_colors: Colors, target_colors: Colors) -> Image:
    image_array = np.array(img)
    for source_color, target_color in zip(source_colors, target_colors):
        r, g, b = source_color
        mask = (
            (image_array[:, :, 0] == r)
            & (image_array[:, :, 1] == g)
            & (image_array[:, :, 2] == b)
        )
        image_array[mask] = target_color
    recolored_img = Image.fromarray(image_array)
    return recolored_img


def extract_text_from_image(img: Image) -> list[str]:
    # text = pytesseract.image_to_string(img)
    smoothed_img = smooth(img, 1.0)
    text = pytesseract.image_to_string(smoothed_img)
    rows = [t for t in text.split("\n") if t]
    return rows


def smooth(img: Image, sigma: float) -> Image:
    smoothed_img = ndimage.gaussian_filter(
        img, sigma=(sigma, sigma, 0), mode="reflect", order=0
    )
    smoothed_img = Image.fromarray(smoothed_img)
    return smoothed_img


def load(data: bytes) -> Image:
    img = Image.open(BytesIO(data))
    img = img.convert("RGBA")
    return img


def read(filepath: str) -> Image:
    img = Image.open(filepath)
    img = img.convert("RGBA")
    return img


def write(img: Image, filepath: str, quality: int = 80):
    img.save(filepath, optimize=True, quality=quality)


def to_bytes(img: Image) -> bytes:
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


def rgb_to_hex(rgb: Color = (0, 0, 0)) -> str:
    return "#%02x%02x%02x" % rgb
