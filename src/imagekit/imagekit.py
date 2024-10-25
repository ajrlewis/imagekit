from collections import Counter
import colorsys
from io import BytesIO
import os

from loguru import logger
import numpy as np

from PIL import Image as ImageLib, ImageChops, ImageDraw, ImageFont
from PIL.Image import Image
from pillow_heif import register_heif_opener
import PyPDF2
import pytesseract
from qrcode import QRCode
from qrcode.constants import ERROR_CORRECT_L
import scipy.ndimage as ndimage

ASSETS_PATH = f"{os.path.dirname(os.path.realpath(__file__))}/assets/"

register_heif_opener()

# imagekit/colors.py
Color = tuple[int]
Colors = list[Color]

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
NOSTR_PURPLE = (169, 21, 255)


def rgb_to_hex(rgb: Color = (0, 0, 0)) -> str:
    return "#%02x%02x%02x" % rgb


def most_frequent_colors(img: Image, top: int = -1) -> Colors:
    width, height = img.size
    colors = img.getcolors(width * height)
    frequencies = sorted(colors, key=lambda x: x[0], reverse=True)
    return [f[1] for f in frequencies][:top]


def make_most_common_color_transparent(img: Image) -> Image:
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


def resize(img: Image, width: int, height: int) -> Image:
    logger.debug(f"Resizing image to {width = } {height = }")
    resized_img = img.resize((width, height))
    return resized_img


def shrink(img: Image, factor: float = 0.5) -> Image:
    width, height = img.size
    new_width, new_height = int(round(factor * width)), int(round(factor * height))
    resized_img = resize(img, new_width, new_height)
    return resized_img


def compress(img: Image, max_size: int = 1024) -> Image:
    compressed_img = img.copy()
    compressed_img.thumbnail((max_size, max_size))
    return compressed_img


def favicon(img: Image) -> Image:
    favicon_img = resize(img, 16, 16)
    return favicon_img


def pad(img: Image, padding: int = 100) -> Image:
    padded_size = (img.size[0] + 2 * padding, img.size[1] + 2 * padding)
    padded_img = ImageLib.new("RGBA", padded_size, WHITE)
    position = (padding, padding)
    padded_img.paste(img, position)
    return padded_img


def cirularize(img: Image) -> Image:
    width, height = img.size[:2]
    size = (min([width, height]), min([width, height]))
    img = img.resize(size)
    bigsize = (img.size[0] * 3, img.size[1] * 3)
    mask = ImageLib.new("L", bigsize, 0)
    ImageDraw.Draw(mask).ellipse((0, 0) + bigsize, fill=255)
    mask = mask.resize(img.size, ImageLib.Resampling.LANCZOS)
    mask = ImageChops.darker(mask, img.split()[-1])
    img.putalpha(mask)
    return img


def convert(input_path: str, output_path: str):
    img = ImageLib.open(input_path)
    img = img.convert("RGBA")
    img.save(output_path)
    return output_path


def replace_colors(img: Image, source_colors: Colors, target_colors: Colors) -> Image:
    recolored_img_array = np.array(img)
    for source_color, target_color in zip(source_colors, target_colors):
        r, g, b = source_color[:3]
        mask = (
            (recolored_img_array[:, :, 0] == r)
            & (recolored_img_array[:, :, 1] == g)
            & (recolored_img_array[:, :, 2] == b)
        )
        recolored_img_array[mask] = target_color
    recolored_img = Image.fromarray(recolored_img_array)
    return recolored_img


def extract_text(img: Image) -> str:
    smoothed_img = smooth(img, 1.0)
    text = pytesseract.image_to_string(smoothed_img)
    texts = [t for t in text.split("\n") if t]  # Remove blank text
    text = "\n".join(texts)
    return text


def smooth(img: Image, sigma: float) -> Image:
    smoothed_img = ndimage.gaussian_filter(
        img, sigma=(sigma, sigma, 0), mode="reflect", order=0
    )
    smoothed_img = ImageLib.fromarray(smoothed_img)
    return smoothed_img


def qrcode(
    data: str,
    box_size: int = 10,
    border: int = 0,
    fill_color: str = "#000000",
    back_color: str = "#ffffff",
) -> Image:
    """
    https://{domain}
    nostr:{npub}
    btc:{address}?amount={amount}
    WIFI:T:WPA;S:{ssid};P:{password};H:true;"
    """
    qr = QRCode(
        version=1,
        error_correction=ERROR_CORRECT_L,
        box_size=box_size,
        border=border,
    )

    qr.add_data(data)
    qr.make(fit=True)

    qr_image = qr.make_image(fill_color=fill_color, back_color=back_color)
    qr_image = qr_image.convert("RGBA")

    return qr_image


def load(data: bytes) -> Image:
    img = ImageLib.open(BytesIO(data))
    img = img.convert("RGBA")
    return img


def to_bytes(img: Image) -> bytes:
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


def read(filepath: str) -> Image:
    img = ImageLib.open(filepath)
    img = img.convert("RGBA")
    return img


def read_pdf(filepath: str) -> str:
    reader = PyPDF2.PdfReader(filepath)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        texts.append(text)
    text = "\n".join(texts)
    return text


def save(img: Image, filepath: str, quality: int = 80):
    img.save(filepath, optimize=True, quality=quality)


def create(width: int, height: int, color: Color = WHITE) -> Image:
    """Creates and returns an image.

    Args:
        width, height: The width and height of the image in pixels.
        color: The background color of the image.
    """
    img = ImageLib.new(mode="RGB", size=(width, height), color=color)
    return img


def card(width: int = 85, height: int = 55, dpi: int = 300, **kwargs) -> tuple[Image]:
    """
    Args:
        width, height: The width and height of the image in millimeters.
        dpi: The dots per inch (pixel density).
    """
    mm_to_px = dpi / 25.4  # 1 inch = 25.4 millimeters.
    width_px, height_px = int(round(width * mm_to_px)), int(round(height * mm_to_px))
    front = create(width=width_px, height=height_px, **kwargs)
    back = front.copy()
    return front, back


def nostr_card(npub: str):
    front, back = card(color=NOSTR_PURPLE)
    width, height = front.size

    # Logo on front

    logo = read(f"{ASSETS_PATH}/nostr-logo-with-text.png")
    logo_width, logo_height = logo.size
    scale_factor = height * 0.8 / logo_height
    logo = shrink(logo, scale_factor)
    logo_width, logo_height = logo.size
    origin = (width - logo_width) // 2, (height - logo_height) // 2
    front.paste(logo, origin, mask=logo)  # Use alpha channel of logo for mask
    qr = qrcode(f"nostr:{npub}", fill_color=WHITE, back_color=NOSTR_PURPLE)
    scale_factor = height * 0.6 / qr.height
    qr = shrink(qr, scale_factor)
    qr_width, qr_height = qr.size
    origin = (width - qr_width) // 2, int(round((height - qr_height) / 2.5))
    logger.debug(f"{origin = }")
    back.paste(qr, origin, mask=qr)  # Use alpha channel of logo for mask

    # QRCode and npub on back

    font_name = f"{ASSETS_PATH}/Ubuntu-Regular.ttf"
    font_size = 26
    font = ImageFont.truetype(font_name, size=font_size)

    bbox = font.getbbox(npub)  # (left, top, right, bottom)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[1] - bbox[3]
    text_origin = (width - text_width) // 2, origin[1] + int(round(1.1 * qr_height))
    draw = ImageDraw.Draw(back)
    draw.text(text_origin, npub, font=font, fill=WHITE)

    return front, back
