from typing import List

from PIL import Image


def concat_horizontal(ims: List[Image.Image]) -> Image.Image:
    dst = Image.new('RGB', (sum([im.width for im in ims]), ims[0].height))
    offset = 0
    for im in ims:
        dst.paste(im, (offset, 0))
        offset += im.width
    return dst


def concat_vertical(ims: List[Image.Image]) -> Image.Image:
    dst = Image.new('RGB', (ims[0].width, sum([im.height for im in ims])))
    offset = 0
    for im in ims:
        dst.paste(im, (0, offset))
        offset += im.height
    return dst
