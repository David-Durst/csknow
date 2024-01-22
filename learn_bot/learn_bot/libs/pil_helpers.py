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


def concat_horizontal_vertical_with_extra(row0: List[Image.Image], row1: List[Image.Image],
                                          extra_height: int, extra_width_spacing: int,
                                          extra_height_spacing: int, last_extra_removal) -> Image.Image:
    assert(len(row0) == len(row1))
    for i in range(len(row0)):
        assert row0[i].width == row0[0].width
        assert row0[i].height == row0[0].height
        assert row1[i].width == row0[0].width
        assert row1[i].height == row0[0].height

    dst = Image.new('RGB', ((row0[0].width + extra_width_spacing) * len(row0) - last_extra_removal, row0[0].height * 2 + extra_height - last_extra_removal),
                    (255, 255, 255))
    offset = 0
    for i in range(len(row0)):
        dst.paste(row0[i], (offset, 0))
        dst.paste(row1[i], (offset, row0[0].height + extra_height_spacing))
        offset += row0[0].width + extra_width_spacing
    return dst


def repeated_paste_horizontal(dst: Image.Image, ims: List[Image.Image], base_offset_x: int, base_offset_y: int,
                              repeat_offset_x: int):
    offset_x = base_offset_x
    for im in ims:
        dst.paste(im, (offset_x, base_offset_y))
        offset_x += repeat_offset_x
    return dst
