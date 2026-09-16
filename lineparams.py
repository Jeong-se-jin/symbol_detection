"""Thresholds as multiples of the drawing's own character height.

Every number that follows is in units of ref_h, the median height of an OCR
word. A sheet rendered at twice the resolution has twice the character height
and twice every threshold, so the same rules hold without retuning. The idea
and most of the multipliers come from the v7 bundle's params.py.
"""
from dataclasses import dataclass, asdict, field
from typing import Tuple

import statistics


# The sheet the drawing thresholds were fitted on. Everything geometric is
# expressed relative to it, because the drawing does not change size when the
# OCR does.
REFERENCE_SPAN = 3910.0
REFERENCE_DRAW_H = 14.0
# On the three raster sheets measured, a stroke came out 2.36 to 2.68px wide
# with a drawing scale of 13.5 to 14.9 -- about 5.6 times the stroke. Stroke
# width is the better anchor: it says how big the drawing is, where the sheet
# size only says how much paper it was put on. Rendering a vector sheet at a
# higher dpi grows the paper and the strokes together, but cropping to the ink
# or laying the same drawing out on a wider sheet grows only one of them.
DRAW_H_PER_STROKE = 5.6

# Better still: a symbol drawn to a standard size. An instrument bubble is a
# fixed circle on every P&ID, so its size in pixels *is* the drawing's scale,
# where stroke width only says how wide the pen was. Measured against the
# draw_h each sheet actually needed:
#
#     sheet   needed   bubble x 0.215   globe valve x 0.323
#     p59      13.2      12.9 (n=51)      13.2 (n=33)
#     p55      13.9      14.4 (n=14)      14.5 (n=17)
#     p56      15.0      14.2 (n=21)      14.7 (n=6)
#     p65      20.8      21.3 (n=50)      20.7 (n=16)
#
# The stroke rule gets p65 wrong by a third: that sheet is the same drawing
# drawn half again larger with the same pen, and the pen is all the stroke
# sees. Classes left out on purpose -- Flange_or_Nozzle, Reducer and box came
# out at 0.53/0.56, 0.32/0.44 and 0.22/0.30 on the same two sheets, because
# what the detector calls them is not one standard shape.
SYMBOL_ANCHORS = {
    'Instrument_Field': 0.215,
    'Globle_valve_NC': 0.323,
    'Check_valve': 0.310,
}
MIN_ANCHOR_COUNT = 5


@dataclass
class Params:
    ref_h: float
    draw_h: float = 0.0

    # text
    text_max_h: float = 0.0          # taller than this is not a character
    merge_gap: float = 0.0           # gap two words may leave and still be one line
    row_slack: float = 0.0           # how far off a shared baseline a word may sit

    # lines
    line_min_len: float = 0.0        # a run shorter than this is not pipe
    probe: float = 0.0               # length of the orientation probe
    bridge_gap: float = 0.0          # rejoin runs whose ends are this close
    endpoint_radius: float = 0.0     # a contact this near an end belongs to it

    # symbols
    sym_min: float = 0.0
    sym_max: float = 0.0
    big_shape: float = 0.0           # bigger than this is equipment, not an instrument
    enclosure_min_side: float = 0.0

    # mapping
    touch: float = 0.0               # how far outside a box to look for ink
    tag_near: float = 0.0            # a tag further than this belongs to nothing
    align_penalty: float = 0.0       # cost of a tag that is not squared up
    label_near: float = 0.0          # a line label further than this belongs to nothing

    @classmethod
    def from_text_height(cls, ref_h: float, draw_h: float = None) -> 'Params':
        """Two scales, not one.

        ref_h is the median OCR box height and drives everything about text:
        how far apart two words may be and still be one label, how far outside
        its box a label reaches. draw_h is the drawing's own scale and drives
        everything geometric: how long a pipe must be, how big an arrowhead is.

        They have to be separate because an OCR box is not a fixed multiple of
        what it contains. On a crisp 1-bit scan PaddleOCR returns boxes 19px
        tall where the same drawing rendered at 230dpi gives 14px -- the letters
        are the same size, the boxes are not. Driving the geometry off ref_h
        then inflates every threshold by a third, and the arrowhead area test
        (0.15 x h squared) rejects almost every arrowhead.
        """
        h = float(ref_h)
        g = float(draw_h if draw_h is not None else REFERENCE_DRAW_H)
        return cls(
            ref_h=h, draw_h=g,
            text_max_h=1.7 * h, merge_gap=0.9 * h, row_slack=0.6 * h,
            line_min_len=3.3 * g, probe=1.3 * g, bridge_gap=0.6 * g,
            endpoint_radius=0.8 * g,
            sym_min=1.0 * g, sym_max=18.0 * g, big_shape=11.5 * g,
            enclosure_min_side=1.2 * g,
            touch=0.7 * h, tag_near=3.7 * h, align_penalty=2.1 * h,
            label_near=2.5 * h,
        )

    @staticmethod
    def drawing_scale(width: int, height: int, stroke: float = None) -> float:
        """The drawing's own scale.

        From the stroke width when it is known, because that is what actually
        tracks how big the drawing is. The sheet size is the fallback, and it
        misleads: the same drawing rendered from vector at 400dpi came out
        6075px wide against 4166 for a raster sheet, which put the geometry
        thresholds half again too high -- 72px of pipe minimum instead of 49,
        an arrowhead area floor of 71px instead of 33, and one arrowhead found
        on the whole sheet instead of twenty.
        """
        if stroke and stroke > 0:
            return DRAW_H_PER_STROKE * float(stroke)
        return REFERENCE_DRAW_H * max(width, height) / REFERENCE_SPAN

    @staticmethod
    def scale_from_symbols(detections):
        """draw_h from the symbols drawn to a standard size, or None.

        detections: an iterable of (class name, long side in px). Only the
        classes in SYMBOL_ANCHORS count, and only once at least
        MIN_ANCHOR_COUNT of one were found, since a median over three boxes is
        noise. Where more than one class qualifies they are averaged by count.
        """
        import collections
        seen = collections.defaultdict(list)
        for name, span in detections:
            if name in SYMBOL_ANCHORS and span > 0:
                seen[name].append(float(span))
        total = weight = 0.0
        for name, spans in seen.items():
            if len(spans) < MIN_ANCHOR_COUNT:
                continue
            total += SYMBOL_ANCHORS[name] * statistics.median(spans) * len(spans)
            weight += len(spans)
        return total / weight if weight else None

    @staticmethod
    def stroke_width(ink) -> float:
        """Mean stroke width: total ink over skeleton length."""
        import cv2
        import numpy as np
        binary = (ink > 0).astype(np.uint8) * 255
        total = int((binary > 0).sum())
        if not total:
            return 0.0
        bones = int((cv2.ximgproc.thinning(binary) > 0).sum())
        return total / bones if bones else 0.0

    @staticmethod
    def estimate_text_height(heights, default=12.0) -> float:
        """Median word height, over words tall enough to be text at all.

        The median, not the mean: a title block or a symbol misread as a word
        would drag a mean up, and there are always a few of those.
        """
        usable = [h for h in heights if 5 <= h <= 40]
        if len(usable) < 20:
            return float(default)
        return float(statistics.median(usable))

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}
