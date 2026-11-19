#!/usr/bin/env python3
"""Core routines for PCB AOI Visual Comparator.

Key ideas:
- load scanner image or PDF page;
- crop PCB from white background;
- detect large drilled holes;
- register Excellon/Gerber to scan using affine or projective transform;
- automatic mirrored-side detection;
- supersampled Gerber rendering to reduce stair-step artefacts;
- operator-editable anchor control points;
- hole-rim / through-hole parallax diagnostics;
- scan-derived copper evidence and review regions.

The review layer is an operator aid, not an automatic acceptance decision.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except Exception:  # optional if PDFs are not used
    fitz = None


def imread_unicode(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def _rotate_for_pdf_page(img: np.ndarray, rotation: int) -> np.ndarray:
    """Apply the PDF page rotation to an extracted scanner raster."""
    r = int(rotation) % 360
    if r == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if r == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _normalize_analysis_size(img: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, float]:
    """Use a stable working resolution for registration.

    Normalising large scanner/PDF rasters to a stable working scale makes
    hole/rim thresholds more reproducible and also suppresses
    fine white silkscreen edges that can otherwise dominate the physical drill rim.
    """
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    scale = float(max_side) / float(m)
    out = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                     interpolation=cv2.INTER_AREA)
    return out, scale


def load_scan(path: str | Path, pdf_page: int = 0, pdf_dpi: int = 150,
              normalize_max_side: int = 2200, return_meta: bool = False):
    """Load a scanner raster or PDF page.

    Rasterising each PDF page at a fixed DPI keeps its page rotation and
    resampling in a reproducible coordinate frame. Direct embedded-image
    extraction can use different page geometry and lead to registration errors.
    Very large rasters are downscaled only when exceeding normalize_max_side.
    """
    path = Path(path)
    meta = {"source_mode": "raster", "native_shape": None, "analysis_scale": 1.0}
    if path.suffix.lower() == ".pdf":
        if fitz is None:
            raise RuntimeError("PDF support requires PyMuPDF (pip install pymupdf)")
        doc = fitz.open(str(path))
        if not 0 <= pdf_page < len(doc):
            raise RuntimeError(f"PDF page {pdf_page + 1} is outside 1..{len(doc)}")
        page = doc[pdf_page]
        scale = float(pdf_dpi) / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        meta["source_mode"] = f"pdf-rendered-{int(pdf_dpi)}dpi"
        doc.close()
    else:
        img = imread_unicode(path)

    meta["native_shape"] = tuple(int(x) for x in img.shape[:2])
    img, scale = _normalize_analysis_size(img, int(normalize_max_side))
    meta["analysis_scale"] = float(scale)
    meta["analysis_shape"] = tuple(int(x) for x in img.shape[:2])
    return (img, meta) if return_meta else img

def pdf_page_count(path: str | Path) -> int:
    path = Path(path)
    if path.suffix.lower() != ".pdf" or fitz is None:
        return 1
    with fitz.open(str(path)) as doc:
        return len(doc)


def pcb_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    thr = max(35, int(np.percentile(sat, 45)))
    m = (sat > thr).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("PCB not found against the scanner background")
    c = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(m)
    cv2.drawContours(out, [c], -1, 255, -1)
    k = max(5, int(round(max(img.shape[:2]) * 0.004)) | 1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return out


def board_bbox(mask: np.ndarray, margin: int = 16) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise RuntimeError("Empty PCB mask")
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(mask.shape[1], int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(mask.shape[0], int(ys.max()) + margin + 1)
    return x0, y0, x1, y1


def _gerber_reference_bounds_mm(path: str | Path | None) -> tuple[float, float, float, float] | None:
    """Return rough Gerber XY bounds in millimetres.

    The bounds are primarily a drill-decoding sanity reference.  Some Altium
    ASCII NC Drill files declare ``METRIC,LZ`` but shorten coordinates on the
    *right* (for example X00525 means 52.5 mm in a 4:4 file).  The nominal
    zero-suppression flag alone is therefore not sufficient; comparing both
    fixed-point interpretations with the copper bounds makes the choice robust.
    """
    if not path:
        return None
    try:
        _aps, ops = parse_gerber(path)
    except Exception:
        return None
    pts = []
    for typ, data, _ap in ops:
        base = typ.replace("_clear", "")
        if base == "line":
            pts.extend([data[0], data[1]])
        elif base == "flash":
            pts.append(data)
        elif base == "region":
            for contour in data:
                pts.extend(contour)
    if len(pts) < 2:
        return None
    a = np.asarray(pts, np.float64)
    if not np.all(np.isfinite(a)):
        a = a[np.all(np.isfinite(a), axis=1)]
    if len(a) < 2:
        return None
    xmin, ymin = np.min(a, axis=0)
    xmax, ymax = np.max(a, axis=0)
    if xmax - xmin <= 1e-6 and ymax - ymin <= 1e-6:
        return None
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _gerber_reference_span_mm(path: str | Path | None) -> float | None:
    """Return a rough maximum XY span of a Gerber in millimetres."""
    b = _gerber_reference_bounds_mm(path)
    if b is None:
        return None
    xmin, ymin, xmax, ymax = b
    span = max(float(xmax - xmin), float(ymax - ymin))
    return span if np.isfinite(span) and span > 1e-6 else None


def _excellon_format_from_header(lines: list[str]):
    """Extract unit / zero suppression / fixed-point format from Excellon header."""
    units = "MM"
    factor = 1.0
    zero = "LZ"  # CAM350 'Leading' zero suppression
    fmt = None
    absolute = True
    for raw in lines:
        u = raw.strip().upper()
        if not u:
            continue
        if "METRIC" in u or u == "M71":
            units, factor = "MM", 1.0
        elif "INCH" in u or u == "M72":
            units, factor = "INCH", 25.4
        if "LZ" in u or "LEADING" in u:
            zero = "LZ"
        elif "TZ" in u or "TRAILING" in u:
            zero = "TZ"
        if u.startswith("G90"):
            absolute = True
        elif u.startswith("G91"):
            absolute = False
        # Common comments: ;FILE_FORMAT=3:2, FORMAT=3.2, FORMAT 3:2
        m = re.search(r"(?:FILE[_ ]?FORMAT|FORMAT)\s*[=: ]\s*(\d+)\s*[:X.]\s*(\d+)", u)
        if m:
            fmt = (int(m.group(1)), int(m.group(2)))
        # Header style: METRIC,LZ,000.000 or INCH,TZ,00.0000
        m = re.search(r"(?:METRIC|INCH)[^0-9]*([0#]+)\.([0#]+)", u)
        if m:
            fmt = (len(m.group(1)), len(m.group(2)))
    return units, factor, zero, fmt, absolute


def _excellon_infer_decimals(raw_coord_tokens: list[str], factor: float,
                              reference_span_mm: float | None) -> int:
    """Infer decimal places for fixed-point coordinates when the file omits FORMAT.

    CAM350 Excellon exports often contain e.g. X12345 with Leading zero
    suppression and rely on the export-dialog 3/2 setting rather than recording
    it in the file.  A nearby Gerber gives a very strong scale reference.
    """
    vals = []
    for tok in raw_coord_tokens:
        t = tok.strip().lstrip("+-")
        if t and "." not in t and t.isdigit():
            vals.append(int(t))
    if not vals:
        return 3
    vmax = max(vals)
    # Candidate decimal counts used by common Excellon metric/inch exports.
    best_d, best_score = 3, float("inf")
    for d in range(1, 7):
        # For LZ fixed-point values, omitted leading zeroes do not affect raw/10^d.
        span = None
        # Need coordinate range rather than max absolute, but tokens mix X/Y.  The
        # max absolute is still useful as a fallback; caller normally supplies Gerber span.
        mag = vmax * factor / (10.0 ** d)
        score = 0.0
        if reference_span_mm and reference_span_mm > 1e-6:
            # Drill coordinates can have a non-zero origin, so compare magnitude and
            # reference span softly rather than demanding exact equality.
            ratio = max(mag, 1e-6) / max(reference_span_mm, 1e-6)
            score += abs(math.log10(ratio))
            # Board-space coordinates are commonly within ~0.4x..4x copper span.
            if ratio < 0.25 or ratio > 6.0:
                score += 2.0
        else:
            # Prefer plausible PCB coordinate magnitude in mm.
            if mag < 5.0:
                score += 4.0 + (5.0 - mag) / 5.0
            elif mag > 1500.0:
                score += 4.0 + math.log10(mag / 1500.0 + 1.0)
            else:
                score += abs(math.log10(max(mag, 1e-6) / 150.0)) * 0.25
        # Mild preference for 2/3/4 decimal places over exotic formats.
        score += 0.04 * min(abs(d - 3), abs(d - 2), abs(d - 4))
        if score < best_score:
            best_score, best_d = score, d
    return int(best_d)


def _parse_excellon_fixed(token: str, fmt: tuple[int, int] | None, zero: str,
                           inferred_decimals: int, factor: float,
                           padding: str | None = None) -> float:
    token = token.strip()
    if not token:
        return 0.0
    if "." in token:
        return float(token) * factor
    sign = -1.0 if token.startswith("-") else 1.0
    digits = token.lstrip("+-")
    if not digits:
        return 0.0
    if fmt is None:
        return sign * (int(digits) / (10.0 ** inferred_decimals)) * factor
    idec, ddec = fmt
    total = max(1, int(idec) + int(ddec))
    # ``left`` means missing leading digits -> pad on the left (ordinary LZ).
    # ``right`` means missing trailing digits -> pad on the right.  Altium's
    # ASCII RoundHoles export can require the latter even while declaring LZ.
    if padding is None:
        padding = "right" if zero == "TZ" else "left"
    if padding == "right":
        digits = digits.ljust(total, "0")
    else:
        digits = digits.rjust(total, "0")
    return sign * (int(digits) / (10.0 ** int(ddec))) * factor


def _excellon_choose_fixed_padding(lines: list[str], fmt: tuple[int, int] | None,
                                    zero: str, factor: float,
                                    reference_bounds_mm: tuple[float, float, float, float] | None):
    """Choose left/right fixed-point padding by checking coordinates against Gerber.

    Without Gerber bounds we preserve the normal Excellon interpretation.  With
    bounds available we score both interpretations.  This specifically catches
    Altium ASCII RoundHoles files such as ``FILE_FORMAT=4:4 / METRIC,LZ`` where
    short tokens are effectively right-padded (X00525 -> 52.5 mm).
    """
    default = "right" if zero == "TZ" else "left"
    if fmt is None or reference_bounds_mm is None:
        return default, False, {}
    xmin, ymin, xmax, ymax = map(float, reference_bounds_mm)
    xspan=max(1e-6,xmax-xmin); yspan=max(1e-6,ymax-ymin)
    total=max(1,int(fmt[0])+int(fmt[1]))
    tokens={"X":[],"Y":[]}
    for raw in lines:
        u=raw.strip().upper()
        for axis in ("X","Y"):
            for tok in re.findall(axis+r"([+-]?[0-9.]+)",u):
                d=tok.lstrip("+-")
                if "." not in d and d.isdigit() and len(d)<total:
                    tokens[axis].append(tok)
    n=sum(len(v) for v in tokens.values())
    if n < 4:
        return default, False, {}

    def score(mode: str) -> float:
        good=0.0; count=0
        for axis,lo,hi,span in (("X",xmin,xmax,xspan),("Y",ymin,ymax,yspan)):
            # Drill features may sit just outside copper, so allow a generous margin.
            margin=max(2.0,0.18*span)
            for tok in tokens[axis]:
                v=_parse_excellon_fixed(tok,fmt,zero,fmt[1],factor,padding=mode)
                count+=1
                if lo-margin <= v <= hi+margin:
                    good += 1.0
                    if lo <= v <= hi:
                        good += 0.15
                else:
                    # Very remote values are more suspicious than a small edge overrun.
                    dist=(lo-v) if v<lo else (v-hi)
                    good -= min(1.0, dist/max(span,1.0))*0.35
        return good/max(1,count)

    scores={"left":score("left"),"right":score("right")}

    # An in-bounds test alone cannot disambiguate coordinates compressed around
    # the origin.  Example: Altium METRIC,LZ 4:3 'X-00368' can describe a
    # mounting hole at -36.8 mm, not -0.368 mm.  Both fall within an 80 mm
    # Gerber, so inspect the *two-dimensional spread of each drill tool*.
    # Only a clear mounting-hole pattern may override the standards-based default.
    groups = {"left": {}, "right": {}}
    tool_diam = {}
    active_tool = None
    cx = {"left": 0.0, "right": 0.0}
    cy = {"left": 0.0, "right": 0.0}
    for raw in lines:
        u = raw.strip().upper()
        definition = re.match(r"T(\d+).*?C([+-]?[0-9.]+)", u)
        if definition:
            tool_diam[int(definition.group(1))] = float(definition.group(2)) * factor
            continue
        selection = re.fullmatch(r"T(\d+)", u)
        if selection:
            active_tool = int(selection.group(1))
            continue
        if active_tool is None or float(tool_diam.get(active_tool, 0)) <= 0:
            continue
        # Routed slots are not a set of discrete mounting-hole centers.
        if re.match(r"(?:G0?[0123]|G85)\b", u):
            continue
        mx = re.search(r"X([+-]?[0-9.]+)", u)
        my = re.search(r"Y([+-]?[0-9.]+)", u)
        if not (mx or my):
            continue
        for mode in ("left", "right"):
            if mx:
                cx[mode] = _parse_excellon_fixed(mx.group(1), fmt, zero, fmt[1], factor, padding=mode)
            if my:
                cy[mode] = _parse_excellon_fixed(my.group(1), fmt, zero, fmt[1], factor, padding=mode)
            groups[mode].setdefault(active_tool, []).append((cx[mode], cy[mode]))

    # A spatially dispersed tool (including small non-plated tooling holes) is
    # decisive when one interpretation collapses both axes around the origin.
    # Avoid relying solely on large mounting holes: exported files may split
    # NPTH and PTH into separate short coordinate programs.
    geometry_choice = None
    for tool in tool_diam:
        a = groups["left"].get(tool, [])
        b = groups["right"].get(tool, [])
        if len(a) < 3 or len(b) < 3:
            continue
        def coverage(points):
            coords = np.asarray(points, np.float64)
            return (float(np.ptp(coords[:,0])) / xspan,
                    float(np.ptp(coords[:,1])) / yspan)
        left_span, right_span = coverage(a), coverage(b)
        for small, large in (("left", "right"), ("right", "left")):
            ss = left_span if small == "left" else right_span
            ll = right_span if large == "right" else left_span
            # Four tooling holes may span almost the whole PCB on just one
            # axis (e.g. an NPTH column).  In that case a nearly-zero-range
            # interpretation is still clearly implausible.
            dispersed = (min(ll) > 0.35 or (max(ll) > 0.45 and min(ll) > 0.04))
            if max(ss) < 0.035 and dispersed and max(ll) < 1.6:
                geometry_choice = large
                break
        if geometry_choice is not None:
            break

    if geometry_choice is not None:
        chosen = geometry_choice
        scores["geometry_signal"] = 1.0 if geometry_choice == "right" else -1.0
    else:
        best=max(("left", "right"),key=lambda k:scores[k])
        # Do not flip a standards-compliant file on a tiny statistical difference.
        chosen=best if scores[best] >= scores[default] + 0.08 else default
    return chosen, bool(chosen != default), scores


def parse_excellon(path: str | Path, reference_span_mm: float | None = None,
                    reference_bounds_mm: tuple[float, float, float, float] | None = None,
                    return_meta: bool = False):
    """Parse decimal and fixed-point Excellon.

    Supports ordinary ASCII Excellon as well as Altium's ASCII round-hole TXT
    output where plated and non-plated tool groups are contained in one file.
    Binary EIA .DRL files are intentionally ignored; when Altium exports both
    EIA .DRL and ASCII .TXT, the TXT file is the useful source for this app.
    """
    pp = Path(path)
    raw_bytes = pp.read_bytes()
    if raw_bytes:
        sample = raw_bytes[:4096]
        ctrl = sum(1 for b in sample if b < 9 or (13 < b < 32))
        if ctrl > max(8, len(sample) // 25):
            meta = {
                "path": str(pp), "units": None, "zero_suppression": None,
                "format": None, "inferred_decimals": None, "absolute": True,
                "points": 0, "binary_eia": True, "tool_plating": {},
            }
            return ([], meta) if return_meta else []
    lines = raw_bytes.decode("latin-1", errors="ignore").splitlines()
    units, factor, zero, fmt, absolute = _excellon_format_from_header(lines)
    raw_tokens = []
    for raw in lines:
        raw_tokens.extend(re.findall(r"[XY]([+-]?[0-9.]+)", raw.strip().upper()))
    inferred_decimals = fmt[1] if fmt is not None else _excellon_infer_decimals(
        raw_tokens, factor, reference_span_mm
    )
    fixed_padding, padding_autodetected, padding_scores = _excellon_choose_fixed_padding(
        lines, fmt, zero, factor, reference_bounds_mm
    )

    tools: dict[int, float] = {}
    tool_plating: dict[int, bool | None] = {}
    current_plating: bool | None = None
    for raw in lines:
        u = raw.strip().upper()
        if "TYPE=NON_PLATED" in u or "TYPE=NON-PLATED" in u or "TYPE=NPTH" in u:
            current_plating = False
            continue
        if "TYPE=PLATED" in u or "TYPE=PTH" in u:
            current_plating = True
            continue
        m = re.match(r"T(\d+).*?C([+-]?[0-9.]+)", u)
        if m:
            tid = int(m.group(1))
            tools[tid] = float(m.group(2)) * factor
            tool_plating[tid] = current_plating

    points: list[tuple[float, float, float, int]] = []
    tool = None
    cur_x = cur_y = 0.0
    routed_moves = 0
    for raw in lines:
        sline = raw.strip().upper()
        if not sline:
            continue
        m = re.fullmatch(r"T(\d+)", sline)
        if m:
            tool = int(m.group(1))
            continue
        if re.match(r"T\d+.*C[+-]?[0-9.]+", sline):
            continue
        if tool is None:
            continue
        mx = re.search(r"X([+-]?[0-9.]+)", sline)
        my = re.search(r"Y([+-]?[0-9.]+)", sline)
        if not (mx or my):
            continue
        nx = cur_x if mx is None else _parse_excellon_fixed(
            mx.group(1), fmt, zero, inferred_decimals, factor, padding=fixed_padding
        )
        ny = cur_y if my is None else _parse_excellon_fixed(
            my.group(1), fmt, zero, inferred_decimals, factor, padding=fixed_padding
        )
        if absolute:
            cur_x, cur_y = nx, ny
        else:
            cur_x += 0.0 if mx is None else nx
            cur_y += 0.0 if my is None else ny
        # Excellon routing (G00 positioning, G01/G02/G03 cutting, G85 slots)
        # describes milled paths, NOT independent circular drill hits.  Keep
        # the modal coordinates but exclude those points from hole registration.
        if re.match(r"G0?[0123](?=[XY*\s]|$)|G85(?=[XY*\s]|$)", sline):
            routed_moves += 1
            continue
        points.append((float(cur_x), float(cur_y), float(tools.get(tool, 0.0)), int(tool)))

    meta = {
        "path": str(pp), "units": units, "zero_suppression": zero,
        "format": fmt, "inferred_decimals": int(inferred_decimals),
        "fixed_padding": fixed_padding, "padding_autodetected": bool(padding_autodetected),
        "padding_scores": {k: round(float(v), 4) for k, v in padding_scores.items()},
        "absolute": bool(absolute), "points": len(points),
        "binary_eia": False, "tool_plating": dict(tool_plating),
        "routed_moves_skipped": routed_moves,
    }
    return (points, meta) if return_meta else points

def detect_large_holes(img: np.ndarray, board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, S = lab[:, :, 0], hsv[:, :, 1]
    vals = L[board > 0]
    bright_thr = max(180, int(np.percentile(vals, 85)))
    cand = ((L >= bright_thr) & (S < 95) & (board > 0)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    x0, y0, x1, y1 = board_bbox(board, 0)
    bw = max(1, x1 - x0)
    dmin, dmax = max(5, int(0.004 * bw)), max(24, int(0.05 * bw))
    pts = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (dmin <= w <= dmax and dmin <= h <= dmax):
            continue
        if min(w, h) < max(7, int(0.007 * bw)) or max(w, h) < max(10, int(0.009 * bw)):
            continue
        if not (0.55 <= w / max(h, 1) <= 1.8):
            continue
        fill = area / float(max(1, w * h))
        if not (0.45 <= fill <= 0.95):
            continue
        pts.append(cents[i])
    return np.asarray(pts, np.float64), cand


def similarity_from_pair(q1, q2, p1, p2):
    vq, vp = q2 - q1, p2 - p1
    dq, dp = np.linalg.norm(vq), np.linalg.norm(vp)
    if dq < 1e-9 or dp < 1e-9:
        return None
    scale = dp / dq
    th = math.atan2(vp[1], vp[0]) - math.atan2(vq[1], vq[0])
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], np.float64) * scale
    t = p1 - R @ q1
    return R, t


def apply_sim(q, R, t):
    return q @ R.T + t


def score_transform(q, p, R, t, tol):
    pred = apply_sim(q, R, t)
    dist = np.sqrt(((pred[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))
    nearest = dist.argmin(axis=1)
    dmin = dist[np.arange(len(q)), nearest]
    order = np.argsort(dmin)
    used_p, pairs = set(), []
    for qi in order:
        pj = int(nearest[qi])
        d = float(dmin[qi])
        if d > tol or pj in used_p:
            continue
        used_p.add(pj)
        pairs.append((int(qi), pj, d))
    if not pairs:
        return 0, 1e9, []
    rms = math.sqrt(np.mean([x[2] ** 2 for x in pairs]))
    return len(pairs), rms, pairs


def _to_homography_from_affine(A: np.ndarray) -> np.ndarray:
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = A
    return H


def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    if pts.size == 0:
        return np.empty((0, 2), np.float64)
    hp = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H.astype(np.float64))
    return hp.reshape(-1, 2)


def fit_registration_points(src: np.ndarray, dst: np.ndarray, mode: str = "auto") -> dict[str, Any]:
    """Fit affine/projective mapping from Gerber-space control points to scan pixels."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if len(src) < 3:
        raise RuntimeError("Need at least 3 control points")

    # With only three controls a full 6-DOF affine can map any triangle onto any other
    # triangle exactly.  That is dangerous for partially scanned / circular boards because a
    # wrong correspondence can look perfect.  Keep the physical scanner model (rotation +
    # uniform scale + translation) until at least four independent anchors are available.
    if len(src) <= 3:
        A, in_aff = cv2.estimateAffinePartial2D(
            src.astype(np.float32), dst.astype(np.float32), method=cv2.RANSAC,
            ransacReprojThreshold=1.8, confidence=0.999, refineIters=80,
        )
    else:
        A, in_aff = cv2.estimateAffine2D(
            src.astype(np.float32), dst.astype(np.float32), method=cv2.RANSAC,
            ransacReprojThreshold=1.5, confidence=0.999, refineIters=80,
        )
    if A is None:
        A, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32))
    if A is None:
        raise RuntimeError("Affine registration failed")
    H_aff = _to_homography_from_affine(A)
    pa = transform_points(H_aff, src)
    ea = np.linalg.norm(pa - dst, axis=1)
    aff_rms = float(np.sqrt(np.mean(ea ** 2)))
    aff_max = float(ea.max()) if len(ea) else 0.0

    H_proj = None
    proj_rms = float("inf")
    proj_max = float("inf")
    ep = None
    if len(src) >= 4:
        H_proj, in_proj = cv2.findHomography(
            src.astype(np.float32), dst.astype(np.float32), method=cv2.RANSAC,
            ransacReprojThreshold=1.4, confidence=0.999, maxIters=5000,
        )
        if H_proj is not None and np.isfinite(H_proj).all() and abs(H_proj[2, 2]) > 1e-12:
            H_proj = H_proj / H_proj[2, 2]
            pp = transform_points(H_proj, src)
            ep = np.linalg.norm(pp - dst, axis=1)
            proj_rms = float(np.sqrt(np.mean(ep ** 2)))
            proj_max = float(ep.max()) if len(ep) else 0.0
            # Reject implausibly strong perspective. Scanner distortion should be gentle.
            if abs(H_proj[2, 0]) > 0.01 or abs(H_proj[2, 1]) > 0.01:
                H_proj = None
                proj_rms = float("inf")
                proj_max = float("inf")
                ep = None

    mode_l = mode.lower()
    if mode_l not in {"auto", "affine", "projective"}:
        mode_l = "auto"

    use_projective = False
    if mode_l == "projective" and H_proj is not None:
        use_projective = True
    elif mode_l == "auto" and H_proj is not None and len(src) >= 6:
        # With many anchors, accept a mild homography when it gives a meaningful improvement.
        improvement = aff_rms - proj_rms
        use_projective = (improvement >= 0.05 and proj_rms <= aff_rms * 0.95) or (aff_max - proj_max >= 0.20)

    H = H_proj if use_projective else H_aff
    model = "projective" if use_projective else "affine"
    pred = transform_points(H, src)
    err = np.linalg.norm(pred - dst, axis=1)

    # Residual trend by vertical position helps diagnose perspective-like scanner distortion.
    order = np.argsort(dst[:, 1]) if len(dst) else np.array([], int)
    n3 = max(1, len(order) // 3) if len(order) else 1
    top_idx = order[:n3]
    bottom_idx = order[-n3:]
    top_rms = float(np.sqrt(np.mean(err[top_idx] ** 2))) if len(top_idx) else 0.0
    bottom_rms = float(np.sqrt(np.mean(err[bottom_idx] ** 2))) if len(bottom_idx) else 0.0

    return {
        "H": H,
        "model": model,
        "rms": float(np.sqrt(np.mean(err ** 2))),
        "maxerr": float(err.max()) if len(err) else 0.0,
        "errors": err,
        "pred": pred,
        "affine_H": H_aff,
        "affine_rms": aff_rms,
        "affine_max": aff_max,
        "projective_H": H_proj,
        "projective_rms": proj_rms,
        "projective_max": proj_max,
        "top_rms": top_rms,
        "bottom_rms": bottom_rms,
    }


def auto_point_registration(drill_xy: np.ndarray, scan_pts: np.ndarray, mode: str = "auto") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Robust partial point-set registration for PCB drill anchors.

    v0.15.1 keeps *several* geometrically distinct registration candidates instead of one
    candidate per mirror state.  This matters for round boards: a symmetric ring of mounting
    holes can have several rotations with the same anchor count and almost the same RMS.  The
    caller can then resolve that ambiguity using the complete drill map.
    """
    q0 = np.asarray(drill_xy, np.float64)
    p = np.asarray(scan_pts, np.float64)
    if len(q0) < 3 or len(p) < 3:
        raise RuntimeError(f"Not enough anchor points (drill={len(q0)}, scan={len(p)})")

    q_extent = float(np.linalg.norm(np.ptp(q0, axis=0))) if len(q0) else 0.0
    q_min_pair = max(8.0, 0.08 * q_extent)
    p_extent = float(np.linalg.norm(np.ptp(p, axis=0))) if len(p) else 0.0
    p_min_pair = max(18.0, 0.035 * p_extent)

    q_pairs0 = []
    for i in range(len(q0)):
        for j in range(i + 1, len(q0)):
            d = float(np.linalg.norm(q0[j] - q0[i]))
            if d >= q_min_pair:
                q_pairs0.append((i, j, d))
    p_pairs = []
    for a in range(len(p)):
        for b in range(a + 1, len(p)):
            d = float(np.linalg.norm(p[b] - p[a]))
            if d >= p_min_pair:
                p_pairs.append((a, b, d))
    if not q_pairs0 or not p_pairs:
        raise RuntimeError("Anchor registration failed: not enough separated point pairs")

    min_scale_pxmm, max_scale_pxmm = 1.2, 30.0
    coarse_tol = max(5.0, min(12.0, 0.018 * max(100.0, p_extent)))

    def _fit_signature(fit, q):
        # Distinguish rotational aliases on circular boards while collapsing repeated solutions
        # produced by different anchor pairs.  The bins are deliberately coarse: final ranking is
        # still by actual anchor count/RMS.
        H = fit["H"]
        o = transform_points(H, np.array([[0.0, 0.0], [1.0, 0.0]], np.float64))
        v = o[1] - o[0]
        ang = (math.degrees(math.atan2(float(v[1]), float(v[0]))) + 360.0) % 360.0
        scale = float(np.linalg.norm(v))
        c0 = np.mean(q, axis=0, keepdims=True)
        c = transform_points(H, c0)[0]
        return (int(round(ang / 2.0)), int(round(float(c[0]) / 10.0)),
                int(round(float(c[1]) / 10.0)), int(round(scale / 0.03)))

    results = []
    for mirror in (False, True):
        q = q0.copy()
        if mirror:
            q[:, 0] *= -1

        hypotheses = []
        for i, j, dq in q_pairs0:
            for a, b, dp in p_pairs:
                sc = dp / max(dq, 1e-9)
                if not (min_scale_pxmm <= sc <= max_scale_pxmm):
                    continue
                for aa, bb in ((a, b), (b, a)):
                    st = similarity_from_pair(q[i], q[j], p[aa], p[bb])
                    if st is None:
                        continue
                    R, t = st
                    cnt, rms, pairs = score_transform(q, p, R, t, tol=coarse_tol)
                    if cnt < 3:
                        continue
                    hypotheses.append((cnt, rms, -dq, R, t, pairs))

        if not hypotheses:
            continue
        hypotheses.sort(key=lambda z: (-z[0], z[1], z[2]))

        # Keep several unique fits for this mirror state.  The old code retained only one, which
        # discarded equally good rotations of symmetric circular anchor patterns too early.
        unique = {}
        for cnt0, rms0, negdq, R, t, pairs0 in hypotheses[:120]:
            src0 = np.asarray([q[i] for i, j, d in pairs0], np.float64)
            dst0 = np.asarray([p[j] for i, j, d in pairs0], np.float64)
            if len(src0) < 3:
                continue
            try:
                fit0 = fit_registration_points(src0, dst0, mode=mode)
            except Exception:
                continue
            pred = transform_points(fit0["H"], q)
            dist = np.sqrt(((pred[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))
            nearest = dist.argmin(axis=1)
            dmin = dist[np.arange(len(q)), nearest]
            order = np.argsort(dmin)
            used = set(); pairs = []
            for qi in order:
                pj = int(nearest[qi]); dd = float(dmin[qi])
                if dd > coarse_tol or pj in used:
                    continue
                used.add(pj); pairs.append((int(qi), pj, dd))
            if len(pairs) < 3:
                continue
            src = np.asarray([q[i] for i, j, d in pairs], np.float64)
            dst = np.asarray([p[j] for i, j, d in pairs], np.float64)
            try:
                fit = fit_registration_points(src, dst, mode=mode)
            except Exception:
                continue
            fit.update({
                "mirror": mirror,
                "count": len(src),
                "pairs": [(int(i), int(j)) for i, j, d in pairs],
                "control_src": src.copy(),
                "control_dst": dst.copy(),
                "control_dst_auto": dst.copy(),
                "partial_registration": True,
            })
            sig = _fit_signature(fit, q)
            key = (-len(src), float(fit["rms"]), float(fit["maxerr"]))
            prev = unique.get(sig)
            if prev is None or key < prev[0]:
                unique[sig] = (key, fit)

        cand = [v[1] for v in unique.values()]
        cand.sort(key=lambda z: (-z["count"], z["rms"], z.get("maxerr", 1e9)))
        results.extend(cand[:8])

    if not results:
        raise RuntimeError("Anchor registration failed")
    results.sort(key=lambda z: (-z["count"], z["rms"], z.get("maxerr", 1e9)))
    return results[0], results[:16]

def parse_gerber(path: str | Path):
    """Parse a practical RS-274X subset used by the visual overlay.

    Besides KiCad/CAM350 exports this handles common Altium output: standalone
    modal D01/D02/D03 commands and embedded aperture macros made from circle,
    center-line, vector-line, outline and polygon primitives.
    """
    apertures, ops = {}, []
    macros: dict[str, dict[str, Any]] = {}
    cur_ap = None
    cur = (0.0, 0.0)
    x_scale = y_scale = 1e6
    unit_factor = 1.0
    region = False
    contours, contour = [], []
    polarity_dark = True
    interp = "G01"
    cur_op: int | None = None
    in_macro = None
    macro_lines: list[str] = []

    def op_name(base: str) -> str:
        return base if polarity_dark else base + "_clear"

    def _rot_pts(pts, deg):
        if abs(float(deg)) < 1e-12:
            return [(float(x), float(y)) for x, y in pts]
        a = math.radians(float(deg)); c, ss = math.cos(a), math.sin(a)
        return [(c*float(x)-ss*float(y), ss*float(x)+c*float(y)) for x,y in pts]

    def _finish_macro():
        nonlocal in_macro, macro_lines
        if not in_macro:
            return
        primitives = []
        joined = "".join(macro_lines)
        for prim in joined.split("*"):
            prim = prim.strip().strip("%")
            if not prim or "$" in prim:
                continue
            vals = [v.strip() for v in prim.split(",")]
            try:
                code = int(float(vals[0]))
                if code == 1 and len(vals) >= 5:  # circle
                    exp = int(float(vals[1])); dia=float(vals[2]); cx=float(vals[3]); cy=float(vals[4])
                    rot=float(vals[5]) if len(vals)>5 else 0.0
                    cx,cy=_rot_pts([(cx,cy)],rot)[0]
                    primitives.append({"type":"circle","exposure":exp,"dia":dia,"cx":cx,"cy":cy})
                elif code == 20 and len(vals) >= 7:  # vector line
                    exp=int(float(vals[1])); width=float(vals[2])
                    x1,y1,x2,y2=map(float,vals[3:7]); rot=float(vals[7]) if len(vals)>7 else 0.0
                    (x1,y1),(x2,y2)=_rot_pts([(x1,y1),(x2,y2)],rot)
                    primitives.append({"type":"vector","exposure":exp,"width":width,
                                       "x1":x1,"y1":y1,"x2":x2,"y2":y2})
                elif code == 21 and len(vals) >= 6:  # center line rectangle
                    exp=int(float(vals[1])); w=float(vals[2]); h=float(vals[3]); cx=float(vals[4]); cy=float(vals[5])
                    rot=float(vals[6]) if len(vals)>6 else 0.0
                    primitives.append({"type":"rect","exposure":exp,"w":w,"h":h,"cx":cx,"cy":cy,"rot":rot})
                elif code == 4 and len(vals) >= 8:  # outline
                    exp=int(float(vals[1])); n=int(float(vals[2])); nums=[float(v) for v in vals[3:3+2*n]]
                    if len(nums)==2*n:
                        pts=[(nums[i],nums[i+1]) for i in range(0,len(nums),2)]
                        rot=float(vals[3+2*n]) if len(vals)>3+2*n else 0.0
                        pts=_rot_pts(pts,rot)
                        primitives.append({"type":"poly","exposure":exp,"points":pts})
                elif code == 5 and len(vals) >= 7:  # regular polygon
                    exp=int(float(vals[1])); n=max(3,int(float(vals[2]))); cx=float(vals[3]); cy=float(vals[4]); dia=float(vals[5]); rot=float(vals[6])
                    a0=math.radians(rot)
                    pts=[(cx+dia*0.5*math.cos(a0+2*math.pi*k/n),cy+dia*0.5*math.sin(a0+2*math.pi*k/n)) for k in range(n)]
                    primitives.append({"type":"poly","exposure":exp,"points":pts})
            except Exception:
                continue
        first_poly = next((q.get("points") for q in primitives if q.get("type")=="poly" and q.get("exposure",1)==1), None)
        macros[in_macro] = {"primitives": primitives, "points": first_poly, "raw": list(macro_lines)}
        in_macro = None
        macro_lines = []

    with open(path, "r", errors="ignore") as f:
        for raw in f:
            sline = raw.strip()
            if not sline:
                continue

            if in_macro is not None:
                if sline == "%" or sline.endswith("%"):
                    macro_lines.append(sline[:-1] if sline.endswith("%") else "")
                    _finish_macro()
                else:
                    macro_lines.append(sline)
                continue
            mm = re.match(r"%AM([^*%]+)\*(.*)", sline)
            if mm:
                in_macro = mm.group(1).strip(); rest = mm.group(2)
                if rest.endswith("%"):
                    macro_lines.append(rest[:-1]); _finish_macro()
                elif rest:
                    macro_lines.append(rest)
                continue

            m = re.match(r"%ADD(\d+)([CROP]),([0-9.+-]+)(?:X([0-9.+-]+))?(?:X([0-9.+-]+))?\*%", sline)
            if m:
                d=int(m.group(1)); sh=m.group(2); a=float(m.group(3)); b=float(m.group(4)) if m.group(4) else a
                extra=float(m.group(5)) if m.group(5) else 0.0
                apertures[d]=(sh,a,b,extra); continue
            m = re.match(r"%ADD(\d+)([A-Za-z_][A-Za-z0-9_]*)(?:,([^*%]+))?\*%", sline)
            if m:
                d=int(m.group(1)); name=m.group(2); mods=m.group(3)
                apertures[d] = ("M", name, mods, macros.get(name)); continue

            m = re.search(r"%FS[A-Z]*X(\d)(\d)Y(\d)(\d)", sline)
            if m:
                x_scale=10**int(m.group(2)); y_scale=10**int(m.group(4)); continue
            if "%MOIN" in sline:
                unit_factor=25.4; continue
            if "%MOMM" in sline:
                unit_factor=1.0; continue
            if sline.startswith("%LPD"):
                polarity_dark=True; continue
            if sline.startswith("%LPC"):
                polarity_dark=False; continue

            if "G01" in sline: interp="G01"
            elif "G02" in sline: interp="G02"
            elif "G03" in sline: interp="G03"

            if sline.startswith("G36"):
                region,contours,contour=True,[],[]; continue
            if sline.startswith("G37"):
                if contour: contours.append(contour)
                if contours: ops.append((op_name("region"),contours,None))
                region,contours,contour=False,[],[]; continue

            # Aperture selection D10+ (including legacy G54Dnn).
            m = re.fullmatch(r"(?:G54)?D(\d+)\*", sline)
            if m and int(m.group(1)) >= 10:
                cur_ap=int(m.group(1)); continue

            # Altium commonly writes coordinate D02 followed by a standalone D03.
            # D01/D02/D03 are modal operation codes, not aperture selections.
            m = re.fullmatch(r"D0?([123])\*", sline)
            if m:
                cur_op=int(m.group(1))
                if cur_op==3 and not region and cur_ap is not None:
                    ops.append((op_name("flash"), cur, cur_ap))
                elif cur_op==2 and region:
                    if contour: contours.append(contour)
                    contour=[cur]
                continue

            mx=re.search(r"X([+-]?\d+)",sline); my=re.search(r"Y([+-]?\d+)",sline)
            if not (mx or my):
                continue
            x=cur[0] if mx is None else (int(mx.group(1))/x_scale)*unit_factor
            y=cur[1] if my is None else (int(my.group(1))/y_scale)*unit_factor
            new=(x,y)
            md=re.search(r"D0?([123])(?:\*|$)",sline)
            if md: cur_op=int(md.group(1))
            code=cur_op

            if region:
                if code==2:
                    if contour: contours.append(contour)
                    contour=[new]
                elif code==1:
                    if not contour: contour=[cur]
                    mi=re.search(r"I([+-]?\d+)",sline); mj=re.search(r"J([+-]?\d+)",sline)
                    if interp in ("G02","G03") and (mi or mj):
                        io=0.0 if mi is None else (int(mi.group(1))/x_scale)*unit_factor
                        jo=0.0 if mj is None else (int(mj.group(1))/y_scale)*unit_factor
                        cx,cy=cur[0]+io,cur[1]+jo; r0=math.hypot(cur[0]-cx,cur[1]-cy)
                        if r0>1e-9:
                            a0=math.atan2(cur[1]-cy,cur[0]-cx); a1=math.atan2(new[1]-cy,new[0]-cx)
                            if interp=="G02":
                                while a1>=a0: a1-=2*math.pi
                            else:
                                while a1<=a0: a1+=2*math.pi
                            da=a1-a0; n=max(2,int(math.ceil(abs(da)*r0/0.10)))
                            for k in range(1,n+1):
                                a=a0+da*k/n; contour.append((cx+r0*math.cos(a),cy+r0*math.sin(a)))
                        else: contour.append(new)
                    else: contour.append(new)
                cur=new; continue

            if code==3:
                if cur_ap is not None: ops.append((op_name("flash"),new,cur_ap))
            elif code==1:
                mi=re.search(r"I([+-]?\d+)",sline); mj=re.search(r"J([+-]?\d+)",sline)
                if interp in ("G02","G03") and (mi or mj):
                    io=0.0 if mi is None else (int(mi.group(1))/x_scale)*unit_factor
                    jo=0.0 if mj is None else (int(mj.group(1))/y_scale)*unit_factor
                    cx,cy=cur[0]+io,cur[1]+jo; r0=math.hypot(cur[0]-cx,cur[1]-cy)
                    if r0>1e-9:
                        a0=math.atan2(cur[1]-cy,cur[0]-cx); a1=math.atan2(new[1]-cy,new[0]-cx)
                        if interp=="G02":
                            while a1>=a0: a1-=2*math.pi
                        else:
                            while a1<=a0: a1+=2*math.pi
                        da=a1-a0; n=max(2,int(math.ceil(abs(da)*r0/0.10))); prev=cur
                        for k in range(1,n+1):
                            a=a0+da*k/n; nxt=(cx+r0*math.cos(a),cy+r0*math.sin(a))
                            if cur_ap is not None: ops.append((op_name("line"),(prev,nxt),cur_ap))
                            prev=nxt
                    elif cur_ap is not None: ops.append((op_name("line"),(cur,new),cur_ap))
                elif cur_ap is not None:
                    ops.append((op_name("line"),(cur,new),cur_ap))
            cur=new

    if in_macro is not None: _finish_macro()
    return apertures, ops

def map_xy(H: np.ndarray, mirror: bool, xy):
    x, y = xy
    q = np.array([[[-x if mirror else x, -y]]], np.float64)
    return cv2.perspectiveTransform(q, H.astype(np.float64))[0, 0]

def registration_angle_deg(H: np.ndarray, mirror: bool = False) -> float:
    """Displayed CAM/Gerber +X direction in scan coordinates, in degrees."""
    p0 = map_xy(H, mirror, (0.0, 0.0))
    p1 = map_xy(H, mirror, (1.0, 0.0))
    v = np.asarray(p1, np.float64) - np.asarray(p0, np.float64)
    return float((math.degrees(math.atan2(float(v[1]), float(v[0]))) + 360.0) % 360.0)


def _local_px_per_mm(H: np.ndarray, mirror: bool, xy) -> float:
    x, y = xy
    p = map_xy(H, mirror, (x, y))
    px = map_xy(H, mirror, (x + 1.0, y))
    py = map_xy(H, mirror, (x, y + 1.0))
    sx = np.linalg.norm(px - p)
    sy = np.linalg.norm(py - p)
    return float((sx + sy) * 0.5)


def render_gerber_to_scan(path: str | Path, shape, H: np.ndarray, mirror: bool, supersample: int | None = None) -> np.ndarray:
    """Render Gerber directly in scan coordinates with polarity support."""
    h, w = shape[:2]
    if supersample is None:
        mp = (h * w) / 1e6
        supersample = 4 if mp <= 3.0 else (2 if mp <= 12.0 else 1)
    ss = max(1, int(supersample))
    canvas = np.zeros((h * ss, w * ss), np.uint8)
    aps, ops = parse_gerber(path)

    def P(xy):
        return map_xy(H, mirror, xy) * ss

    def _macro_poly(prim, fx, fy):
        typ=prim.get("type")
        if typ=="circle":
            r=float(prim.get("dia",0.0))*0.5; cx=fx+float(prim.get("cx",0.0)); cy=fy+float(prim.get("cy",0.0))
            pts=[(cx+r*math.cos(t),cy+r*math.sin(t)) for t in np.linspace(0,2*math.pi,64,endpoint=False)]
        elif typ=="rect":
            w=float(prim.get("w",0.0)); h=float(prim.get("h",0.0)); cx=float(prim.get("cx",0.0)); cy=float(prim.get("cy",0.0)); a=math.radians(float(prim.get("rot",0.0)))
            c,ssn=math.cos(a),math.sin(a); base=[(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
            pts=[(fx+cx+c*px-ssn*py,fy+cy+ssn*px+c*py) for px,py in base]
        elif typ=="vector":
            x1=float(prim.get("x1",0.0)); y1=float(prim.get("y1",0.0)); x2=float(prim.get("x2",0.0)); y2=float(prim.get("y2",0.0)); width=float(prim.get("width",0.0))
            dx,dy=x2-x1,y2-y1; L=max(1e-12,math.hypot(dx,dy)); nx=-dy/L*width/2; ny=dx/L*width/2
            pts=[(fx+x1+nx,fy+y1+ny),(fx+x2+nx,fy+y2+ny),(fx+x2-nx,fy+y2-ny),(fx+x1-nx,fy+y1-ny)]
        elif typ=="poly":
            pts=[(fx+float(px),fy+float(py)) for px,py in prim.get("points",[])]
        else:
            return None
        if len(pts)<3:return None
        return np.asarray([P(q) for q in pts],np.float32)

    def _draw_macro(meta, fx, fy, overall_value):
        prims=[] if not meta else list(meta.get("primitives") or [])
        if not prims:
            pts=None if not meta else meta.get("points")
            if pts:
                poly=np.rint(np.asarray([P((fx+px,fy+py)) for px,py in pts],np.float32)).astype(np.int32)
                cv2.fillPoly(canvas,[poly],overall_value,lineType=cv2.LINE_AA)
            return
        polys=[]
        for prim in prims:
            poly=_macro_poly(prim,fx,fy)
            if poly is not None:
                polys.append((int(prim.get("exposure",1)),poly))
        if not polys:return
        allpts=np.concatenate([q for _,q in polys],axis=0)
        x0=max(0,int(math.floor(float(allpts[:,0].min())))-2); y0=max(0,int(math.floor(float(allpts[:,1].min())))-2)
        x1=min(canvas.shape[1],int(math.ceil(float(allpts[:,0].max())))+3); y1=min(canvas.shape[0],int(math.ceil(float(allpts[:,1].max())))+3)
        if x1<=x0 or y1<=y0:return
        mask=np.zeros((y1-y0,x1-x0),np.uint8)
        for exposure,poly in polys:
            local=np.rint(poly-np.array([x0,y0],np.float32)).astype(np.int32)
            cv2.fillPoly(mask,[local],255 if exposure else 0,lineType=cv2.LINE_AA)
        roi=canvas[y0:y1,x0:x1]
        if overall_value:
            np.maximum(roi,mask,out=roi)
        else:
            roi[mask>0]=0

    def draw_flash(apdef, data, value):
        sh = apdef[0]
        x, y = data
        if sh == "C":
            a = float(apdef[1])
            angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            pts = [(x + a * 0.5 * math.cos(t), y + a * 0.5 * math.sin(t)) for t in angles]
            poly = np.rint(np.asarray([P(p) for p in pts], np.float32)).astype(np.int32)
            cv2.fillConvexPoly(canvas, poly, value, lineType=cv2.LINE_AA)
        elif sh == "O":
            a, b = float(apdef[1]), float(apdef[2])
            # Obround: rectangle plus semicircular ends. Polygon approximation works for both orientations.
            n = 24
            if a >= b:
                r = b/2; half = max(0.0, (a-b)/2)
                pts=[]
                for t in np.linspace(-math.pi/2, math.pi/2, n):
                    pts.append((x+half+r*math.cos(t), y+r*math.sin(t)))
                for t in np.linspace(math.pi/2, 3*math.pi/2, n):
                    pts.append((x-half+r*math.cos(t), y+r*math.sin(t)))
            else:
                r = a/2; half = max(0.0, (b-a)/2)
                pts=[]
                for t in np.linspace(0, math.pi, n):
                    pts.append((x+r*math.cos(t), y+half+r*math.sin(t)))
                for t in np.linspace(math.pi, 2*math.pi, n):
                    pts.append((x+r*math.cos(t), y-half+r*math.sin(t)))
            poly=np.rint(np.asarray([P(p) for p in pts],np.float32)).astype(np.int32)
            cv2.fillConvexPoly(canvas,poly,value,lineType=cv2.LINE_AA)
        elif sh == "P":
            dia = float(apdef[1]); vertices = max(3, int(round(float(apdef[2]))))
            rot = math.radians(float(apdef[3]) if len(apdef) > 3 else 0.0)
            pts=[(x+dia*0.5*math.cos(rot+2*math.pi*k/vertices),
                  y+dia*0.5*math.sin(rot+2*math.pi*k/vertices)) for k in range(vertices)]
            poly=np.rint(np.asarray([P(p) for p in pts],np.float32)).astype(np.int32)
            cv2.fillConvexPoly(canvas,poly,value,lineType=cv2.LINE_AA)
        elif sh == "M":
            meta = apdef[3] if len(apdef) > 3 else None
            _draw_macro(meta, x, y, value)
        else:  # rectangle and safe fallback
            a, b = float(apdef[1]), float(apdef[2])
            corners=[(x-a/2,y-b/2),(x+a/2,y-b/2),(x+a/2,y+b/2),(x-a/2,y+b/2)]
            pts=np.asarray([P(q) for q in corners],np.float32)
            cv2.fillConvexPoly(canvas,np.rint(pts).astype(np.int32),value,lineType=cv2.LINE_AA)

    for typ, data, ap in ops:
        clear = typ.endswith("_clear")
        base = typ[:-6] if clear else typ
        value = 0 if clear else 255

        if base == "region":
            polys=[]
            for contour in data:
                pts=np.asarray([P(p) for p in contour],np.float32)
                if len(pts)>=3:
                    polys.append(np.rint(pts).astype(np.int32))
            if polys:
                cv2.fillPoly(canvas,polys,value,lineType=cv2.LINE_AA)
            continue
        if ap not in aps:
            continue
        apdef=aps[ap]
        sh=apdef[0]

        if base == "line":
            p0d,p1d=data
            mid=((p0d[0]+p1d[0])*0.5,(p0d[1]+p1d[1])*0.5)
            pxmm=_local_px_per_mm(H,mirror,mid)
            if sh=="C":
                width_mm=float(apdef[1])
            elif sh=="O":
                width_mm=min(float(apdef[1]),float(apdef[2]))
            elif sh=="R":
                width_mm=min(float(apdef[1]),float(apdef[2]))
            elif sh=="P":
                width_mm=float(apdef[1])
            elif sh=="M":
                meta=apdef[3] if len(apdef)>3 else None
                pts=None if not meta else meta.get("points")
                if pts:
                    xs=[q[0] for q in pts]; ys=[q[1] for q in pts]
                    width_mm=min(max(xs)-min(xs),max(ys)-min(ys))
                else:
                    continue
            else:
                width_mm=float(apdef[1])
            width=max(1,int(round(width_mm*pxmm*ss)))
            p0=tuple(np.rint(P(p0d)).astype(int)); p1=tuple(np.rint(P(p1d)).astype(int))
            cv2.line(canvas,p0,p1,value,width,lineType=cv2.LINE_AA)
        elif base == "flash":
            draw_flash(apdef,data,value)

    if ss > 1:
        return cv2.resize(canvas,(w,h),interpolation=cv2.INTER_AREA)
    return canvas




# ---------------------------------------------------------------------------
# CAMvu / CAM text database support (v0.12)
# ---------------------------------------------------------------------------

CAMVU_COORD_MM = 2.0e-6  # coordinate / NC-tool scale used by a legacy CAM350 v6 database layout
# Some legacy CAM350 v6 exports have a scale difference:
# aperture table dimensions are stored on a different numeric scale than XY/NC tools.
# Using CAMVU_COORD_MM for aperture sizes makes every draw/pad almost exactly 2x too large.
# The compatibility scaling is applied for those legacy exports.
CAMVU_APERTURE_MM = 1.0e-6
# Some legacy CAM350 exports encode: GRAPHIC draw apertures and
# PADS flashes use the same aperture IDs but not the same effective size scale.
# Draws use CAMVU_APERTURE_MM, while PADS flashes use the normal
# coordinate scale (2x here).  Keeping the distinction recovers side fingers /
# component pads without making all routed tracks twice as wide.
CAMVU_PAD_APERTURE_FACTOR = CAMVU_COORD_MM / CAMVU_APERTURE_MM
# Some legacy CAM350 text databases use APERID 0 for a real routed draw while
# omitting that aperture from *APERTURES1.  The compatibility profile treats
# these draws as the 1.5 mm track class when the legacy format signature matches.
CAMVU_APER0_DRAW_MM = 1.5
# Some older CAM350 text databases use GRAPHIC APERID values as internal
# draw-style IDs rather than literal physical stroke widths from *APERTURES1.
# Keep this compatibility profile isolated behind a content-signature check.
# Unknown IDs fall back to the aperture dimensions.
CAMVU_DRAW_WIDTH_LEGACY_PROFILE = {
    0: 1.50,
    20: 1.55,
    35: 1.00,
    37: 0.80,
    38: 0.90,
    39: 0.60,
    40: 1.50,
    41: 0.525,
}

def parse_camvu(path: str | Path) -> dict[str, Any]:
    """Parse a CAMvu / CAM350 text CAM database.

    The parser intentionally focuses on information needed by AOI: layer names/maps,
    apertures, copper graphics, pad flashes and NC drill hits. Unknown records are ignored.
    Coordinates/aperture sizes are converted to millimetres immediately.
    """
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()
    layers: dict[int, str] = {}
    layer_maps: dict[int, int] = {}
    apertures: dict[int, tuple[str, float, float]] = {}
    # Extra aperture metadata.  CAM350 can define a rotated/custom aperture with a
    # bounding box in columns 3 and the real base primitive + angle later in the row.
    # Keep the legacy 3-tuple in ``apertures`` for compatibility and store the richer
    # description here.
    aperture_meta: dict[int, dict[str, Any]] = {}
    aperture_order: list[int] = []
    primitives: dict[int, list[tuple]] = {}
    pads: list[tuple[float, float, int, int]] = []
    nc_tools: dict[int, float] = {}
    holes: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    notes: dict[str, str] = {}

    i = 0
    in_layers = in_graph = in_pads = False
    layer = None
    aper = -1
    width_mm = -1.0
    poly_void = False
    drill_layer = None
    drill_tool = None
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if not s:
            continue

        if s == "*LAYERSWID3":
            in_layers = True
            continue
        if s == "*END_LAYERS":
            in_layers = False
            continue
        if in_layers:
            t = s.split()
            if len(t) >= 2 and t[0].lstrip("-").isdigit():
                layers[int(t[0])] = t[1]
            continue

        if s == "*LAYERMAPS":
            while i < len(lines):
                ss = lines[i].strip(); i += 1
                if ss == "*END_LAYERMAPS":
                    break
                t = ss.split()
                if len(t) >= 3 and t[0] == "FromTo":
                    layer_maps[int(t[1])] = int(t[2])
            continue

        if s.startswith("NOTELINE ") and "=" in s:
            body = s[len("NOTELINE "):].strip().strip('\"')
            k, _, v = body.partition("=")
            notes[k.strip()] = v.strip().strip('\"')
            continue

        if s.startswith("*APERTURES1"):
            while i < len(lines):
                ss = lines[i].strip(); i += 1
                if ss == "*END_APERTURES":
                    break
                t = ss.split()
                if len(t) < 3 or not t[0].lstrip("-").isdigit():
                    continue
                try:
                    aid = int(t[0]); shape = t[1]
                    ab = t[2].split(":")
                    a = float(ab[0]) * CAMVU_APERTURE_MM
                    b = float(ab[1] if len(ab) > 1 else ab[0]) * CAMVU_APERTURE_MM
                    apertures[aid] = (shape, a, b)
                    meta = {"shape": shape, "a": a, "b": b, "base_shape": None,
                            "base_a": None, "base_b": None, "angle_deg": 0.0}
                    # Example from CAM350:
                    # 27 rect3.200x4.200_45 2616296:2616296 1 1 Rectangle
                    #    1600000:2100000 45.000000
                    # The 2.616296 value is only the rotated bounding box.  The actual
                    # flash is a 1.6 x 2.1 base rectangle (later scaled for PADS) at 45°.
                    if len(t) >= 8 and str(t[5]).upper() != "NA":
                        try:
                            bab = t[6].split(":")
                            meta["base_shape"] = str(t[5])
                            meta["base_a"] = float(bab[0]) * CAMVU_APERTURE_MM
                            meta["base_b"] = float(bab[1] if len(bab) > 1 else bab[0]) * CAMVU_APERTURE_MM
                            meta["angle_deg"] = float(t[7])
                        except Exception:
                            pass
                    aperture_meta[aid] = meta
                    aperture_order.append(aid)
                except Exception:
                    pass
            continue

        if s.startswith("NC_TOOL "):
            t = s.split()
            # CAMvu stores NC diameter in the 5th token in the same internal coordinate units.
            if len(t) >= 5:
                try:
                    nc_tools[int(t[1])] = float(t[4]) * CAMVU_COORD_MM
                except Exception:
                    pass
            continue

        if s == "*GRAPHIC":
            in_graph = True
            continue
        if s.startswith("*END_GRAPHIC"):
            in_graph = False
            continue
        if in_graph:
            if s.startswith("LAYER "):
                layer = int(s.split()[1]); primitives.setdefault(layer, [])
                continue
            if s.startswith("APERID "):
                aper = int(s.split()[1])
                continue
            if s.startswith("WIDTH "):
                try: width_mm = float(s.split()[1]) * CAMVU_COORD_MM
                except Exception: width_mm = -1.0
                continue
            if s == "POLYVOIDS":
                poly_void = True
                continue
            if s == "ENDPOLYVOIDS":
                poly_void = False
                continue
            if s.startswith("LINE ") or s.startswith("POLYGON "):
                kind = "line" if s.startswith("LINE ") else "polygon"
                t = s.split()
                try:
                    n = int(t[1]); vals = t[2:]
                except Exception:
                    continue
                need = 2 * n + 1  # final 0 terminator
                while len(vals) < need and i < len(lines):
                    vals += lines[i].strip().split(); i += 1
                try:
                    xy = np.asarray(list(map(float, vals[:2*n])), np.float64).reshape(n, 2) * CAMVU_COORD_MM
                except Exception:
                    continue
                if layer is not None:
                    primitives.setdefault(layer, []).append((kind, xy, int(aper), float(width_mm), bool(poly_void)))
                continue

        if s.startswith("*PADS "):
            in_pads = True
            continue
        if s == "*END_PADS":
            in_pads = False
            continue
        if in_pads:
            t = s.split()
            if len(t) >= 5:
                try:
                    pads.append((float(t[0]) * CAMVU_COORD_MM, float(t[1]) * CAMVU_COORD_MM,
                                 int(t[2]), int(t[3])))
                except Exception:
                    pass
            continue

        if s.startswith("PLATED_DRILL_PATH "):
            t = s.split()
            if len(t) >= 3:
                drill_layer = int(t[1]); drill_tool = int(t[2])
            continue
        if s.startswith("NC_LOC HIT "):
            t = s.split()
            if len(t) >= 4 and drill_tool is not None:
                try:
                    x = float(t[2]) * CAMVU_COORD_MM; y = float(t[3]) * CAMVU_COORD_MM
                    holes.append({"x": x, "y": y, "diameter": float(nc_tools.get(drill_tool, 0.0)),
                                  "tool": int(drill_tool), "layer": drill_layer})
                except Exception:
                    pass
            continue
        if s.startswith("MILL_SLOT "):
            t = s.split()
            if len(t) >= 12:
                try:
                    tool = int(t[2]); x0, y0, x1, y1 = map(float, t[-4:])
                    slots.append({"x0": x0 * CAMVU_COORD_MM, "y0": y0 * CAMVU_COORD_MM,
                                  "x1": x1 * CAMVU_COORD_MM, "y1": y1 * CAMVU_COORD_MM,
                                  "diameter": float(nc_tools.get(tool, 0.0)), "tool": tool})
                except Exception:
                    pass
            continue

    return {
        "path": str(path), "layers": layers, "layer_maps": layer_maps,
        "apertures": apertures, "aperture_meta": aperture_meta, "aperture_order": aperture_order, "primitives": primitives, "pads": pads,
        "nc_tools": nc_tools, "holes": holes, "slots": slots, "notes": notes,
        "coord_mm": CAMVU_COORD_MM,
    }

def camvu_copper_layers(cam: dict[str, Any]) -> list[tuple[int, str]]:
    """Return likely copper layers in a stable Top/Bottom-first order."""
    out = []
    for lid, name in cam.get("layers", {}).items():
        low = name.lower()
        if ("top" in low or "bottom" in low or low.startswith("bot")) and not any(k in low for k in ("silk", "msk", "mask")):
            out.append((int(lid), str(name)))
    if not out:
        out = [(int(lid), str(name)) for lid, name in cam.get("layers", {}).items() if int(lid) in cam.get("primitives", {})]
    out.sort(key=lambda z: (0 if "top" in z[1].lower() else 1 if ("bottom" in z[1].lower() or z[1].lower().startswith("bot")) else 2, z[0]))
    return out

def camvu_layer_id(cam: dict[str, Any], layer: int | str | None) -> int:
    if layer is None:
        cands = camvu_copper_layers(cam)
        if not cands:
            raise RuntimeError("CAM: copper layer not found")
        return int(cands[0][0])
    if isinstance(layer, (int, np.integer)):
        return int(layer)
    ss = str(layer).strip()
    if ss.lstrip("-").isdigit():
        return int(ss)
    for lid, name in cam.get("layers", {}).items():
        if name == ss:
            return int(lid)
    for lid, name in cam.get("layers", {}).items():
        if name.lower() == ss.lower():
            return int(lid)
    raise RuntimeError(f"CAM layer not found: {layer}")

def camvu_opposite_layer(cam: dict[str, Any], layer_id: int) -> int | None:
    lm = cam.get("layer_maps", {})
    if int(layer_id) in lm and int(lm[int(layer_id)]) != int(layer_id):
        return int(lm[int(layer_id)])
    name = cam.get("layers", {}).get(int(layer_id), "").lower()
    cands = camvu_copper_layers(cam)
    if "top" in name:
        for lid, nm in cands:
            if "bottom" in nm.lower() or nm.lower().startswith("bot"):
                return int(lid)
    if "bottom" in name or name.startswith("bot"):
        for lid, nm in cands:
            if "top" in nm.lower():
                return int(lid)
    return None

def _draw_cam_aperture(canvas: np.ndarray, cam: dict[str, Any], aid: int, xy: tuple[float, float],
                       H: np.ndarray, mirror: bool, ss: int, value: int = 255) -> None:
    ap = cam.get("apertures", {}).get(int(aid))
    if ap is None:
        return
    shape, a, b = ap
    x, y = xy
    P = lambda p: map_xy(H, mirror, p) * ss
    sh = str(shape).lower()
    if sh.startswith("round"):
        # project the circle rather than assuming isotropic pixels
        ang = np.linspace(0, 2*np.pi, 48, endpoint=False)
        pts = [(x + 0.5*a*math.cos(t), y + 0.5*a*math.sin(t)) for t in ang]
        poly = np.rint(np.asarray([P(p) for p in pts], np.float32)).astype(np.int32)
        cv2.fillConvexPoly(canvas, poly, int(value), lineType=cv2.LINE_AA)
    else:
        corners = [(x-a/2, y-b/2), (x+a/2, y-b/2), (x+a/2, y+b/2), (x-a/2, y+b/2)]
        pts = np.rint(np.asarray([P(p) for p in corners], np.float32)).astype(np.int32)
        cv2.fillConvexPoly(canvas, pts, int(value), lineType=cv2.LINE_AA)

def _draw_cam_rect_sweep(canvas: np.ndarray, xy: np.ndarray, a_mm: float, b_mm: float,
                         H: np.ndarray, mirror: bool, ss: int, value: int = 255) -> None:
    """Sweep a fixed-orientation rectangular CAM aperture along a polyline.

    CAM350 stores many wide tracks and SMD copper objects as LINE records drawn with
    a rectangular aperture.  Drawing those records with ``min(a,b)`` as a round pen
    makes the result too thin.  The correct geometry is the Minkowski sum of every
    segment with the axis-aligned aperture rectangle, i.e. the convex hull of the
    rectangle placed at both segment ends.
    """
    if len(xy) < 2:
        return
    hx = 0.5 * float(a_mm)
    hy = 0.5 * float(b_mm)
    offsets = np.asarray([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], np.float64)
    for p0, p1 in zip(xy[:-1], xy[1:]):
        corners = np.vstack((p0 + offsets, p1 + offsets))
        pts = np.asarray([map_xy(H, mirror, tuple(p)) * ss for p in corners], np.float32)
        hull = cv2.convexHull(np.rint(pts).astype(np.int32))
        if hull is not None and len(hull) >= 3:
            cv2.fillConvexPoly(canvas, hull, int(value), lineType=cv2.LINE_AA)


def _camvu_use_legacy_draw_profile(cam: dict[str, Any]) -> bool:
    """Detect a legacy CAM350 aperture-ID convention by content signature.

    Some older text databases use GRAPHIC APERID values as draw-style references
    rather than literal physical aperture sizes.  This check deliberately depends
    on aperture content, not on a board name, project name, path, or filename.
    """
    aps = cam.get("apertures", {})
    a35 = aps.get(35); a37 = aps.get(37); a41 = aps.get(41)
    if not (a35 and a37 and a41):
        return False
    try:
        return (str(a35[0]).lower().startswith("round") and abs(float(a35[1]) - 0.925) < 0.03
                and str(a37[0]).lower().startswith("rect") and abs(min(float(a37[1]), float(a37[2])) - 1.525) < 0.04
                and str(a41[0]).lower().startswith("round") and abs(float(a41[1]) - 0.525) < 0.03)
    except Exception:
        return False

def _camvu_draw_width_mm(cam: dict[str, Any], aid: int, ap_shape: str, a_mm: float, b_mm: float) -> float:
    """Return physical routed-track width for this CAM database."""
    if _camvu_use_legacy_draw_profile(cam):
        w = CAMVU_DRAW_WIDTH_LEGACY_PROFILE.get(int(aid))
        if w is not None:
            return float(w)
    sh = str(ap_shape).lower()
    return float(a_mm) if sh.startswith("round") else float(min(a_mm, b_mm))

def _camvu_pad_geometry(cam: dict[str, Any], aid: int) -> tuple[str, float, float, float]:
    """Return (shape, a_mm, b_mm, angle_deg) for a PADS flash.

    Custom rotated CAM350 apertures encode the real base rectangle and its angle
    separately from the bounding-box dimensions.
    """
    ap = cam.get("apertures", {}).get(int(aid))
    if ap is None:
        return ("", 0.0, 0.0, 0.0)
    shape_name, pa, pb = ap
    meta = cam.get("aperture_meta", {}).get(int(aid), {})
    if meta.get("base_shape") and meta.get("base_a") is not None:
        shape_name = str(meta.get("base_shape"))
        pa = float(meta.get("base_a")); pb = float(meta.get("base_b"))
        ang = float(meta.get("angle_deg", 0.0))
    else:
        ang = 0.0
    return (str(shape_name), float(pa) * CAMVU_PAD_APERTURE_FACTOR,
            float(pb) * CAMVU_PAD_APERTURE_FACTOR, ang)

def render_camvu_layer_to_scan(cam_or_path: dict[str, Any] | str | Path, layer: int | str, shape,
                               H: np.ndarray, mirror: bool, supersample: int | None = None,
                               exclude_round_pth_pads: bool = False) -> np.ndarray:
    """Rasterise one CAMvu/CAM350 copper layer directly into scan coordinates.

    CAM350's GRAPHIC records use real aperture sweeps.  Round apertures are rendered
    as round strokes, while rectangular/square apertures are swept as rectangles.
    APERID 0 means that no real copper aperture is assigned; older prototypes drew
    those records as an arbitrary 0.10 mm line, which produced the thin red artefacts
    visible around some component pads.  Such records are now ignored for Cu overlay.
    """
    cam = parse_camvu(cam_or_path) if not isinstance(cam_or_path, dict) else cam_or_path
    lid = camvu_layer_id(cam, layer)
    h, w = shape[:2]
    if supersample is None:
        mp = (h*w)/1e6
        supersample = 4 if mp <= 3.0 else (2 if mp <= 12.0 else 1)
    ss = max(1, int(supersample))
    canvas = np.zeros((h*ss, w*ss), np.uint8)
    aps = cam.get("apertures", {})

    def P(p):
        return map_xy(H, mirror, p) * ss

    for kind, xy, aid, width_mm, is_void in cam.get("primitives", {}).get(lid, []):
        value = 0 if is_void else 255
        if kind == "polygon":
            pts = np.rint(np.asarray([P(tuple(p)) for p in xy], np.float32)).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(canvas, [pts], int(value), lineType=cv2.LINE_AA)
            continue

        if int(aid) == 0:
            # CAM350 omits the aperture table entry for one legitimate copper
            # track class.  In this project APERID 0 is consistently the 1.5 mm
            # routed conductor class (including the C17 connections).
            ap_shape, a_mm, b_mm = "Round", CAMVU_APER0_DRAW_MM, CAMVU_APER0_DRAW_MM
        else:
            ap = aps.get(int(aid))
            if ap is None:
                continue
            ap_shape, a_mm, b_mm = ap
        sh = str(ap_shape).lower()

        # In this CAM350 database a rectangular aperture used by a LINE is a
        # routed draw, not a fixed-axis rectangle swept across the whole path.
        # The routed conductor width is the smaller aperture dimension.  Treating
        # the full rectangle as a fixed-axis sweep makes parallel tracks merge.
        mid = tuple(np.mean(xy, axis=0))
        pxmm = _local_px_per_mm(H, mirror, mid)
        # APERID is a CAM350 draw-style reference.  Some IDs point at
        # rectangular pad apertures whose dimensions are *not* the routed track
        # width.  Use the decoded draw-width profile when available.
        draw_w_mm = _camvu_draw_width_mm(cam, int(aid), ap_shape, float(a_mm), float(b_mm))
        wp = max(1, int(round(draw_w_mm * pxmm * ss)))
        pts = np.rint(np.asarray([P(tuple(p)) for p in xy], np.float32)).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, int(value), wp, lineType=cv2.LINE_AA)
            rr = max(1, int(round(0.5 * draw_w_mm * pxmm * ss)))
            cv2.circle(canvas, tuple(pts[0]), rr, int(value), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts[-1]), rr, int(value), -1, lineType=cv2.LINE_AA)

    # CAM350 stores component/SMD/edge-contact flashes in *PADS separately from
    # the routed GRAPHIC draws.  Important format detail: the fourth PADS field is
    # NOT an APERTURES1 ID.  It is a zero-based index into the aperture table in
    # file order.  Treating it as the literal ID caused merged transistor pads,
    # incomplete edge fingers and many missing flashes.
    #
    # PADS dimensions use the coordinate aperture scale (2x the GRAPHIC draw
    # scale on this database), so draw them with CAMVU_PAD_APERTURE_FACTOR.
    hole_xy = None
    if exclude_round_pth_pads:
        hh = [(float(z["x"]), float(z["y"])) for z in cam.get("holes", [])]
        if hh:
            hole_xy = np.asarray(hh, np.float64)

    ap_order = cam.get("aperture_order", [])
    for x, y, pad_layer, pad_ap_index in cam.get("pads", []):
        if int(pad_layer) != int(lid):
            continue
        idx = int(pad_ap_index)
        if idx < 0 or idx >= len(ap_order):
            continue
        aid = int(ap_order[idx])
        ap = aps.get(aid)
        if ap is None:
            continue
        shape_name, pa, pb, pad_angle_deg = _camvu_pad_geometry(cam, aid)
        px, py = float(x), float(y)
        sh2 = str(shape_name).lower()
        if exclude_round_pth_pads and hole_xy is not None:
            d2 = np.sum((hole_xy - np.asarray([px, py], np.float64))**2, axis=1)
            # Display-only suppression of *pad flashes* centred on plated drills.
            # Keep the routed conductors that approach the pad, but do not paint the
            # annular/rectangular PTH copper over the real hole in the scan.
            if len(d2) and float(np.min(d2)) <= 0.12**2:
                continue
        if sh2.startswith("round"):
            ang = np.linspace(0, 2*np.pi, 48, endpoint=False)
            poly = np.rint(np.asarray([P((px + 0.5*pa*math.cos(t), py + 0.5*pa*math.sin(t)))
                                       for t in ang], np.float32)).astype(np.int32)
            if len(poly) >= 3:
                cv2.fillConvexPoly(canvas, poly, 255, lineType=cv2.LINE_AA)
        else:
            local = np.asarray([[-pa/2, -pb/2], [pa/2, -pb/2],
                                [pa/2, pb/2], [-pa/2, pb/2]], np.float64)
            if abs(float(pad_angle_deg)) > 1e-9:
                th = math.radians(float(pad_angle_deg))
                Rr = np.asarray([[math.cos(th), -math.sin(th)],
                                 [math.sin(th),  math.cos(th)]], np.float64)
                local = local @ Rr.T
            corners = [(px + float(q[0]), py + float(q[1])) for q in local]
            poly = np.rint(np.asarray([P(q) for q in corners], np.float32)).astype(np.int32)
            cv2.fillConvexPoly(canvas, poly, 255, lineType=cv2.LINE_AA)

    if ss > 1:
        return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return canvas


def suppress_cam_pth_overlay(copper_mask: np.ndarray, cam: dict[str, Any], H: np.ndarray, mirror: bool,
                             extra_radius_mm: float = 0.04) -> np.ndarray:
    """Display-only mask that reveals the physical drill bore without erasing pads/tracks.

    Older versions removed ``hole radius + 0.70 mm`` around every PTH.  That hid
    entire edge fingers and short conductors.  We now remove only the actual drill
    opening plus a tiny optical margin.  Copper analysis always uses the unmodified
    full mask.
    """
    out = copper_mask.copy()
    extra = max(0.0, float(extra_radius_mm))
    for h in cam.get("holes", []):
        d = float(h.get("diameter", 0.0))
        if d <= 0:
            continue
        xy = (float(h["x"]), float(h["y"]))
        p = map_xy(H, mirror, xy)
        ppm = _local_px_per_mm(H, mirror, xy)
        rr = max(1, int(round((0.5*d + extra) * ppm)))
        cv2.circle(out, tuple(np.rint(p).astype(int)), rr, 0, -1, lineType=cv2.LINE_AA)
    return out

def apply_cam_thermal_display_assist(copper_mask: np.ndarray, cam: dict[str, Any], H: np.ndarray,
                                     mirror: bool, copper_layer_id: int) -> tuple[np.ndarray, int]:
    """Display-only approximation of thermal reliefs around round PTH pads in broad copper.

    Some CAM350 text database exports do not expose a distinct thermal-aperture object in
    the parsed records.  For operator overlay only, identify round plated pads that sit inside
    a nearly solid copper neighbourhood and render a conventional four-spoke relief.  The AOI
    comparison mask remains untouched, so this helper cannot hide a real copper defect.
    """
    src = copper_mask.copy(); out = copper_mask.copy(); lid = int(copper_layer_id)
    meta = camvu_hole_metadata(cam, lid)
    ap_order = cam.get("aperture_order", []); aps = cam.get("apertures", {})
    # Exact pad lookup for this copper side.
    pads = []
    for x,y,pl,idx in cam.get("pads", []):
        if int(pl) != lid or not (0 <= int(idx) < len(ap_order)): continue
        aid = int(ap_order[int(idx)]); ap = aps.get(aid)
        if ap is not None: pads.append((float(x),float(y),ap))
    made = 0
    hgt,wid = out.shape[:2]
    for h in meta:
        if h.get("category") != "PTH_OPEN": continue
        x,y=float(h["x"]),float(h["y"]); d=float(h.get("diameter",0.0))
        near=[p for p in pads if (p[0]-x)**2+(p[1]-y)**2 <= 0.10**2]
        if not near: continue
        _,_,ap=near[0]; sh,pa,pb=ap
        if not str(sh).lower().startswith("round"): continue
        pad_d = float(pa)*CAMVU_PAD_APERTURE_FACTOR
        if pad_d <= d + 0.20: continue
        ppm=_local_px_per_mm(H,mirror,(x,y)); pc=map_xy(H,mirror,(x,y))
        cx,cy=np.rint(pc).astype(int); pad_r=0.5*pad_d*ppm
        # Require a broad copper field around the pad; ordinary isolated pads/tracks are left alone.
        yy0=max(0,int(cy-pad_r-0.65*ppm)); yy1=min(hgt,int(cy+pad_r+0.65*ppm)+1)
        xx0=max(0,int(cx-pad_r-0.65*ppm)); xx1=min(wid,int(cx+pad_r+0.65*ppm)+1)
        if yy1<=yy0 or xx1<=xx0: continue
        yy,xx=np.indices((yy1-yy0,xx1-xx0)); rr=np.hypot(xx+(xx0-cx),yy+(yy0-cy))
        ring=(rr >= pad_r+0.10*ppm) & (rr <= pad_r+0.45*ppm)
        if ring.sum()<10: continue
        coverage=float(np.mean(src[yy0:yy1,xx0:xx1][ring] > 127))
        if coverage < 0.78: continue
        inner=max(0.5*d*ppm+0.05*ppm, pad_r*0.88)
        outer=pad_r+0.28*ppm
        ann=(rr>=inner)&(rr<=outer)
        loc=out[yy0:yy1,xx0:xx1]; orig=src[yy0:yy1,xx0:xx1]
        loc[ann]=0
        # Restore four orthogonal copper spokes, only where copper existed originally.
        spoke=max(1.0,0.16*ppm)
        dx=np.abs(xx+(xx0-cx)); dy=np.abs(yy+(yy0-cy))
        spokes=ann & ((dy<=spoke) | (dx<=spoke))
        loc[spokes]=orig[spokes]
        made += 1
    return out, made


def camvu_drill_tuples(cam_or_path: dict[str, Any] | str | Path) -> list[tuple[float, float, float, int]]:
    cam = parse_camvu(cam_or_path) if not isinstance(cam_or_path, dict) else cam_or_path
    return [(float(h["x"]), float(h["y"]), float(h.get("diameter", 0.0)), int(h.get("tool", 0)))
            for h in cam.get("holes", []) if float(h.get("diameter", 0.0)) > 0]

def camvu_mask_layer_for_copper(cam: dict[str, Any], copper_layer_id: int) -> int | None:
    """Return the solder-mask layer that belongs to a copper side, when identifiable."""
    name = str(cam.get("layers", {}).get(int(copper_layer_id), "")).lower()
    want_top = ("top" in name) or name.endswith("tp")
    want_bot = ("bottom" in name) or ("bot" in name) or name.endswith("bt")
    cands = []
    for lid, nm in cam.get("layers", {}).items():
        low = str(nm).lower()
        if "msk" not in low and "mask" not in low:
            continue
        score = 0
        if want_top and ("tp" in low or "top" in low): score += 3
        if want_bot and ("bt" in low or "bot" in low or "bottom" in low): score += 3
        if score: cands.append((score, int(lid)))
    if cands:
        return max(cands)[1]
    # Some CAM350 exports map Top/Bottom copper to mask layers in pairs.
    mp = cam.get("layer_maps", {})
    for lid, nm in cam.get("layers", {}).items():
        low = str(nm).lower()
        if ("msk" in low or "mask" in low) and int(mp.get(int(lid), -999)) == int(lid):
            return int(lid)
    return None


def camvu_hole_metadata(cam: dict[str, Any], copper_layer_id: int) -> list[dict[str, Any]]:
    """Classify CAM drill hits for the selected side.

    Categories are derived from information that is actually present in the database:
      * NPTH      — drill hit is on the non-plated drill layer (e.g. DRILL_N);
      * VIA_TENTED — plated hit without a solder-mask opening on this side;
      * PTH_OPEN  — plated hit with a solder-mask opening on this side.

    The mask-open test intentionally uses the CAM PADS coordinates rather than the scan,
    so optical shadows cannot change the engineering category.
    """
    mask_lid = camvu_mask_layer_for_copper(cam, int(copper_layer_id))
    mask_pts = np.asarray([[float(x), float(y)] for x, y, lid, ap in cam.get("pads", [])
                           if mask_lid is not None and int(lid) == int(mask_lid)], np.float64)
    layer_names = cam.get("layers", {})
    out = []
    for h in cam.get("holes", []):
        z = dict(h)
        drill_name = str(layer_names.get(int(h.get("layer", -1)), "")).upper()
        # CAM350 may still write the record keyword PLATED_DRILL_PATH for an NPTH
        # drill layer; the layer name is the reliable discriminator in this database.
        is_npth = ("DRILL_N" in drill_name) or ("NPTH" in drill_name) or ("NON" in drill_name and "PLAT" in drill_name)
        mask_open = False
        if not is_npth and mask_pts.size:
            q = np.asarray([float(h["x"]), float(h["y"])], np.float64)
            mask_open = bool(np.min(np.linalg.norm(mask_pts - q, axis=1)) <= 0.16)
        if is_npth:
            cat = "NPTH"
            plated_expected = False
        elif mask_open:
            cat = "PTH_OPEN"
            plated_expected = True
        else:
            cat = "VIA_TENTED"
            plated_expected = True
        z.update({"category": cat, "mask_open": bool(mask_open), "plated_expected": bool(plated_expected),
                  "drill_layer_name": drill_name, "mask_layer_id": mask_lid})
        out.append(z)
    return out


def _hole_plating_score(img: np.ndarray, center: np.ndarray, radius_px: float,
                        prepared: dict[str, np.ndarray] | None = None) -> tuple[float | None, dict[str, float]]:
    """Very conservative visual hint for a metallic bore wall.

    This is deliberately *not* a pass/fail test.  It looks for a dark, relatively
    low-saturation circumferential wall between the bright bore centre and the PCB
    surface.  Shadows can imitate this cue, especially in large NPTH holes, so the
    result is exposed as advisory evidence only.
    """
    R = float(radius_px)
    if R < 2.3:
        return None, {}
    if prepared is not None and "L" in prepared and "S" in prepared:
        L = prepared["L"].astype(np.float32, copy=False)
        S = prepared["S"].astype(np.float32, copy=False)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        L = lab[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
    th = np.linspace(0, 2*np.pi, 48, endpoint=False)
    ca, sa = np.cos(th), np.sin(th)
    def sample(arr, rr):
        xs = center[0] + rr * ca; ys = center[1] + rr * sa
        return cv2.remap(arr, xs.astype(np.float32).reshape(-1,1), ys.astype(np.float32).reshape(-1,1),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1)
    core = sample(L, 0.24*R)
    core_l = float(np.median(core))
    rings_l = np.vstack([sample(L, f*R) for f in (0.55, 0.68, 0.80, 0.90)])
    rings_s = np.vstack([sample(S, f*R) for f in (0.55, 0.68, 0.80, 0.90)])
    idx = np.argmin(rings_l, axis=0)
    wall_l = rings_l[idx, np.arange(rings_l.shape[1])]
    wall_s = rings_s[idx, np.arange(rings_s.shape[1])]
    dark = core_l - wall_l
    coverage = float(np.mean(dark > 16.0))
    wall_sat = float(np.median(wall_s))
    sat_term = float(np.clip((220.0 - wall_sat) / 90.0, 0.0, 1.0))
    score = float(np.clip(0.68*coverage + 0.32*sat_term, 0.0, 1.0))
    return score, {"dark_coverage": coverage, "wall_saturation": wall_sat, "core_L": core_l}


def _update_hole_summary(hole_check: dict[str, Any]) -> None:
    from collections import defaultdict
    groups = defaultdict(lambda: {"expected":0,"visible_expected":0,"found":0,"OK":0,"WARN":0,"SHIFT":0,"UNKNOWN":0,"OUTSIDE":0,
                                  "metal_likely":0,"metal_suspect":0,"metal_unknown":0})
    cat_counts = defaultdict(int)
    plating_counts = defaultdict(int)
    for it in hole_check.get("items", []):
        cat = str(it.get("category", "PTH_OPEN"))
        dia = round(float(it.get("diameter_mm",0.0)), 2)
        g = groups[(cat,dia)]; g["expected"] += 1; cat_counts[cat] += 1
        st = str(it.get("status","UNKNOWN")); g[st] += 1
        if st != "OUTSIDE":
            g["visible_expected"] += 1
            if st != "UNKNOWN": g["found"] += 1
        pv = str(it.get("plating_visual","NA"))
        if pv == "LIKELY":
            g["metal_likely"] += 1
            if cat == "PTH_OPEN": plating_counts["LIKELY"] += 1
        elif pv == "SUSPECT":
            g["metal_suspect"] += 1
            if cat == "PTH_OPEN": plating_counts["SUSPECT"] += 1
            elif cat == "NPTH": plating_counts["NPTH_METAL_LIKE"] += 1
        elif pv == "UNKNOWN":
            g["metal_unknown"] += 1
            if cat == "PTH_OPEN": plating_counts["UNKNOWN"] += 1
    label_order = {"VIA_TENTED":0,"PTH_OPEN":1,"NPTH":2}
    hole_check["groups"] = [{"category":k[0],"diameter_mm":k[1],**v}
                            for k,v in sorted(groups.items(), key=lambda kv:(label_order.get(kv[0][0],9),kv[0][1]))]
    hole_check["category_counts"] = dict(cat_counts)
    hole_check["plating_counts"] = dict(plating_counts)

def _cam_primary_anchors(cam: dict[str, Any]) -> tuple[list[tuple[float,float,float]], float | None]:
    cand = [(float(h["x"]), float(h["y"]), float(h.get("diameter", 0.0)))
            for h in cam.get("holes", []) if float(h.get("diameter", 0.0)) >= 1.5]
    anchors: list[tuple[float,float,float]] = []
    primary_diam = None
    if cand:
        ds = sorted({round(float(d), 3) for x,y,d in cand}, reverse=True)
        allxy = np.asarray([[x,y] for x,y,d in cand], np.float64)
        fullx = max(1e-6, float(np.ptp(allxy[:,0]))); fully = max(1e-6, float(np.ptp(allxy[:,1])))
        for dd in ds:
            grp = [(x,y,d) for x,y,d in cand if abs(float(d)-dd) < 0.03]
            if len(grp) < 4: continue
            gxy = np.asarray([[x,y] for x,y,d in grp], np.float64)
            if float(np.ptp(gxy[:,0])) >= 0.45*fullx and float(np.ptp(gxy[:,1])) >= 0.45*fully:
                anchors = grp; primary_diam = dd; break
    if len(anchors) < 4:
        anchors = [(x,y,d) for x,y,d in cand if d >= 2.5]
    if len(anchors) < 4:
        anchors = cand
    return anchors, primary_diam

def _circle_edge_localizer(img: np.ndarray, pred: np.ndarray, radius_px: float, px_per_mm: float,
                           prepared: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, float, float]:
    """Small-hole localizer using circular edge energy near an already accurate prediction.

    Returns (center, confidence, score). It intentionally searches only a few pixels so pads,
    silkscreen and neighbouring holes cannot hijack the expected CAM drill position.
    """
    if prepared is not None and all(k in prepared for k in ("L","S","mag")):
        gray = prepared["L"]; sat = prepared["S"]; mag = prepared["mag"]
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.GaussianBlur(lab[:,:,0], (3,3), 0).astype(np.float32)
        sat = cv2.GaussianBlur(hsv[:,:,1], (3,3), 0).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
    angles = np.linspace(0, 2*np.pi, 40, endpoint=False); ca, sa = np.cos(angles), np.sin(angles)
    R = max(2.0, float(radius_px))
    search = max(1.25, min(4.0, 0.45*float(px_per_mm)))
    step = 0.5

    def sample(arr, xs, ys):
        return cv2.remap(arr, np.asarray(xs,np.float32).reshape(-1,1), np.asarray(ys,np.float32).reshape(-1,1),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1)
    vals = []
    for dy in np.arange(-search, search+1e-6, step):
        for dx in np.arange(-search, search+1e-6, step):
            cc = np.asarray(pred, np.float64) + np.array([dx,dy], np.float64)
            edge = 0.0
            for rr in (max(1.1,0.78*R), R, 1.20*R):
                edge = max(edge, float(np.mean(sample(mag, cc[0]+rr*ca, cc[1]+rr*sa))))
            rin = max(0.8, 0.35*R); rout = max(R+1.0, 1.45*R)
            sin = float(np.mean(sample(sat, cc[0]+rin*ca, cc[1]+rin*sa)))
            sout = float(np.mean(sample(sat, cc[0]+rout*ca, cc[1]+rout*sa)))
            lin = float(np.mean(sample(gray, cc[0]+rin*ca, cc[1]+rin*sa)))
            lout = float(np.mean(sample(gray, cc[0]+rout*ca, cc[1]+rout*sa)))
            score = edge + 0.45*abs(sout-sin) + 0.12*abs(lout-lin) - 2.5*math.hypot(dx,dy)
            vals.append((score, cc))
    vals.sort(key=lambda z: -z[0])
    best_score, best_center = vals[0]
    scores = np.asarray([v[0] for v in vals], np.float64)
    q70 = float(np.percentile(scores,70)); spread = float(np.std(scores))+1e-6
    conf = float(np.clip((best_score-q70)/(2.2*spread), 0.0, 1.0))
    return np.asarray(best_center,np.float64), conf, float(best_score)

def evaluate_cam_holes(img: np.ndarray, board: np.ndarray, H: np.ndarray, mirror: bool,
                       cam: dict[str, Any], tolerance_mm: float = 0.35,
                       copper_layer_id: int | None = None) -> dict[str, Any]:
    """Estimate drill centres from the scan and compare them with CAM NC hits.

    This is a positional quality aid. Low-confidence holes are marked UNKNOWN rather than FAIL.
    """
    items = []
    tol = max(0.05, float(tolerance_mm))
    if copper_layer_id is None:
        cands = camvu_copper_layers(cam)
        copper_layer_id = int(cands[0][0]) if cands else 0
    meta_list = camvu_hole_metadata(cam, int(copper_layer_id))
    meta_by_key = {(round(float(z["x"]),4), round(float(z["y"]),4), int(z.get("tool",0))): z for z in meta_list}
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    L = cv2.GaussianBlur(lab[:,:,0], (3,3), 0).astype(np.float32)
    S = cv2.GaussianBlur(hsv[:,:,1], (3,3), 0).astype(np.float32)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    prepared = {"L": L, "S": S, "mag": cv2.magnitude(gx, gy)}
    for h in cam.get("holes", []):
        d = float(h.get("diameter",0.0))
        if d <= 0: continue
        xy = (float(h["x"]), float(h["y"]))
        pred = map_xy(H, mirror, xy)
        ppm = _local_px_per_mm(H, mirror, xy)
        rpx = max(2.0, 0.5*d*ppm)
        # A partial scan is valid input in v0.14.  Holes that are physically outside the
        # scanned board fragment are not missing/failed holes; mark them OUTSIDE and exclude
        # them from positional QA statistics.
        hh, ww = board.shape[:2]
        xi, yi = int(round(float(pred[0]))), int(round(float(pred[1])))
        visible_here = (0 <= xi < ww and 0 <= yi < hh and board[yi, xi] > 0)
        if not visible_here:
            hm = meta_by_key.get((round(xy[0],4), round(xy[1],4), int(h.get("tool",0))), {})
            items.append({
                "x": xy[0], "y": xy[1], "diameter_mm": d, "tool": int(h.get("tool",0)),
                "expected": np.asarray(pred,np.float64), "actual": None,
                "confidence": 0.0, "shift_px": None, "shift_mm": None,
                "status": "OUTSIDE", "method": "outside", "px_per_mm": float(ppm),
                "category": str(hm.get("category","PTH_OPEN")), "mask_open": bool(hm.get("mask_open",False)),
                "plated_expected": bool(hm.get("plated_expected",True)),
                "plating_visual": "NA", "plating_score": None, "plating_features": {},
            })
            continue
        actual = None; conf = 0.0; method = "edge"
        if d >= 1.45:
            actual, conf = _surface_rim_center(img, pred, rpx, board=board, prepared=prepared)
            method = "rim"
        if actual is None or conf < 0.30:
            actual2, conf2, _ = _circle_edge_localizer(img, pred, rpx, ppm, prepared=prepared)
            if actual is None or conf2 > conf:
                actual, conf, method = actual2, conf2, "edge"
        shift_px = None; shift_mm = None
        if actual is not None:
            shift_px = float(np.linalg.norm(np.asarray(actual)-pred))
            shift_mm = shift_px / max(ppm,1e-9)
        # Be conservative: an uncertain optical result is UNKNOWN, not a false reject.
        if actual is None or conf < 0.38:
            status = "UNKNOWN"
        elif shift_mm is not None and shift_mm <= tol:
            status = "OK"
        elif shift_mm is not None and shift_mm <= 1.6*tol:
            status = "WARN"
        else:
            status = "SHIFT"
        hm = meta_by_key.get((round(xy[0],4), round(xy[1],4), int(h.get("tool",0))), {})
        category = str(hm.get("category","PTH_OPEN"))
        plating_score = None; plating_features = {}; plating_visual = "NA"
        if category != "VIA_TENTED" and actual is not None and conf >= 0.30:
            plating_score, plating_features = _hole_plating_score(img, np.asarray(actual,np.float64), rpx, prepared=prepared)
            if plating_score is not None:
                if category == "PTH_OPEN":
                    plating_visual = "LIKELY" if plating_score >= 0.50 else ("SUSPECT" if plating_score <= 0.20 else "UNKNOWN")
                else:
                    # For NPTH this is only a note that the optical rim looks metal-like.
                    plating_visual = "SUSPECT" if plating_score >= 0.78 else "UNKNOWN"
        items.append({
            "x": xy[0], "y": xy[1], "diameter_mm": d, "tool": int(h.get("tool",0)),
            "expected": np.asarray(pred,np.float64), "actual": None if actual is None else np.asarray(actual,np.float64),
            "confidence": float(conf), "shift_px": shift_px, "shift_mm": shift_mm,
            "status": status, "method": method, "px_per_mm": float(ppm),
            "category": category, "mask_open": bool(hm.get("mask_open",False)),
            "plated_expected": bool(hm.get("plated_expected",True)),
            "plating_visual": plating_visual, "plating_score": plating_score,
            "plating_features": plating_features,
        })
    counts = {k: sum(1 for it in items if it["status"]==k) for k in ("OK","WARN","SHIFT","UNKNOWN","OUTSIDE")}
    valid = [it["shift_mm"] for it in items if it["shift_mm"] is not None and it["status"] != "UNKNOWN"]
    result = {
        "items": items, "counts": counts, "tolerance_mm": tol,
        "median_shift_mm": None if not valid else float(np.median(valid)),
        "max_shift_mm": None if not valid else float(np.max(valid)),
    }
    _update_hole_summary(result)
    return result

def reclassify_hole_check(hole_check: dict[str, Any] | None, tolerance_mm: float) -> dict[str, Any] | None:
    if not hole_check:
        return hole_check
    tol = max(0.05, float(tolerance_mm))
    for it in hole_check.get("items", []):
        if str(it.get("status","")) == "OUTSIDE" or str(it.get("method","")) == "outside":
            it["status"] = "OUTSIDE"
            continue
        conf = float(it.get("confidence",0.0)); shift = it.get("shift_mm")
        if shift is None or conf < 0.38:
            st = "UNKNOWN"
        elif float(shift) <= tol:
            st = "OK"
        elif float(shift) <= 1.6*tol:
            st = "WARN"
        else:
            st = "SHIFT"
        it["status"] = st
    hole_check["tolerance_mm"] = tol
    hole_check["counts"] = {k: sum(1 for it in hole_check.get("items",[]) if it.get("status")==k)
                            for k in ("OK","WARN","SHIFT","UNKNOWN","OUTSIDE")}
    valid = [it.get("shift_mm") for it in hole_check.get("items",[])
             if it.get("shift_mm") is not None and it.get("status") != "UNKNOWN"]
    hole_check["median_shift_mm"] = None if not valid else float(np.median(valid))
    hole_check["max_shift_mm"] = None if not valid else float(np.max(valid))
    _update_hole_summary(hole_check)
    return hole_check

def hole_overlay(img: np.ndarray, hole_check: dict[str, Any] | None, show_actual: bool = True) -> np.ndarray:
    """Overlay hole position and engineering type without hiding the bore.

    OK colors encode CAM type; positional/optical problems override the type color:
      green  = open plated PTH, violet = tented plated via, cyan = NPTH,
      orange = position warning, red = shifted, yellow = optically uncertain.
    """
    if not hole_check:
        return img.copy()
    out = img.copy()
    ok_colors = {"PTH_OPEN": (0,220,0), "VIA_TENTED": (220,70,220), "NPTH": (255,220,0)}
    problem_colors = {"WARN": (0,165,255), "SHIFT": (0,0,255), "UNKNOWN": (0,230,230), "OUTSIDE": (150,150,150)}
    for it in hole_check.get("items", []):
        p = np.asarray(it["expected"], np.float64)
        ppm = float(it.get("px_per_mm",1.0)); d = float(it.get("diameter_mm",0.0))
        bore_r = max(2, int(round(0.5*d*ppm)))
        ring_r = bore_r + max(2, int(round(0.16*ppm)))
        st=str(it.get("status","UNKNOWN")); cat=str(it.get("category","PTH_OPEN"))
        if st == "OUTSIDE":
            continue
        c = ok_colors.get(cat,(0,220,0)) if st=="OK" else problem_colors.get(st,(0,230,230))
        pc = tuple(np.rint(p).astype(int))
        cv2.circle(out, pc, ring_r, c, 2, cv2.LINE_AA)
        if show_actual and it.get("actual") is not None and st in {"WARN","SHIFT","UNKNOWN"}:
            aa = np.asarray(it["actual"], np.float64); a = tuple(np.rint(aa).astype(int))
            cv2.circle(out, a, 2, c, 1, cv2.LINE_AA)
            if st in {"WARN","SHIFT"}: cv2.line(out, pc, a, c, 1, cv2.LINE_AA)
        # A tiny white tick marks an advisory plating concern without recoloring the hole type.
        if st=="OK" and it.get("category")=="PTH_OPEN" and it.get("plating_visual")=="SUSPECT":
            cv2.line(out,(pc[0]-2,pc[1]-ring_r-3),(pc[0]+2,pc[1]-ring_r-3),(255,255,255),2,cv2.LINE_AA)
    cnt = hole_check.get("counts", {})
    cats = hole_check.get("category_counts", {})
    label = (f"PTH {cats.get('PTH_OPEN',0)}  VIA(mask) {cats.get('VIA_TENTED',0)}  NPTH {cats.get('NPTH',0)}   "
             f"WARN {cnt.get('WARN',0)} SHIFT {cnt.get('SHIFT',0)} ? {cnt.get('UNKNOWN',0)} OUT {cnt.get('OUTSIDE',0)}")
    cv2.putText(out, label, (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
    return out


def copper_evidence(img: np.ndarray, gerber: np.ndarray, board: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    blur4 = cv2.GaussianBlur(L, (0, 0), 4)
    blur16 = cv2.GaussianBlur(L, (0, 0), 16)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    F = np.stack([L, a, b, L - blur4, L - blur16, np.log1p(grad)], axis=2)
    g = (gerber > 127).astype(np.uint8)
    pos = cv2.erode(g, np.ones((5, 5), np.uint8)) > 0
    neg = cv2.dilate(g, np.ones((9, 9), np.uint8)) == 0
    valid = (board > 0) & (hsv[:, :, 1] > 55) & (L > 45) & (L < 190)
    pos &= valid
    neg &= valid
    P, N = F[pos], F[neg]
    if len(P) < 500 or len(N) < 500:
        raise RuntimeError("Not enough pixels to calibrate copper appearance")
    rng = np.random.default_rng(1)
    n = min(40000, len(P), len(N))
    P = P[rng.choice(len(P), n, replace=False)]
    N = N[rng.choice(len(N), n, replace=False)]
    m1, m0 = P.mean(0), N.mean(0)
    C = (np.cov(P, rowvar=False) + np.cov(N, rowvar=False)) / 2 + np.eye(F.shape[2]) * 2.0
    w = np.linalg.solve(C, m1 - m0)
    threshold = float(0.5 * (m1 + m0) @ w)
    score = F @ w - threshold
    train = np.concatenate([P @ w - threshold, N @ w - threshold])
    s = max(1e-6, float(np.percentile(np.abs(train), 75)))
    prob = 1 / (1 + np.exp(-np.clip(score / s * 2.0, -12, 12)))
    prob[board == 0] = 0
    return prob.astype(np.float32)


def safe_copper_evidence(img: np.ndarray, gerber: np.ndarray, board: np.ndarray):
    """Best-effort legacy copper appearance map.

    The operator UI no longer depends on this automatic classifier. Some valid
    Gerbers (especially dense CAM350 exports) do not leave enough clean
    positive/negative pixels for calibration. In that case keep the visual
    comparator usable instead of aborting the whole analysis.
    """
    try:
        return copper_evidence(img, gerber, board), None
    except Exception as e:
        return np.zeros(board.shape, np.float32), str(e)



def _local_px_per_mm_internal(H: np.ndarray, q: np.ndarray) -> float:
    q = np.asarray(q, np.float64)
    p = transform_points(H, q[None, :])[0]
    px = transform_points(H, (q + np.array([1.0, 0.0]))[None, :])[0]
    py = transform_points(H, (q + np.array([0.0, 1.0]))[None, :])[0]
    return float((np.linalg.norm(px - p) + np.linalg.norm(py - p)) * 0.5)


def _bright_hole_center(img: np.ndarray, rim_center: np.ndarray, radius_px: float) -> tuple[np.ndarray | None, float]:
    """Estimate centroid of the visible opening *inside* an already located surface rim.

    Important: the search is clipped to the physical front-hole aperture. This prevents a
    mounting hole near the PCB outline from merging with the white scanner background and
    dragging the centroid outside the hole.
    """
    h, w = img.shape[:2]
    cx, cy = map(float, rim_center)
    rr = int(max(8, round(radius_px * 1.35 + 5)))
    x0, x1 = max(0, int(cx) - rr), min(w, int(cx) + rr + 1)
    y0, y1 = max(0, int(cy) - rr), min(h, int(cy) + rr + 1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None, 0.0
    patch = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    L, S = lab[:, :, 0], hsv[:, :, 1]
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    lx, ly = cx - x0, cy - y0
    d2 = (xx - lx) ** 2 + (yy - ly) ** 2

    # The visible far-side opening must remain inside the front-side aperture. 0.90R keeps
    # the external white scanner bed from leaking into the segmentation at board edges.
    local = d2 <= (max(3.0, radius_px * 0.90) ** 2)
    vals = L[local]
    if vals.size < 20:
        return None, 0.0
    lthr = max(140, int(np.percentile(vals, 62)))
    sth = min(135, max(55, int(np.percentile(S[local], 58) + 28)))
    mask = ((L >= lthr) & (S <= sth) & local).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, 0.0
    expected_area = math.pi * max(2.0, radius_px * 0.60) ** 2
    best = None
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < max(6.0, expected_area * 0.08) or area > expected_area * 2.2:
            continue
        c = np.asarray(cents[i], np.float64)
        dc = np.linalg.norm(c - np.array([lx, ly]))
        if dc > max(4.0, radius_px * 0.72):
            continue
        compact = area / max(1.0, float(stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]))
        # Prefer a substantial compact component, but allow true parallax displacement.
        score = 0.9 * min(1.2, area / max(expected_area, 1.0)) + 0.35 * compact - 0.035 * dc
        if best is None or score > best[0]:
            best = (score, c, area, compact, dc)
    if best is None:
        return None, 0.0
    c = best[1] + np.array([x0, y0], np.float64)
    conf = float(np.clip(0.35 + 0.28 * best[3] + 0.22 * min(1.0, best[2] / max(expected_area, 1.0))
                         - 0.08 * best[4] / max(radius_px, 1.0), 0, 1))
    return c, conf

def _robust_circle_fit(points: np.ndarray, weights: np.ndarray | None = None,
                       max_iters: int = 6) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Robust algebraic circle fit used for shadow-resistant drill-rim detection."""
    P = np.asarray(points, np.float64)
    if len(P) < 6:
        return None
    base_w = np.ones(len(P), np.float64) if weights is None else np.asarray(weights, np.float64)
    w = np.clip(base_w, 1e-3, None).copy()
    center = np.zeros(2, np.float64)
    radius = 0.0
    resid = np.full(len(P), np.inf, np.float64)
    for _ in range(max_iters):
        x, y = P[:, 0], P[:, 1]
        A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(P))])
        b = x * x + y * y
        sw = np.sqrt(np.clip(w, 1e-6, None))
        sol, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        cx, cy, c0 = sol
        radius = math.sqrt(max(1e-9, float(c0 + cx * cx + cy * cy)))
        center = np.array([cx, cy], np.float64)
        resid = np.abs(np.linalg.norm(P - center[None, :], axis=1) - radius)
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) + 1e-6
        scale = max(0.35, 1.4826 * mad)
        robust_w = 1.0 / (1.0 + (resid / (2.5 * scale)) ** 2)
        w = base_w * robust_w
    return center, float(radius), resid


def _surface_rim_center(img: np.ndarray, center: np.ndarray, radius_px: float,
                        board: np.ndarray | None = None,
                        return_meta: bool = False, prepared: dict[str, np.ndarray] | None = None):
    """Locate the front-surface drill rim while deliberately ignoring the hole shadow.

    The important physical cue is *not* brightness of the hole interior.  Moving radially
    outward across the true front rim normally produces a strong rise in green saturation:
    neutral/grey hole -> green solder mask.  A dark wall or shadow inside the 2 mm PCB is
    mostly a luminance edge and therefore receives little weight here.

    We sample many radial profiles around the expected Excellon radius, keep only sectors
    where the outside really looks like PCB, and robustly fit a circle to those edge points.
    Sectors contaminated by shadow, silkscreen or the white scanner bed are rejected.
    """
    h, w = img.shape[:2]
    seed = np.asarray(center, np.float64)
    R = max(3.0, float(radius_px))

    if prepared is not None and "L" in prepared and "S" in prepared:
        L = prepared["L"]
        S = prepared["S"]
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        L = cv2.GaussianBlur(lab[:, :, 0], (3, 3), 0).astype(np.float32)
        S = cv2.GaussianBlur(hsv[:, :, 1], (3, 3), 0).astype(np.float32)

    n_ang = 96
    angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    # Dense enough for small holes, but keep the search narrowly tied to Excellon radius.
    n_rad = max(30, int(round(4.0 * R)))
    # Keep the search close to the Excellon radius.  White silkscreen circles around
    # mounting holes live farther outside and used to be mistaken for the drill rim.
    rvals = np.linspace(max(1.5, 0.62 * R), 1.28 * R, n_rad)
    dr = float(rvals[1] - rvals[0]) if len(rvals) > 1 else 1.0
    halfwin = max(1, int(round(max(1.0, 0.10 * R) / max(dr, 1e-6))))

    edge_pts: list[np.ndarray] = []
    edge_weights: list[float] = []
    edge_scores: list[float] = []

    def sample(arr, xs, ys):
        mx = np.asarray(xs, np.float32).reshape(-1, 1)
        my = np.asarray(ys, np.float32).reshape(-1, 1)
        return cv2.remap(arr, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101).reshape(-1)

    for th in angles:
        ct, st = math.cos(float(th)), math.sin(float(th))
        xs = seed[0] + rvals * ct
        ys = seed[1] + rvals * st
        if xs.min() < 1 or ys.min() < 1 or xs.max() >= w - 1 or ys.max() >= h - 1:
            continue
        ss = sample(S, xs, ys)
        ll = sample(L, xs, ys)
        scores = np.full(len(rvals), -1e9, np.float64)
        aux = []
        for i in range(halfwin + 1, len(rvals) - halfwin - 1):
            sin_ = float(np.mean(ss[i-halfwin:i]))
            sout = float(np.mean(ss[i+1:i+1+halfwin]))
            lin = float(np.mean(ll[i-halfwin:i]))
            lout = float(np.mean(ll[i+1:i+1+halfwin]))
            sat_step = sout - sin_
            radius_pen = abs(float(rvals[i]) - R) / R
            # Main cue: neutral hole -> saturated green mask. Luminance only gently assists.
            score = 1.45 * sat_step + 0.14 * (lin - lout) - 58.0 * radius_pen
            # Candidates substantially outside the expected drill radius are allowed only
            # when the colour transition is exceptionally convincing.  This suppresses
            # circular white silkscreen legends without hard-coding their colour/diameter.
            if rvals[i] > 1.16 * R:
                score -= 24.0 * ((float(rvals[i]) / R - 1.16) / 0.12)
            if sout < 70.0:
                score -= 1.5 * (70.0 - sout)
            scores[i] = score
            aux.append((i, sat_step, sout))
        i = int(np.argmax(scores))
        best_score = float(scores[i])
        sin_ = float(np.mean(ss[max(0, i-halfwin):i]))
        sout = float(np.mean(ss[i+1:min(len(ss), i+1+halfwin)]))
        sat_step = sout - sin_
        if best_score < 12.0 or sat_step < 8.0 or sout < 55.0:
            continue
        rr = float(rvals[i])
        p = seed + np.array([rr * ct, rr * st], np.float64)

        # The pixel a little *outside* the candidate rim must still belong to the PCB.
        # This is what prevents a mounting hole near an outline from locking to scanner white.
        if board is not None:
            ro = rr + max(1.5, 0.15 * R)
            xo, yo = seed[0] + ro * ct, seed[1] + ro * st
            xi, yi = int(round(xo)), int(round(yo))
            if not (0 <= xi < w and 0 <= yi < h and board[yi, xi] > 0):
                continue

        edge_pts.append(p)
        edge_weights.append(float(np.clip(best_score, 1.0, 90.0)))
        edge_scores.append(best_score)

    meta = {
        "seed": seed.copy(), "expected_radius": R,
        "edge_points": np.asarray(edge_pts, np.float64) if edge_pts else np.empty((0, 2), np.float64),
        "accepted_points": np.empty((0, 2), np.float64), "coverage": 0.0,
        "fit_radius": None, "fit_residual_median": None,
    }
    if len(edge_pts) < 18:
        return (None, 0.0, meta) if return_meta else (None, 0.0)

    P = np.asarray(edge_pts, np.float64)
    W = np.asarray(edge_weights, np.float64)
    fit = _robust_circle_fit(P, W)
    if fit is None:
        return (None, 0.0, meta) if return_meta else (None, 0.0)
    cen, rfit, resid = fit

    # Explicitly drop one-sided shadow/silkscreen sectors with large radial residuals and refit.
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med))) + 1e-6
    lim = max(1.0, med + 3.0 * 1.4826 * mad, 0.12 * R)
    keep = resid <= lim
    if int(keep.sum()) >= 14 and int(keep.sum()) < len(P):
        fit2 = _robust_circle_fit(P[keep], W[keep])
        if fit2 is not None:
            cen, rfit, resid2 = fit2
            Pkeep = P[keep]
            final_resid = resid2
        else:
            Pkeep, final_resid = P, resid
    else:
        Pkeep, final_resid = P, resid

    coverage = len(Pkeep) / float(n_ang)
    radius_err = abs(float(rfit) - R) / R
    seed_shift = float(np.linalg.norm(cen - seed)) / R
    residual_med = float(np.median(final_resid)) if len(final_resid) else 99.0
    # A good front rim has broad angular support, the expected drill radius and a smooth circle.
    conf = (0.16 + 0.78 * coverage - 0.85 * radius_err - 0.22 * seed_shift
            - 0.10 * min(2.0, residual_med / max(0.5, 0.08 * R)))
    conf = float(np.clip(conf, 0.0, 1.0))

    meta.update({
        "accepted_points": Pkeep.copy(), "coverage": float(coverage),
        "fit_radius": float(rfit), "radius_error": float(radius_err),
        "seed_shift": float(seed_shift), "fit_residual_median": residual_med,
    })
    if return_meta:
        return cen.astype(np.float64), conf, meta
    return cen.astype(np.float64), conf

def analyze_hole_parallax(img: np.ndarray, H: np.ndarray, control_src: np.ndarray, control_dst: np.ndarray,
                          control_diam_mm: np.ndarray | None = None, pcb_thickness_mm: float = 2.0,
                          board: np.ndarray | None = None) -> dict[str, Any]:
    """Measure apparent through-hole parallax at registration anchors.

    v0.7 deliberately locates the *surface rim first* and only then looks for the bright
    far-side opening inside that rim. The older order could be pulled toward the white
    scanner bed for holes close to the PCB outline.

    These vectors are diagnostic only; they do not move the Gerber registration.
    """
    src = np.asarray(control_src, np.float64)
    dst = np.asarray(control_dst, np.float64)
    if control_diam_mm is None or len(control_diam_mm) != len(src):
        control_diam_mm = np.full(len(src), 3.2, np.float64)
    else:
        control_diam_mm = np.asarray(control_diam_mm, np.float64)
    items = []
    for i, (q, initial, diam) in enumerate(zip(src, dst, control_diam_mm)):
        pxmm = _local_px_per_mm_internal(H, q)
        radius = max(3.0, float(diam) * pxmm * 0.5)
        predicted = transform_points(H, np.asarray(q, np.float64)[None, :])[0]
        # The global model is usually a better seed than a locally biased bright-hole centroid.
        seed = 0.65 * predicted + 0.35 * np.asarray(initial, np.float64)
        rim, cr = _surface_rim_center(img, seed, radius, board=board)
        bright, cb = (None, 0.0) if rim is None else _bright_hole_center(img, rim, radius)
        if bright is None or rim is None:
            items.append({
                "index": i, "valid": False, "radius_px": radius, "confidence": 0.0,
                "bright_center": None, "rim_center": None, "vector": None,
                "px_per_mm": pxmm, "pcb_thickness_mm": float(pcb_thickness_mm),
            })
            continue
        v = bright - rim
        conf = float(min(cb, cr))
        mag = float(np.linalg.norm(v))
        shift_mm = mag / max(pxmm, 1e-9)
        angle_deg = float(np.degrees(np.arctan2(shift_mm, max(float(pcb_thickness_mm), 1e-6))))
        # A rim result that had to move an implausibly large fraction of the hole radius is suspect.
        rim_seed_shift = float(np.linalg.norm(rim - seed))
        if rim_seed_shift > max(3.0, radius * 0.55):
            conf *= 0.45
        items.append({
            "index": i, "valid": True, "radius_px": radius, "confidence": conf,
            "bright_center": bright.copy(), "rim_center": rim.copy(), "vector": v,
            "bright_center_auto": bright.copy(), "rim_center_auto": rim.copy(),
            "manual": False, "field_outlier": False,
            "magnitude": mag, "px_per_mm": pxmm, "pcb_thickness_mm": float(pcb_thickness_mm),
            "equivalent_angle_deg": angle_deg, "rim_seed_shift_px": rim_seed_shift,
        })

    return recalc_parallax_stats({"items": items, "stats": {}, "pcb_thickness_mm": float(pcb_thickness_mm)})

def recalc_parallax_stats(parallax: dict[str, Any]) -> dict[str, Any]:
    """Recompute parallax summary and robustly reject isolated automatic mis-detections."""
    items = parallax.get("items", [])
    prelim = []
    for it in items:
        it["field_outlier"] = False
        if not it.get("valid"):
            continue
        rim = it.get("rim_center")
        bright = it.get("bright_center")
        if rim is None or bright is None:
            continue
        rim = np.asarray(rim, np.float64)
        bright = np.asarray(bright, np.float64)
        v = bright - rim
        it["vector"] = v
        it["magnitude"] = float(np.linalg.norm(v))
        pxmm = float(it.get("px_per_mm", 1.0))
        thick = float(it.get("pcb_thickness_mm", parallax.get("pcb_thickness_mm", 2.0)))
        it["equivalent_angle_deg"] = float(np.degrees(np.arctan2(it["magnitude"] / max(pxmm, 1e-9), max(thick, 1e-6))))
        if it.get("manual", False) or it.get("confidence", 0) >= 0.30:
            prelim.append(it)

    # Fit a smooth first-order vector field and reject isolated auto points with large residuals.
    def fit_field(group):
        if len(group) < 4:
            return None
        rim = np.asarray([it["rim_center"] for it in group], np.float64)
        vec = np.asarray([it["vector"] for it in group], np.float64)
        xspan = max(1.0, float(np.ptp(rim[:, 0])))
        yspan = max(1.0, float(np.ptp(rim[:, 1])))
        xm, ym = float(rim[:, 0].mean()), float(rim[:, 1].mean())
        X = np.column_stack([np.ones(len(rim)), (rim[:, 0]-xm)/xspan, (rim[:, 1]-ym)/yspan])
        bx, *_ = np.linalg.lstsq(X, vec[:, 0], rcond=None)
        by, *_ = np.linalg.lstsq(X, vec[:, 1], rcond=None)
        pred = np.column_stack([X @ bx, X @ by])
        resid = np.linalg.norm(pred - vec, axis=1)
        return rim, vec, X, bx, by, pred, resid, xm, ym, xspan, yspan

    fit = fit_field(prelim)
    good = list(prelim)
    if fit is not None and len(prelim) >= 6:
        resid = fit[6]
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) + 1e-6
        limit = max(1.5, med + 3.5 * 1.4826 * mad)
        good = []
        for it, r in zip(prelim, resid):
            if (not it.get("manual", False)) and float(r) > limit:
                it["field_outlier"] = True
            else:
                good.append(it)
        fit = fit_field(good)

    stats: dict[str, Any] = {"count": len(good), "total": len(items), "auto_candidates": len(prelim)}
    if good:
        vec = np.asarray([it["vector"] for it in good], np.float64)
        mags = np.linalg.norm(vec, axis=1)
        angles = np.asarray([it.get("equivalent_angle_deg", np.nan) for it in good], np.float64)
        stats.update({
            "median_dx": float(np.median(vec[:, 0])),
            "median_dy": float(np.median(vec[:, 1])),
            "median_mag": float(np.median(mags)),
            "max_mag": float(np.max(mags)),
            "median_angle_deg": float(np.nanmedian(angles)),
        })
        if fit is not None:
            _, _, _, bx, by, pred, resid, xm, ym, xspan, yspan = fit
            stats["field_coeff_x"] = bx
            stats["field_coeff_y"] = by
            stats["field_origin"] = (xm, ym)
            stats["field_span"] = (xspan, yspan)
            stats["field_rms"] = float(np.sqrt(np.mean(resid ** 2)))
    parallax["stats"] = stats
    return parallax

def parallax_overlay(img: np.ndarray, parallax: dict[str, Any], arrow_scale: float = 3.0) -> np.ndarray:
    out = img.copy()
    for it in parallax.get("items", []):
        if not it.get("valid"):
            continue
        rim = np.asarray(it["rim_center"], np.float64)
        bright = np.asarray(it["bright_center"], np.float64)
        v = bright - rim
        r = int(max(5, round(it.get("radius_px", 6))))
        pr = tuple(np.rint(rim).astype(int))
        pb = tuple(np.rint(bright).astype(int))
        pe = tuple(np.rint(rim + v * arrow_scale).astype(int))
        manual = bool(it.get("manual", False))
        conf = float(it.get("confidence", 0.0))
        outlier = bool(it.get("field_outlier", False))
        rim_col = (0, 255, 0) if (conf >= 0.30 and not outlier) or manual else (0, 165, 255)
        # Surface rim: green/orange; through-hole opening: cyan; vector: magenta.
        cv2.circle(out, pr, r, rim_col, 1, cv2.LINE_AA)
        cv2.drawMarker(out, pr, rim_col, cv2.MARKER_CROSS, 11, 2, cv2.LINE_AA)
        cv2.circle(out, pb, 5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.arrowedLine(out, pr, pe, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
        mag = float(np.linalg.norm(v))
        angle = it.get("equivalent_angle_deg")
        flag = " M" if manual else (" OUT" if outlier else (" ?" if conf < 0.30 else ""))
        label = f"{mag:.1f}px" + (f"/{float(angle):.1f}deg" if angle is not None else "") + flag
        cv2.putText(out, label, (pr[0] + 6, pr[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(out, label, (pr[0] + 6, pr[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def refresh_parallax_result(result: dict[str, Any]) -> dict[str, Any]:
    par = analyze_hole_parallax(
        result["scan_full"], result["H"], result["control_src"], result["control_dst"],
        result.get("control_diam_mm"), result.get("pcb_thickness_mm", 2.0),
    )
    result["parallax"] = par
    st = par.get("stats", {})
    result["report"].update({
        "parallax_count": int(st.get("count", 0)),
        "parallax_total": int(st.get("total", 0)),
        "parallax_median_dx_px": round(float(st.get("median_dx", 0.0)), 3) if st.get("count", 0) else None,
        "parallax_median_dy_px": round(float(st.get("median_dy", 0.0)), 3) if st.get("count", 0) else None,
        "parallax_median_mag_px": round(float(st.get("median_mag", 0.0)), 3) if st.get("count", 0) else None,
        "parallax_max_mag_px": round(float(st.get("max_mag", 0.0)), 3) if st.get("count", 0) else None,
        "parallax_field_rms_px": round(float(st.get("field_rms", 0.0)), 3) if "field_rms" in st else None,
        "parallax_median_angle_deg": round(float(st.get("median_angle_deg", 0.0)), 2) if st.get("count", 0) else None,
    })
    return result


def update_parallax_report(result: dict[str, Any]) -> dict[str, Any]:
    par = recalc_parallax_stats(result.get("parallax", {"items": []}))
    st = par.get("stats", {})
    result["parallax"] = par
    result["report"].update({
        "parallax_count": int(st.get("count", 0)),
        "parallax_total": int(st.get("total", 0)),
        "parallax_median_dx_px": round(float(st.get("median_dx", 0.0)), 3) if st.get("count", 0) else None,
        "parallax_median_dy_px": round(float(st.get("median_dy", 0.0)), 3) if st.get("count", 0) else None,
        "parallax_median_mag_px": round(float(st.get("median_mag", 0.0)), 3) if st.get("count", 0) else None,
        "parallax_max_mag_px": round(float(st.get("max_mag", 0.0)), 3) if st.get("count", 0) else None,
        "parallax_field_rms_px": round(float(st.get("field_rms", 0.0)), 3) if "field_rms" in st else None,
        "parallax_median_angle_deg": round(float(st.get("median_angle_deg", 0.0)), 2) if st.get("count", 0) else None,
    })
    return result


def reset_manual_parallax(result: dict[str, Any]) -> dict[str, Any]:
    par = result.get("parallax", {})
    for it in par.get("items", []):
        if it.get("bright_center_auto") is not None:
            it["bright_center"] = np.asarray(it["bright_center_auto"], np.float64).copy()
        if it.get("rim_center_auto") is not None:
            it["rim_center"] = np.asarray(it["rim_center_auto"], np.float64).copy()
        it["manual"] = False
    return update_parallax_report(result)


def use_rim_centers_as_targets(result: dict[str, Any], registration_mode: str = "auto", min_confidence: float = 0.30) -> dict[str, Any]:
    """Replace matched bright-hole targets by estimated surface-rim centers, then refit.

    This is intentionally operator-triggered rather than automatic: it is a diagnostic
    correction for scanner/PCB-thickness parallax, not a reason to hide a real PCB offset.
    """
    par = result.get("parallax") or analyze_hole_parallax(
        result["scan_full"], result["H"], result["control_src"], result["control_dst"],
        result.get("control_diam_mm"), result.get("pcb_thickness_mm", 2.0),
    )
    dst = result["control_dst"].copy()
    used = 0
    for it in par.get("items", []):
        if it.get("valid") and it.get("confidence", 0) >= min_confidence:
            dst[int(it["index"])] = np.asarray(it["rim_center"], np.float64)
            used += 1
    if used < 3:
        raise RuntimeError(f"Not enough reliable surface-rim centers ({used})")
    result["control_dst"] = dst
    refit_result(result, registration_mode=registration_mode, recompute_copper=True)
    return result


def rim_debug_overlay(img: np.ndarray, board: np.ndarray, H: np.ndarray,
                      control_src: np.ndarray, control_dst: np.ndarray,
                      control_diam_mm: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Visualize how the shadow-resistant rim detector sees each coarse anchor.

    Yellow circle/cross = Excellon position predicted by the coarse global transform.
    Small cyan dots = radial edge samples accepted before circle fitting.
    Green circle/cross = fitted front-surface rim.
    Magenta line = correction vector from coarse prediction to fitted rim.

    This is intentionally diagnostic only and does not change registration.
    """
    out = img.copy()
    src = np.asarray(control_src, np.float64)
    dst = np.asarray(control_dst, np.float64)
    diam = np.asarray(control_diam_mm, np.float64)
    items: list[dict[str, Any]] = []
    if len(src) != len(dst) or len(src) != len(diam):
        return out, items
    for i, (q, old, dmm) in enumerate(zip(src, dst, diam)):
        pred = transform_points(H, q[None, :])[0]
        pxmm = _local_px_per_mm_internal(H, q)
        R = max(3.0, float(dmm) * pxmm * 0.5)
        seed = 0.82 * pred + 0.18 * np.asarray(old, np.float64)
        rim, conf, meta = _surface_rim_center(img, seed, R, board=board, return_meta=True)

        pp = tuple(np.rint(pred).astype(int))
        cv2.circle(out, pp, max(3, int(round(R))), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(out, pp, (0, 255, 255), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)

        ep = np.asarray(meta.get("accepted_points", np.empty((0, 2))), np.float64)
        if len(ep):
            step = max(1, len(ep) // 40)
            for p in ep[::step]:
                cv2.circle(out, tuple(np.rint(p).astype(int)), 1, (255, 255, 0), -1, cv2.LINE_AA)

        item = {
            "index": i, "predicted": pred.copy(), "seed": seed.copy(),
            "radius_px": float(R), "confidence": float(conf), "rim": None,
            "coverage": float(meta.get("coverage", 0.0)),
            "radius_error": meta.get("radius_error"),
        }
        if rim is not None:
            rfit = float(meta.get("fit_radius") or R)
            pr = tuple(np.rint(rim).astype(int))
            col = (0, 255, 0) if conf >= 0.32 else (0, 165, 255)
            cv2.circle(out, pr, max(3, int(round(rfit))), col, 2, cv2.LINE_AA)
            cv2.drawMarker(out, pr, col, cv2.MARKER_CROSS, 11, 2, cv2.LINE_AA)
            cv2.arrowedLine(out, pp, pr, (255, 0, 255), 1, cv2.LINE_AA, tipLength=0.28)
            shift = float(np.linalg.norm(rim - pred))
            cv2.putText(out, f"{i}:{shift:.1f}px c={conf:.2f}", (pr[0] + 7, pr[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, f"{i}:{shift:.1f}px c={conf:.2f}", (pr[0] + 7, pr[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            item["rim"] = rim.copy()
            item["shift_px"] = shift
        items.append(item)
    return out, items

def gerber_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    out = img.astype(np.float32).copy()
    a = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    red = np.zeros_like(out)
    red[:, :, 2] = 255
    out = out * (1.0 - a) + red * a
    return np.clip(out, 0, 255).astype(np.uint8)


def anchor_overlay(img: np.ndarray, scan_pts: np.ndarray, drill_xy: np.ndarray, H: np.ndarray, mirror: bool,
                   control_src: np.ndarray | None = None, control_dst: np.ndarray | None = None) -> np.ndarray:
    out = img.copy()
    # All detected scan holes: cyan rings.
    for p in scan_pts:
        cv2.circle(out, tuple(np.rint(p).astype(int)), 7, (255, 255, 0), 1, cv2.LINE_AA)
    # All drill holes transformed: yellow small circles.
    for q in drill_xy:
        # drill_xy is internal (x,-y), so convert back to raw xy for map helper.
        raw = (float(q[0]), float(-q[1]))
        p = map_xy(H, mirror, raw)
        cv2.circle(out, tuple(np.rint(p).astype(int)), 4, (0, 255, 255), 1, cv2.LINE_AA)

    if control_src is not None and control_dst is not None and len(control_src) == len(control_dst):
        pred = transform_points(H, control_src)
        for a, b in zip(pred, control_dst):
            pa = tuple(np.rint(a).astype(int))
            pb = tuple(np.rint(b).astype(int))
            cv2.line(out, pa, pb, (0, 165, 255), 1, cv2.LINE_AA)  # orange residual vector
            cv2.drawMarker(out, pa, (0, 255, 255), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)  # yellow predicted
            cv2.circle(out, pb, 5, (0, 255, 0), 2, cv2.LINE_AA)  # green editable target
    return out


def estimate_registration_px_per_mm(H: np.ndarray, control_src: np.ndarray | None = None) -> float:
    """Estimate working scan scale from the current registration transform.

    ``control_src`` uses the same internal (x, -y) coordinates that are fed to the
    registration fit.  Using several anchors makes the result stable even when a
    projective transform is active.
    """
    if control_src is None or len(control_src) == 0:
        qs = np.asarray([[0.0, 0.0], [50.0, -50.0], [100.0, -100.0]], np.float64)
    else:
        qs = np.asarray(control_src, np.float64)
    vals = []
    for q in qs:
        try:
            v = _local_px_per_mm_internal(H, q)
            if np.isfinite(v) and v > 0.05:
                vals.append(float(v))
        except Exception:
            pass
    return float(np.median(vals)) if vals else 1.0


def _ellipse_kernel_radius(radius_px: float) -> np.ndarray:
    r = max(0, int(round(float(radius_px))))
    if r <= 0:
        return np.ones((1, 1), np.uint8)
    n = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def _distance_allowance(mask: np.ndarray, board: np.ndarray, radius_px: float) -> np.ndarray:
    """Sub-pixel-ish distance based dilation used for optical allowances."""
    m = (mask > 0).astype(np.uint8)
    if radius_px <= 0:
        out = m > 0
    else:
        dist = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5)
        out = (m > 0) | (dist <= float(radius_px))
    return out & (board > 0)


def build_shadow_allowance(gerber: np.ndarray, board: np.ndarray, px_per_mm: float,
                           shadow_mm: float = 0.35, opposite_gerber: np.ndarray | None = None,
                           opposite_mm: float = 0.80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return allowed optical regions around the inspected and opposite copper layers.

    On translucent PCB substrates, apparent trace shadows may originate from
    the opposite copper layer visible through the board.  Because both Gerbers
    are known, that ghost is predictable and should not be reported as extra copper.

    Returns ``(allowed, local_halo, opposite_halo)``.
    """
    copper = (gerber > 127).astype(np.uint8)
    local_allowed = _distance_allowance(copper, board, float(shadow_mm) * max(0.05, float(px_per_mm)))
    local_halo = local_allowed & (copper == 0)

    opposite_halo = np.zeros_like(local_halo, dtype=bool)
    allowed = local_allowed.copy()
    if opposite_gerber is not None:
        opp = (opposite_gerber > 127).astype(np.uint8)
        opp_allowed = _distance_allowance(opp, board, float(opposite_mm) * max(0.05, float(px_per_mm)))
        # The nominal inspected copper is already red/known; only mark the extra allowance
        # from the opposite layer as a ghost zone.
        opposite_halo = opp_allowed & (copper == 0)
        allowed |= opp_allowed
    return allowed, local_halo, opposite_halo

def shadow_model_overlay(img: np.ndarray, gerber: np.ndarray, board: np.ndarray,
                         px_per_mm: float, shadow_mm: float = 0.35, alpha: float = 0.38,
                         opposite_gerber: np.ndarray | None = None, opposite_mm: float = 0.80) -> np.ndarray:
    """Visualise optical allowances used by EXTRA detection.

    Yellow = narrow halo around inspected-layer copper.
    Blue = expected visibility/ghost of the opposite copper layer (when supplied).
    Red = nominal inspected Gerber edge.
    """
    out = img.copy()
    _, halo, opposite_halo = build_shadow_allowance(
        gerber, board, px_per_mm, shadow_mm, opposite_gerber=opposite_gerber, opposite_mm=opposite_mm
    )
    layer = out.copy()
    if np.any(halo):
        layer[halo] = (0, 215, 255)  # yellow/orange
    if np.any(opposite_halo):
        layer[opposite_halo] = (255, 120, 0)  # blue-ish in BGR
    out = cv2.addWeighted(layer, float(alpha), out, 1.0 - float(alpha), 0)
    cont, _ = cv2.findContours((gerber > 127).astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cont, -1, (0, 0, 255), 1, cv2.LINE_AA)
    label = f"local {shadow_mm:.2f} mm"
    if opposite_gerber is not None:
        label += f"   opposite {opposite_mm:.2f} mm"
    cv2.putText(out, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# -----------------------------------------------------------------------------
# Optical golden-reference model (v0.11)
# -----------------------------------------------------------------------------

def _reference_feature(img: np.ndarray, valid_mask: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Return locally-normalised optical features for golden-reference comparison.

    The important point is that broad dark/bright background variation is treated as the
    local baseline.  Copper-like structures are represented mainly by *local* luminance
    and chroma departures, so a uniformly dark empty FR-4/solder-mask patch does not
    automatically look like extra copper.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    sigma = max(6.0, 4.0 * max(0.05, float(px_per_mm)))
    f = np.stack([
        cv2.GaussianBlur(L, (0, 0), sigma) - L,
        a - cv2.GaussianBlur(a, (0, 0), sigma),
        b - cv2.GaussianBlur(b, (0, 0), sigma),
    ], axis=2).astype(np.float32)
    m = valid_mask > 0
    for c in range(f.shape[2]):
        vals = f[:, :, c][m]
        if vals.size < 100:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-3
        f[:, :, c] = (f[:, :, c] - med) / max(0.25, 1.4826 * mad)
    return f


def snapshot_reference_sample(result: dict[str, Any]) -> dict[str, Any]:
    """Make a compact in-memory snapshot of an analysed known-good board."""
    return {
        "scan_full": result["scan_full"].copy(),
        "board_full": result["board_full"].copy(),
        "H": np.asarray(result["H"], np.float64).copy(),
        "px_per_mm": float(result.get("px_per_mm", result.get("report", {}).get("px_per_mm", 1.0))),
        "gerber_path": str(result.get("gerber_path", "")),
    }


def build_optical_reference_model(samples: list[dict[str, Any]], out_path: str | Path, progress_cb=None) -> dict[str, Any]:
    """Build a robust per-pixel optical baseline from 2+ known-good scans.

    All samples are first mapped into the coordinate frame of the first sample using the
    already-verified drill/Gerber registration.  The model stores the median local optical
    appearance and a robust per-pixel scale.
    """
    if len(samples) < 2:
        raise RuntimeError("Нужно минимум два исправных скана для эталонной модели")

    def progress(v, t):
        if progress_cb is not None:
            try:
                progress_cb(int(v), str(t))
            except Exception:
                pass

    progress(5, "Подготовка эталонных сканов…")
    s0 = samples[0]
    H0 = np.asarray(s0["H"], np.float64)
    h, w = s0["scan_full"].shape[:2]
    ppm = float(s0.get("px_per_mm", 1.0))
    aligned = []
    masks = []
    for i, smp in enumerate(samples):
        progress(10 + int(40 * (i + 1) / len(samples)), f"Совмещение эталона {i+1}/{len(samples)}…")
        H = np.asarray(smp["H"], np.float64)
        T = H0 @ np.linalg.inv(H)
        im = cv2.warpPerspective(smp["scan_full"], T, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        bm = cv2.warpPerspective((smp["board_full"] > 0).astype(np.uint8) * 255, T, (w, h),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        aligned.append(im)
        masks.append(bm > 0)

    common = np.logical_and.reduce(masks)
    if int(common.sum()) < 10000:
        raise RuntimeError("Слишком мала общая область эталонных сканов")

    feats = []
    for i, im in enumerate(aligned):
        progress(52 + int(24 * (i + 1) / len(aligned)), f"Оптическая модель {i+1}/{len(aligned)}…")
        feats.append(_reference_feature(im, common.astype(np.uint8) * 255, ppm))
    stack = np.stack(feats, axis=0)
    mu = np.median(stack, axis=0).astype(np.float32)
    dev = np.abs(stack - mu[None, ...])
    scale = (1.4826 * np.median(dev, axis=0) + 0.80).astype(np.float32)
    # Avoid unrealistically tiny variance where all reference boards happen to match perfectly.
    scale = np.maximum(scale, 0.80)

    progress(84, "Сохранение эталонной модели…")
    out_path = Path(out_path)
    np.savez_compressed(
        out_path,
        model_version=np.array([11], np.int32),
        sample_count=np.array([len(samples)], np.int32),
        shape=np.array([h, w], np.int32),
        H_ref=H0.astype(np.float64),
        px_per_mm=np.array([ppm], np.float32),
        common_mask=(common.astype(np.uint8) * 255),
        mean=mu.astype(np.float16),
        scale=scale.astype(np.float16),
        gerber_name=np.array([Path(str(s0.get("gerber_path", ""))).name]),
    )
    progress(100, "Эталонная модель готова")
    return {"path": str(out_path), "sample_count": len(samples), "shape": (h, w), "px_per_mm": ppm}


def load_optical_reference_model(path: str | Path) -> dict[str, Any]:
    z = np.load(str(path), allow_pickle=False)
    return {
        "path": str(path),
        "sample_count": int(np.asarray(z["sample_count"]).ravel()[0]),
        "shape": tuple(int(x) for x in np.asarray(z["shape"]).ravel()),
        "H_ref": np.asarray(z["H_ref"], np.float64),
        "px_per_mm": float(np.asarray(z["px_per_mm"]).ravel()[0]),
        "common_mask": np.asarray(z["common_mask"], np.uint8),
        "mean": np.asarray(z["mean"], np.float32),
        "scale": np.maximum(np.asarray(z["scale"], np.float32), 0.40),
        "gerber_name": str(np.asarray(z.get("gerber_name", np.array([""]))).ravel()[0]),
    }


def apply_optical_reference(result: dict[str, Any], model_or_path: dict[str, Any] | str | Path, progress_cb=None) -> np.ndarray:
    """Compute a golden-reference anomaly score and map it back to the current scan.

    A score around 1 is normal reference-to-reference variation.  Values above ~2.5-3
    are increasingly unusual.  EXTRA detection can require both copper-like appearance
    and a high reference anomaly, which suppresses stable dark empty regions.
    """
    model = model_or_path if isinstance(model_or_path, dict) else load_optical_reference_model(model_or_path)

    def progress(v, t):
        if progress_cb is not None:
            try:
                progress_cb(int(v), str(t))
            except Exception:
                pass

    progress(90, "Сравнение с оптическим эталоном…")
    h, w = model["shape"]
    H_ref = np.asarray(model["H_ref"], np.float64)
    H_cur = np.asarray(result["H"], np.float64)
    T = H_ref @ np.linalg.inv(H_cur)
    aligned = cv2.warpPerspective(result["scan_full"], T, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    bm = cv2.warpPerspective((result["board_full"] > 0).astype(np.uint8) * 255, T, (w, h),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (bm > 0) & (model["common_mask"] > 0)
    feat = _reference_feature(aligned, valid.astype(np.uint8) * 255, float(model["px_per_mm"]))
    d = (feat - model["mean"]) / model["scale"]
    score_ref = np.sqrt(np.mean(d * d, axis=2)).astype(np.float32)
    score_ref[~valid] = 0.0

    h2, w2 = result["scan_full"].shape[:2]
    T_back = H_cur @ np.linalg.inv(H_ref)
    score = cv2.warpPerspective(score_ref, T_back, (w2, h2), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_back = cv2.warpPerspective(valid.astype(np.uint8) * 255, T_back, (w2, h2),
                                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    score[~valid_back] = 0.0
    result["reference_score_full"] = score
    result["reference_model_path"] = model.get("path", "")
    result["reference_sample_count"] = int(model.get("sample_count", 0))
    result["reference_threshold"] = 2.5
    progress(98, "Оптический эталон применён")
    return score


def reference_score_overlay(img: np.ndarray, score: np.ndarray, board: np.ndarray, threshold: float = 2.5) -> np.ndarray:
    norm = np.clip(score / 5.0, 0.0, 1.0)
    heat = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_TURBO)
    heat[board == 0] = 255
    m = ((score >= float(threshold)) & (board > 0)).astype(np.uint8) * 255
    cont, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heat, cont, -1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(heat, f"reference z >= {threshold:.1f}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(heat, f"reference z >= {threshold:.1f}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    return heat

def build_review(img: np.ndarray, gerber: np.ndarray, board: np.ndarray, prob: np.ndarray,
                 threshold: float = 0.66, min_area: int | None = None,
                 *, px_per_mm: float = 1.0, shadow_mm: float = 0.35,
                 min_area_mm2: float | None = None, mode: str = "EXTRA",
                 opposite_gerber: np.ndarray | None = None, opposite_mm: float = 0.80,
                 reference_score: np.ndarray | None = None, reference_threshold: float = 2.5):
    """Build an operator review layer with Gerber-aware shadow suppression.

    EXTRA is the mature path in this revision.  MISSING/BOTH are included as an
    intentionally conservative preview: only the eroded inner core of a Gerber feature
    is considered missing so normal edge/lighting variation does not dominate.
    """
    mode = str(mode).upper()
    if mode not in {"EXTRA", "MISSING", "BOTH"}:
        mode = "EXTRA"
    pred = (prob > float(threshold)) & (board > 0)
    copper = (gerber > 127)
    interior = cv2.erode((board > 0).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0

    allowed, halo, opposite_halo = build_shadow_allowance(
        gerber, board, px_per_mm, shadow_mm, opposite_gerber=opposite_gerber, opposite_mm=opposite_mm
    )
    extra = pred & (~allowed) & interior
    # v0.11: when a golden optical reference is available, an EXTRA candidate must also
    # be unusual compared with known-good boards. This removes stable dark empty areas
    # that are optically dark but are not copper.
    if reference_score is not None:
        extra &= (reference_score >= float(reference_threshold))

    # Missing copper is deliberately conservative: inspect only the feature core.
    edge_tol_px = max(1.0, 0.14 * max(0.05, float(px_per_mm)))
    core = cv2.erode(copper.astype(np.uint8), _ellipse_kernel_radius(edge_tol_px)) > 0
    missing = core & (~pred) & interior

    ex = (extra.astype(np.uint8) * 255)
    mi = (missing.astype(np.uint8) * 255)
    ex = cv2.morphologyEx(ex, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mi = cv2.morphologyEx(mi, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    if min_area_mm2 is not None:
        min_area_px = max(1, int(round(float(min_area_mm2) * float(px_per_mm) ** 2)))
    else:
        min_area_px = max(1, int(10 if min_area is None else min_area))

    review = img.copy()
    # Subtle shadow-halo tint so the operator can see why adjacent dark pixels are ignored.
    layer = review.copy()
    layer[halo] = (0, 180, 255)
    if np.any(opposite_halo):
        layer[opposite_halo] = (255, 110, 0)
    review = cv2.addWeighted(layer, 0.16, review, 0.84, 0)

    regions = []
    masks = []
    if mode in {"EXTRA", "BOTH"}:
        masks.append(("EXTRA", ex, (255, 0, 255)))  # magenta
    if mode in {"MISSING", "BOTH"}:
        masks.append(("MISSING", mi, (255, 255, 0)))  # cyan

    for kind, mask, color in masks:
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, w, h, area = map(int, stats[i])
            if area < min_area_px:
                continue
            area_mm2 = float(area) / max(1e-9, float(px_per_mm) ** 2)
            regions.append({
                "kind": kind, "x": x, "y": y, "w": w, "h": h,
                "area_px": area, "area_mm2": area_mm2,
            })
            cv2.rectangle(review, (x, y), (x + w - 1, y + h - 1), color, 2)

    cont, _ = cv2.findContours(copper.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(review, cont, -1, (0, 0, 255), 1, cv2.LINE_AA)
    combined = np.maximum(ex if mode in {"EXTRA", "BOTH"} else 0,
                          mi if mode in {"MISSING", "BOTH"} else 0)
    return review, regions, combined


def _refresh_report(result: dict[str, Any], fit: dict[str, Any]) -> None:
    result["report"].update({
        "registration_model": fit["model"],
        "anchor_rms_px": round(float(fit["rms"]), 3),
        "anchor_max_error_px": round(float(fit["maxerr"]), 3),
        "affine_rms_px": round(float(fit["affine_rms"]), 3),
        "projective_rms_px": None if not np.isfinite(fit["projective_rms"]) else round(float(fit["projective_rms"]), 3),
        "top_rms_px": round(float(fit["top_rms"]), 3),
        "bottom_rms_px": round(float(fit["bottom_rms"]), 3),
    })



def refine_registration_to_surface_rims(img: np.ndarray, board: np.ndarray, reg: dict[str, Any],
                                        control_diam_mm: np.ndarray, mode: str = "auto") -> dict[str, Any]:
    """Second-pass registration using front-surface drill rims instead of bright-hole centroids.

    The first pass intentionally uses simple bright-hole centroids because they are easy to find
    globally.  On a thick PCB those centroids are biased by parallax/shadow.  Once the coarse
    Gerber/Excellon transform is known we know where every matched hole should be, so we can
    re-locate its *surface rim* locally and refit the transform.  This removes the systematic
    parallax bias from the registration itself while preserving the interior bright-center as a
    separate diagnostic measurement.
    """
    src0 = np.asarray(reg["control_src"], np.float64)
    dst0 = np.asarray(reg["control_dst"], np.float64)
    diam0 = np.asarray(control_diam_mm, np.float64)
    if len(src0) != len(diam0):
        return reg

    H0 = reg["H"]
    keep_src, keep_dst, keep_diam, shifts, confs = [], [], [], [], []
    for q, old, diam in zip(src0, dst0, diam0):
        pred = transform_points(H0, q[None, :])[0]
        pxmm = _local_px_per_mm_internal(H0, q)
        radius = max(3.0, float(diam) * pxmm * 0.5)
        # Seed primarily from the geometric prediction; the old bright centroid is only a weak
        # hint because it is exactly the quantity that parallax can bias.
        seed = 0.82 * pred + 0.18 * old
        rim, conf = _surface_rim_center(img, seed, radius, board=board)
        if rim is None or conf < 0.32:
            continue
        shift = float(np.linalg.norm(rim - old))
        # Reject catastrophic contour catches, but allow several pixels of true parallax.
        if shift > max(8.0, radius * 0.85):
            continue
        keep_src.append(q)
        keep_dst.append(rim)
        keep_diam.append(diam)
        shifts.append(shift)
        confs.append(conf)

    if len(keep_src) < 4:
        reg = dict(reg)
        reg["rim_refined_count"] = 0
        reg["rim_refine_rejected"] = 0
        reg["rim_refine_median_shift_px"] = None
        return reg

    src_all = np.asarray(keep_src, np.float64)
    dst_all = np.asarray(keep_dst, np.float64)
    diam_all = np.asarray(keep_diam, np.float64)

    # Hole shadows/silkscreen can still spoil an individual circumference.  Fit the global
    # transform, reject holes whose *centers* disagree with the rest, and refit.  This keeps
    # a local bad rim from steering the whole Gerber layer.
    idx = np.arange(len(src_all))
    for _ in range(4):
        if len(idx) < 4:
            break
        fit_i = fit_registration_points(src_all[idx], dst_all[idx], mode=mode)
        pred_i = transform_points(fit_i["H"], src_all[idx])
        err_i = np.linalg.norm(pred_i - dst_all[idx], axis=1)
        med = float(np.median(err_i))
        mad = float(np.median(np.abs(err_i - med))) + 1e-6
        lim = max(1.10, med + 2.8 * 1.4826 * mad)
        good = err_i <= lim
        if bool(np.all(good)) or int(good.sum()) < 4:
            break
        idx = idx[good]

    src = src_all[idx]
    dst = dst_all[idx]
    diam = diam_all[idx]
    fit = fit_registration_points(src, dst, mode=mode)
    fit.update({
        "mirror": reg["mirror"],
        "count": len(src),
        "pairs": reg.get("pairs", []),
        "control_src": src.copy(),
        "control_dst": dst.copy(),
        "control_dst_auto": dst.copy(),
        "control_diam_mm": diam.copy(),
        "rim_refined_count": len(src),
        "rim_refine_rejected": int(len(src_all) - len(src)),
        "rim_refine_candidates": int(len(src_all)),
        "rim_refine_median_shift_px": float(np.median(shifts)) if shifts else None,
        "rim_refine_max_shift_px": float(np.max(shifts)) if shifts else None,
        "rim_refine_median_confidence": float(np.median(confs)) if confs else None,
    })
    return fit


def parse_excellon_holes(paths, reference_gerber_path: str | Path | None = None, return_meta: bool = False):
    """Read one or more Excellon/ASCII drill files with PTH/NPTH classification.

    Altium can put plated and non-plated tool groups in a single ASCII TXT file
    using ``;TYPE=PLATED`` / ``;TYPE=NON_PLATED``.  Classification is therefore
    done per tool when that metadata is available, not once per file.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = []
    parse_meta = []
    reference_bounds_mm = _gerber_reference_bounds_mm(reference_gerber_path)
    reference_span_mm = None
    if reference_bounds_mm is not None:
        xmin,ymin,xmax,ymax=reference_bounds_mm
        reference_span_mm=max(float(xmax-xmin),float(ymax-ymin))
    for path in paths or []:
        if not path:
            continue
        pp = Path(path)
        try:
            txt = pp.read_text(errors="ignore").upper()
        except Exception:
            txt = ""
        name = pp.name.upper()
        file_nonplated = ("FILEFUNCTION,NONPLATED" in txt or "NPTH" in name or
                          ("NON" in name and "PLAT" in name))
        file_plated = ("FILEFUNCTION,PLATED" in txt or "PLATED,PTH" in txt or
                       ("PLATED" in name and not file_nonplated) or
                       ("PTH" in name and not file_nonplated))
        parsed, meta = parse_excellon(pp, reference_span_mm=reference_span_mm,
                                     reference_bounds_mm=reference_bounds_mm, return_meta=True)
        parse_meta.append(meta)
        tool_plating = meta.get("tool_plating", {}) or {}
        for x, y, d, tool in parsed:
            tp = tool_plating.get(int(tool), None)
            if tp is None:
                if file_nonplated:
                    tp = False
                elif file_plated:
                    tp = True
            nonplated = (tp is False)
            category = "NPTH" if nonplated else "PTH_OPEN"
            out.append({
                "x": float(x), "y": float(y), "diameter": float(d), "tool": int(tool),
                "category": category, "plated_expected": bool(not nonplated),
                "mask_open": bool(not nonplated), "source_file": str(pp),
                "plating_known": bool(tp is not None or file_nonplated or file_plated),
            })
    return (out, parse_meta) if return_meta else out

def subtract_drill_bores_from_copper(copper_mask: np.ndarray, holes: list[dict[str, Any]],
                                      H: np.ndarray, mirror: bool, samples: int = 28) -> np.ndarray:
    """Return final-board copper with every NC-drill bore removed.

    Many CAM systems (notably Altium) emit copper pads as solid Gerber flashes and
    describe the physical bore only in the Excellon drill file.  A literal Gerber
    render therefore paints copper through the centre of PTH/NPTH pads even though
    the manufactured board is drilled there.  Older Gerbers may already contain a
    clear centre; subtracting the bore again is harmless.

    The subtraction is deliberately independent of plating/category: PTH, tented
    vias and NPTH all have a physical opening in the copper foil.  Solder-mask
    classification remains separate and is not modified here.
    """
    out = np.asarray(copper_mask, np.uint8).copy()
    if not holes:
        return out
    h, w = out.shape[:2]
    n = max(12, int(samples))
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    for hit in holes:
        try:
            x = float(hit.get("x")); y = float(hit.get("y"))
            d = float(hit.get("diameter", hit.get("diameter_mm", 0.0)))
        except Exception:
            continue
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(d) or d <= 0.0:
            continue
        r = 0.5 * d
        pts = np.asarray([map_xy(H, mirror, (x + r * float(c), y + r * float(ss)))
                          for c, ss in zip(ca, sa)], np.float64)
        if pts.size == 0 or not np.all(np.isfinite(pts)):
            continue
        # Skip polygons wholly outside the raster; clipping is handled by OpenCV otherwise.
        if float(np.max(pts[:, 0])) < 0 or float(np.max(pts[:, 1])) < 0 or \
           float(np.min(pts[:, 0])) >= w or float(np.min(pts[:, 1])) >= h:
            continue
        poly = np.rint(pts).astype(np.int32)
        cv2.fillPoly(out, [poly], 0, lineType=cv2.LINE_AA)
    return out


def classify_gerber_holes_with_mask(holes: list[dict[str, Any]], mask_render: np.ndarray | None,
                                      H: np.ndarray, mirror: bool) -> list[dict[str, Any]]:
    """Classify plated Excellon holes as open PTH or tented VIA using solder-mask Gerber.

    Positive solder-mask Gerbers describe *openings*.  After the mask layer is rendered
    into scan coordinates, an opening at the drill centre means the plated hole is open;
    no opening means the via is tented on the inspected side.  NPTH stays NPTH.
    """
    out=[]
    for h in holes:
        z=dict(h)
        if str(z.get("category","")) == "NPTH" or not bool(z.get("plated_expected", True)):
            z["category"]="NPTH"; z["mask_open"]=False
            out.append(z); continue
        if mask_render is None:
            z["category"]="PTH_OPEN"; z["mask_open"]=True; z["mask_known"]=False
            out.append(z); continue
        p=map_xy(H, mirror, (float(z["x"]), float(z["y"])))
        x,y=int(round(float(p[0]))), int(round(float(p[1])))
        if 0 <= y < mask_render.shape[0] and 0 <= x < mask_render.shape[1]:
            r=2
            y0=max(0,y-r); y1=min(mask_render.shape[0],y+r+1)
            x0=max(0,x-r); x1=min(mask_render.shape[1],x+r+1)
            is_open=bool(np.max(mask_render[y0:y1,x0:x1]) > 48)
            z["mask_open"]=is_open; z["mask_known"]=True
            z["category"]="PTH_OPEN" if is_open else "VIA_TENTED"
        else:
            z["category"]="PTH_OPEN"; z["mask_open"]=True; z["mask_known"]=False
        out.append(z)
    return out


def _gerber_primary_anchors(holes: list[dict[str, Any]]) -> tuple[list[tuple[float,float,float]], float | None]:
    cand = [(float(h["x"]), float(h["y"]), float(h.get("diameter",0.0)))
            for h in holes if float(h.get("diameter",0.0)) >= 1.5]
    if not cand:
        return [], None
    allxy = np.asarray([[x,y] for x,y,d in cand], np.float64)
    fullx = max(1e-6, float(np.ptp(allxy[:,0]))); fully = max(1e-6, float(np.ptp(allxy[:,1])))
    for dd in sorted({round(float(d),3) for x,y,d in cand}, reverse=True):
        grp=[(x,y,d) for x,y,d in cand if abs(float(d)-dd) < 0.03]
        if len(grp) < 3:
            continue
        gxy=np.asarray([[x,y] for x,y,d in grp], np.float64)
        # Three well-spread holes are enough for a circular/partially clipped board.
        if (float(np.ptp(gxy[:,0])) >= 0.35*fullx and float(np.ptp(gxy[:,1])) >= 0.35*fully):
            return grp, dd
    # Fallback to all reasonably large holes; auto registration will reject outliers.
    return cand, None


def evaluate_gerber_holes(img: np.ndarray, board: np.ndarray, H: np.ndarray, mirror: bool,
                           holes: list[dict[str, Any]], tolerance_mm: float = 0.35) -> dict[str, Any]:
    """Positional optical check for Excellon holes in Gerber mode."""
    items=[]; tol=max(0.05,float(tolerance_mm))
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    L=cv2.GaussianBlur(lab[:,:,0],(3,3),0).astype(np.float32)
    S=cv2.GaussianBlur(hsv[:,:,1],(3,3),0).astype(np.float32)
    gx=cv2.Sobel(L,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(L,cv2.CV_32F,0,1,ksize=3)
    prepared={"L":L,"S":S,"mag":cv2.magnitude(gx,gy)}
    hh,ww=board.shape[:2]
    for h in holes:
        d=float(h.get("diameter",0.0))
        if d<=0: continue
        xy=(float(h["x"]),float(h["y"])); pred=map_xy(H,mirror,xy)
        ppm=_local_px_per_mm(H,mirror,xy); rpx=max(2.0,0.5*d*ppm)
        xi,yi=int(round(float(pred[0]))),int(round(float(pred[1])))
        visible=(0<=xi<ww and 0<=yi<hh and board[yi,xi]>0)
        cat=str(h.get("category","PTH_OPEN"))
        # A bore narrower than about two output pixels cannot be optically
        # localised reliably.  Mark it explicitly unresolved rather than
        # performing expensive circle searches and reporting false offsets.
        if visible and d * ppm < 2.1:
            items.append({
                "x":xy[0],"y":xy[1],"diameter_mm":d,"tool":int(h.get("tool",0)),
                "expected":np.asarray(pred,np.float64),"actual":None,"confidence":0.0,
                "shift_px":None,"shift_mm":None,"status":"UNKNOWN","method":"subpixel",
                "px_per_mm":float(ppm),"category":cat,"mask_open":bool(h.get("mask_open",False)),
                "plated_expected":bool(h.get("plated_expected",cat!="NPTH")),
                "plating_visual":"NA","plating_score":None,"plating_features":{},
            })
            continue
        if not visible:
            items.append({
                "x":xy[0],"y":xy[1],"diameter_mm":d,"tool":int(h.get("tool",0)),
                "expected":np.asarray(pred,np.float64),"actual":None,"confidence":0.0,
                "shift_px":None,"shift_mm":None,"status":"OUTSIDE","method":"outside",
                "px_per_mm":float(ppm),"category":cat,"mask_open":bool(h.get("mask_open",False)),
                "plated_expected":bool(h.get("plated_expected",cat!="NPTH")),
                "plating_visual":"NA","plating_score":None,"plating_features":{},
            }); continue
        actual=None; conf=0.0; method="edge"
        if d>=1.45:
            actual,conf=_surface_rim_center(img,pred,rpx,board=board,prepared=prepared); method="rim"
        if actual is None or conf<0.30:
            a2,c2,_=_circle_edge_localizer(img,pred,rpx,ppm,prepared=prepared)
            if actual is None or c2>conf: actual,conf,method=a2,c2,"edge"
        shift_px=shift_mm=None
        if actual is not None:
            shift_px=float(np.linalg.norm(np.asarray(actual)-pred)); shift_mm=shift_px/max(ppm,1e-9)
        if actual is None or conf<0.38: status="UNKNOWN"
        elif shift_mm is not None and shift_mm<=tol: status="OK"
        elif shift_mm is not None and shift_mm<=1.6*tol: status="WARN"
        else: status="SHIFT"
        plating_score=None; plating_features={}; plating_visual="NA"
        if actual is not None and conf>=0.30:
            plating_score,plating_features=_hole_plating_score(img,np.asarray(actual,np.float64),rpx,prepared=prepared)
            if plating_score is not None:
                if cat=="PTH_OPEN":
                    plating_visual="LIKELY" if plating_score>=0.50 else ("SUSPECT" if plating_score<=0.20 else "UNKNOWN")
                elif cat=="NPTH":
                    plating_visual="SUSPECT" if plating_score>=0.78 else "UNKNOWN"
        items.append({
            "x":xy[0],"y":xy[1],"diameter_mm":d,"tool":int(h.get("tool",0)),
            "expected":np.asarray(pred,np.float64),"actual":None if actual is None else np.asarray(actual,np.float64),
            "confidence":float(conf),"shift_px":shift_px,"shift_mm":shift_mm,"status":status,"method":method,
            "px_per_mm":float(ppm),"category":cat,"mask_open":bool(h.get("mask_open",False)),
            "plated_expected":bool(h.get("plated_expected",cat!="NPTH")),
            "plating_visual":plating_visual,"plating_score":plating_score,"plating_features":plating_features,
        })
    counts={k:sum(1 for it in items if it["status"]==k) for k in ("OK","WARN","SHIFT","UNKNOWN","OUTSIDE")}
    valid=[it["shift_mm"] for it in items if it["shift_mm"] is not None and it["status"]!="UNKNOWN"]
    result={"items":items,"counts":counts,"tolerance_mm":tol,
            "median_shift_mm":None if not valid else float(np.median(valid)),
            "max_shift_mm":None if not valid else float(np.max(valid))}
    _update_hole_summary(result)
    return result



def _profile_orientation_scores(board: np.ndarray, profile_path: str | Path | None,
                                candidates: list[dict[str, Any]]) -> list[float | None]:
    """Compare transformed Gerber Profile geometry with the optical PCB outline.

    Mounting-hole patterns on round boards can have strong 90/180-degree aliases.  The
    physical board outline is independent evidence and is especially useful for boards that
    are circular but have a flat, notch or other asymmetric edge feature.  Scores are robust
    pixel distances from the rendered Profile stroke to the detected board boundary; lower is
    better.  A missing/unreadable Profile simply yields ``None`` scores and preserves the
    legacy hole-only behaviour.
    """
    if not profile_path or not candidates:
        return [None] * len(candidates)
    try:
        b = (np.asarray(board) > 0).astype(np.uint8) * 255
        edge = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        dist = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
        h, w = b.shape[:2]
        vals: list[float | None] = []
        for alt in candidates:
            try:
                pm = render_gerber_to_scan(profile_path, (h, w, 3), alt["H"], bool(alt.get("mirror", False)))
                yy, xx = np.where(pm > 32)
                if len(xx) < 30:
                    vals.append(None); continue
                dd = dist[yy, xx].astype(np.float64)
                # p75 makes a badly rotated profile expensive even if a short arc happens to
                # touch the round part of the board. Median keeps the score stable to tiny
                # scanner/profile offsets and the mean is only a weak tail term.
                score = float(np.percentile(dd, 75) + 0.50 * np.median(dd) + 0.01 * np.mean(dd))
                vals.append(score if np.isfinite(score) else None)
            except Exception:
                vals.append(None)
        return vals
    except Exception:
        return [None] * len(candidates)


def _profile_gate_candidates(board: np.ndarray, profile_path: str | Path | None,
                             candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float | None], bool]:
    """Reject clearly wrong rotational aliases using Gerber Profile when it is informative."""
    scores = _profile_orientation_scores(board, profile_path, candidates)
    finite = [(i, float(s)) for i, s in enumerate(scores) if s is not None and np.isfinite(s)]
    if len(finite) < 2:
        return candidates, scores, False
    finite.sort(key=lambda z: z[1])
    best = finite[0][1]
    med = float(np.median([s for _, s in finite]))
    # Only trust the outline when it actually separates candidates.  Circular/symmetric
    # profiles naturally score almost the same for several rotations and must fall back to
    # the complete drill map.
    informative = (med >= best + 18.0) or (best > 0 and med / best >= 1.55)
    if not informative:
        return candidates, scores, False
    ys, xs = np.where(np.asarray(board) > 0)
    diag = float(np.hypot(np.ptp(xs), np.ptp(ys))) if len(xs) else 1000.0
    margin = max(14.0, 0.035 * diag)
    keep_idx = [i for i, s in finite if s <= best + margin]
    if not keep_idx:
        return candidates, scores, False
    gated = [candidates[i] for i in keep_idx]
    return gated, scores, len(gated) < len(candidates)

def analyze_gerber(scan_path: str | Path, gerber_path: str | Path, drill_paths,
                   pdf_page: int = 0, pdf_dpi: int = 150, registration_mode: str = "auto",
                   hole_tolerance_mm: float = 0.35, progress_cb=None,
                   opposite_gerber_path: str | Path | None = None,
                   solder_mask_gerber_path: str | Path | None = None,
                   profile_gerber_path: str | Path | None = None) -> dict[str, Any]:
    """Operator-mode analysis using Gerber copper + one or more Excellon drill files."""
    def progress(value:int,text:str):
        if progress_cb is not None:
            try: progress_cb(int(value),str(text))
            except Exception: pass
    progress(4,"Чтение скана / PDF…")
    img,scan_meta=load_scan(scan_path,pdf_page=pdf_page,pdf_dpi=pdf_dpi,return_meta=True)
    progress(13,"Поиск платы…"); board=pcb_mask(img)
    progress(21,"Поиск монтажных отверстий…"); scan_pts,_=detect_large_holes(img,board)
    progress(28,"Чтение Excellon…"); holes,drill_parse_meta=parse_excellon_holes(drill_paths, reference_gerber_path=gerber_path, return_meta=True)
    anchors,primary_diam=_gerber_primary_anchors(holes)
    if len(anchors)<3:
        raise RuntimeError(f"Gerber: insufficient usable drill anchors ({len(anchors)}). "
                           "Check the NC Drill files and their coordinate format.")
    q=np.asarray([[x,-y] for x,y,d in anchors],np.float64); anchor_diam=np.asarray([d for x,y,d in anchors],np.float64)
    progress(40,"Грубая регистрация платы…")
    try:
        reg,alternatives=auto_point_registration(q,scan_pts,mode=registration_mode)
    except RuntimeError as e:
        fmts=[]
        for m in drill_parse_meta:
            f=m.get("format")
            fmt_txt=(f"{f[0]}.{f[1]}" if f else f"auto/*.{m.get('inferred_decimals')}")
            pad=m.get("fixed_padding")
            auto=" auto" if m.get("padding_autodetected") else ""
            fmts.append(f"{Path(m.get('path','')).name}: {m.get('units')} {m.get('zero_suppression')} {fmt_txt}, pad={pad}{auto}, holes={m.get('points')}")
        detail="; ".join(fmts) if fmts else "Excellon empty"
        raise RuntimeError(f"Anchor registration failed. Drill decode: {detail}") from e
    # Round/symmetric boards can have several rotations with the same mounting-hole fit.
    # Keep those alternatives and let the complete drill map choose the orientation.
    orientation_disambiguated=False; orientation_scores=[]
    profile_orientation_used=False; profile_candidate_scores=[]
    if len(alternatives)>1:
        best_count=int(reg.get("count",0)); cands=[a for a in alternatives[:12] if int(a.get("count",0))>=max(3,best_count-1)]
        # Ranking dozens of rotations against hundreds of tiny vias is expensive.
        # Pick a deterministic, spatially distributed evidence subset here; the
        # final hole inspection below still checks EVERY circular drill hit.
        if len(holes) > 260:
            usable=[h for h in holes if float(h.get("diameter",0)) >= 0.5]
            selection_holes=usable if len(usable)>=16 else holes
            xy = np.asarray([(h["x"],h["y"]) for h in selection_holes], np.float64)
            mins, spans = np.min(xy,axis=0), np.maximum(np.ptp(xy,axis=0), 1e-6)
            cells = np.floor((xy-mins)/spans*np.array([13.0,11.0])).astype(int)
            chosen = set(np.argsort([-float(h.get("diameter",0)) for h in selection_holes])[:32].tolist())
            for cell in sorted({(int(c[0]),int(c[1])) for c in cells}):
                ii = np.where(np.all(cells==np.array(cell),axis=1))[0]
                if len(ii): chosen.add(int(ii[len(ii)//2]))
            for ii in np.linspace(0,len(selection_holes)-1,80,dtype=int): chosen.add(int(ii))
            orientation_holes = [selection_holes[i] for i in sorted(chosen)]
        else:
            orientation_holes = holes
        # v0.15.12: the folder scanner already knew about Profile/outline Gerbers but the
        # registration code never used them.  Gate equally-good mounting-hole aliases by
        # the physical PCB outline before the full-drill optical score.  This specifically
        # prevents 90-degree locks on round boards with a flat/notched edge.
        cands_before=list(cands)
        gated, pscores, profile_orientation_used = _profile_gate_candidates(board, profile_gerber_path, cands)
        for a, ps in zip(cands_before, pscores):
            profile_candidate_scores.append({
                "mirror":bool(a.get("mirror",False)),
                "angle_deg":round(float(registration_angle_deg(a["H"],a.get("mirror",False))),1),
                "anchor_count":int(a.get("count",0)),
                "profile_score_px":None if ps is None else round(float(ps),3),
            })
        if profile_orientation_used and gated:
            cands=gated
        sigs=set((bool(a.get("mirror",False)),int(round(registration_angle_deg(a["H"],a.get("mirror",False))/5.0))) for a in cands)
        if len(sigs)>1:
            progress(47,"Проверка поворота по контуру и полной карте отверстий…" if profile_orientation_used else "Проверка поворота по полной карте отверстий…"); scored=[]
            touches_now=bool(np.any(board[0,:]) or np.any(board[-1,:]) or np.any(board[:,0]) or np.any(board[:,-1]))
            pscore_lookup={id(a):ps for a,ps in zip(cands_before,pscores)}
            for alt in cands:
                try:
                    chk=evaluate_gerber_holes(img,board,alt["H"],alt["mirror"],orientation_holes,max(0.45,float(hole_tolerance_mm)))
                    cc=chk.get("counts",{}); ok=float(cc.get("OK",0)); warn=float(cc.get("WARN",0)); shift=float(cc.get("SHIFT",0)); unk=float(cc.get("UNKNOWN",0)); out=float(cc.get("OUTSIDE",0))
                    vis=max(1.0,ok+warn+shift+unk); total=max(1.0,vis+out); quality=(ok+0.30*warn-1.80*shift-0.45*unk)/vis; coverage=vis/total
                    score=100.0*quality+(30.0 if not touches_now else 5.0)*coverage+0.35*float(alt.get("count",0))-0.05*float(alt.get("rms",0.0))
                    ang=registration_angle_deg(alt["H"],alt.get("mirror",False)); scored.append((score,alt,cc,ang)); ps=pscore_lookup.get(id(alt))
                    orientation_scores.append({"mirror":bool(alt.get("mirror",False)),"angle_deg":round(float(ang),1),"anchor_count":int(alt.get("count",0)),"score":round(score,3),"profile_score_px":None if ps is None else round(float(ps),3),"counts":dict(cc)})
                except Exception: pass
            if scored:
                scored.sort(key=lambda z:(-z[0],-int(z[1].get("count",0)),float(z[1].get("rms",1e9))))
                if scored[0][1] is not reg: reg=scored[0][1]; orientation_disambiguated=True
        elif len(cands)==1 and cands[0] is not reg:
            reg=cands[0]; orientation_disambiguated=True
    coarse_reg=reg; coarse_diam=np.asarray([anchor_diam[i] for i,j in reg["pairs"]],np.float64)
    progress(54,"Уточнение по кромкам отверстий…"); reg=refine_registration_to_surface_rims(img,board,reg,coarse_diam,mode=registration_mode)
    H,mirror=reg["H"],reg["mirror"]; px_per_mm=estimate_registration_px_per_mm(H,reg.get("control_src"))
    progress(66,"Рендер Gerber…")
    copper_raw=render_gerber_to_scan(gerber_path,img.shape,H,mirror)
    copper=subtract_drill_bores_from_copper(copper_raw,holes,H,mirror)
    opposite=None
    if opposite_gerber_path:
        try:
            progress(72,"Рендер второго слоя…")
            opposite_raw=render_gerber_to_scan(opposite_gerber_path,img.shape,H,mirror)
            opposite=subtract_drill_bores_from_copper(opposite_raw,holes,H,mirror)
        except Exception:
            opposite=None
    mask_render=None
    if solder_mask_gerber_path:
        try:
            progress(77,"Рендер solder-mask…")
            mask_render=render_gerber_to_scan(solder_mask_gerber_path,img.shape,H,mirror)
            holes=classify_gerber_holes_with_mask(holes,mask_render,H,mirror)
        except Exception:
            mask_render=None
    progress(82,"Проверка отверстий…"); hole_check=evaluate_gerber_holes(img,board,H,mirror,holes,hole_tolerance_mm)
    progress(91,"Подготовка отображения…")
    prob, copper_evidence_warning = safe_copper_evidence(img, copper, board)
    crop=board_bbox(board,20)
    touches={"top":bool(np.any(board[0,:]>0)),"bottom":bool(np.any(board[-1,:]>0)),"left":bool(np.any(board[:,0]>0)),"right":bool(np.any(board[:,-1]>0))}
    report={
        "source_type":"gerber","layer_name":Path(gerber_path).name,"primary_anchor_diameter_mm":None if primary_diam is None else round(float(primary_diam),3),
        "detected_scan_anchors":int(len(scan_pts)),"drill_anchors":int(len(q)),"matched_anchors":int(reg["count"]),
        "coarse_matched_anchors":int(coarse_reg["count"]),"orientation_disambiguated":bool(orientation_disambiguated),"orientation_scores":orientation_scores,
        "profile_orientation_used":bool(profile_orientation_used),"profile_orientation_scores":profile_candidate_scores,
        "profile_path":None if not profile_gerber_path else str(profile_gerber_path),
        "partial_scan":bool(any(touches.values())),"board_touches_frame":touches,"rim_refined_anchors":int(reg.get("rim_refined_count",0)),
        "rim_refine_rejected":int(reg.get("rim_refine_rejected",0)),"mirrored":bool(mirror),"orientation_angle_deg":round(registration_angle_deg(H, mirror),1),"orientation_candidates_checked":max(len(orientation_scores),len(profile_candidate_scores)),"crop":crop,
        "scan_source_mode":scan_meta.get("source_mode"),"scan_native_shape":scan_meta.get("native_shape"),"scan_analysis_shape":scan_meta.get("analysis_shape"),
        "scan_analysis_scale":round(float(scan_meta.get("analysis_scale",1.0)),4),"px_per_mm":round(float(px_per_mm),4),
        "drill_parse_meta":drill_parse_meta,
        "drill_bores_subtracted_from_copper":int(len(holes)),
        "copper_coverage_pct": round(float(np.count_nonzero((copper>127) & (board>0))) / max(1, int(np.count_nonzero(board>0))) * 100.0, 2),
        "copper_evidence_warning": copper_evidence_warning,
        "solder_mask_present": bool(mask_render is not None),
        "solder_mask_path": None if not solder_mask_gerber_path else str(solder_mask_gerber_path),
        "hole_check_counts":hole_check.get("counts",{}),"hole_category_counts":hole_check.get("category_counts",{}),
        "hole_median_shift_mm":None if hole_check.get("median_shift_mm") is None else round(float(hole_check["median_shift_mm"]),3),
        "hole_max_shift_mm":None if hole_check.get("max_shift_mm") is None else round(float(hole_check["max_shift_mm"]),3),
    }
    result={
        "source_type":"gerber","scan_full":img,"board_full":board,"gerber_full":copper,"gerber_display_full":copper,"gerber_overlay_full":copper,
        "opposite_gerber_full":opposite,"solder_mask_full":mask_render,"prob_full":prob,"scan_pts":scan_pts,"drill_xy":q,"H":H,"mirror":mirror,"crop":crop,"report":report,
        "alternatives":alternatives,"control_src":reg["control_src"].copy(),"control_dst":reg["control_dst"].copy(),"control_dst_auto":reg["control_dst_auto"].copy(),
        "control_diam_mm":np.asarray(reg.get("control_diam_mm",coarse_diam),np.float64),"registration_mode_requested":registration_mode,
        "gerber_path":str(gerber_path),"opposite_gerber_path":None if not opposite_gerber_path else str(opposite_gerber_path),
        "solder_mask_gerber_path":None if not solder_mask_gerber_path else str(solder_mask_gerber_path),
        "drill_paths":[str(x) for x in (drill_paths if isinstance(drill_paths,(list,tuple)) else [drill_paths]) if x],
        "gerber_holes":holes,"hole_check":hole_check,"hole_tolerance_mm":float(hole_tolerance_mm),
        "coarse_H":coarse_reg["H"].copy(),"coarse_control_src":coarse_reg["control_src"].copy(),"coarse_control_dst":coarse_reg["control_dst"].copy(),
        "coarse_control_diam_mm":coarse_diam.copy(),"coarse_registration_model":coarse_reg.get("model","affine"),"coarse_anchor_rms_px":float(coarse_reg.get("rms",0.0)),
        "px_per_mm":float(px_per_mm),
    }
    _refresh_report(result,reg); progress(100,"Готово"); return result


def analyze(scan_path: str | Path, gerber_path: str | Path, drill_path: str | Path,
            pdf_page: int = 0, pdf_dpi: int = 150, registration_mode: str = "auto",
            pcb_thickness_mm: float = 2.0, progress_cb=None,
            opposite_gerber_path: str | Path | None = None) -> dict[str, Any]:
    def progress(value: int, text: str):
        if progress_cb is not None:
            try:
                progress_cb(int(value), str(text))
            except Exception:
                pass

    progress(4, "Чтение скана / PDF…")
    img, scan_meta = load_scan(scan_path, pdf_page=pdf_page, pdf_dpi=pdf_dpi, return_meta=True)
    progress(13, "Поиск контура платы…")
    board = pcb_mask(img)
    progress(22, "Поиск отверстий…")
    scan_pts, hole_mask = detect_large_holes(img, board)
    progress(30, "Чтение Excellon и выбор механических якорей…")
    drills = parse_excellon(drill_path)
    # v0.8.4: prefer one large mechanical-hole diameter class for registration.
    # On this board the 3.5 mm mounting holes are clean, while the 2.9 mm internal
    # holes are surrounded by white silkscreen circles that can bias optical rim finding.
    # Smaller/lower-priority holes remain useful as validation points, but should not
    # steer the global Gerber transform unless there are too few clean mechanical holes.
    cand = [(x, y, d) for x, y, d, t in drills if d >= 1.5]
    anchors = []
    primary_diam = None
    if cand:
        ds = sorted({round(float(d), 3) for x, y, d in cand}, reverse=True)
        allxy = np.asarray([[x, y] for x, y, d in cand], np.float64)
        fullx = max(1e-6, float(np.ptp(allxy[:, 0])))
        fully = max(1e-6, float(np.ptp(allxy[:, 1])))
        for dd in ds:
            grp = [(x, y, d) for x, y, d in cand if abs(float(d) - dd) < 0.03]
            if len(grp) < 4:
                continue
            gxy = np.asarray([[x, y] for x, y, d in grp], np.float64)
            # Require spatial coverage, not just four holes in one connector footprint.
            if float(np.ptp(gxy[:, 0])) >= 0.45 * fullx and float(np.ptp(gxy[:, 1])) >= 0.45 * fully:
                anchors = grp
                primary_diam = dd
                break
    if len(anchors) < 4:
        anchors = [(x, y, d) for x, y, d, t in drills if d >= 2.5]
    if len(anchors) < 4:
        anchors = [(x, y, d) for x, y, d, t in drills if d >= 1.5]
    q = np.asarray([[x, -y] for x, y, d in anchors], np.float64)
    anchor_diam = np.asarray([d for x, y, d in anchors], np.float64)
    progress(42, "Грубая регистрация платы…")
    reg, alternatives = auto_point_registration(q, scan_pts, mode=registration_mode)
    coarse_reg = reg
    coarse_diam = np.asarray([anchor_diam[i] for i, j in reg["pairs"]], np.float64)
    # v0.7: the global bright-hole match is only a coarse pass.  Refit to the physical
    # front-surface rims so hole shadows / PCB thickness do not bias Gerber registration.
    progress(57, "Уточнение по кромкам монтажных отверстий…")
    refined = refine_registration_to_surface_rims(img, board, reg, coarse_diam, mode=registration_mode)
    reg = refined
    H, mirror = reg["H"], reg["mirror"]
    px_per_mm = estimate_registration_px_per_mm(H, reg.get("control_src"))
    progress(70, "Рендер Gerber в координаты скана…")
    gerber = render_gerber_to_scan(gerber_path, img.shape, H, mirror)
    opposite_gerber = None
    if opposite_gerber_path:
        try:
            progress(78, "Рендер противоположного слоя меди…")
            opposite_gerber = render_gerber_to_scan(opposite_gerber_path, img.shape, H, mirror)
        except Exception:
            opposite_gerber = None
    progress(88, "Подготовка отображения…")
    prob, copper_evidence_warning = safe_copper_evidence(img, gerber, board)
    crop = board_bbox(board, 20)

    report = {
        "detected_scan_anchors": int(len(scan_pts)),
        "drill_anchors": int(len(q)),
        "primary_anchor_diameter_mm": None if primary_diam is None else round(float(primary_diam), 3),
        "anchor_policy": "largest-spread mechanical diameter class" if primary_diam is not None else "fallback mixed diameters",
        "matched_anchors": int(reg["count"]),
        "coarse_matched_anchors": int(coarse_reg["count"]),
        "rim_refined_anchors": int(reg.get("rim_refined_count", 0)),
        "rim_refine_rejected": int(reg.get("rim_refine_rejected", 0)),
        "rim_refine_median_shift_px": None if reg.get("rim_refine_median_shift_px") is None else round(float(reg.get("rim_refine_median_shift_px")), 3),
        "rim_refine_max_shift_px": None if reg.get("rim_refine_max_shift_px") is None else round(float(reg.get("rim_refine_max_shift_px")), 3),
        "mirrored": bool(mirror),
        "crop": crop,
        "scan_source_mode": scan_meta.get("source_mode"),
        "scan_native_shape": scan_meta.get("native_shape"),
        "scan_analysis_shape": scan_meta.get("analysis_shape"),
        "scan_analysis_scale": round(float(scan_meta.get("analysis_scale", 1.0)), 4),
        "px_per_mm": round(float(px_per_mm), 4),
        "copper_evidence_warning": copper_evidence_warning,
    }

    result = {
        "scan_full": img,
        "board_full": board,
        "gerber_full": gerber,
        "prob_full": prob,
        "scan_pts": scan_pts,
        "drill_xy": q,
        "H": H,
        "mirror": mirror,
        "crop": crop,
        "report": report,
        "alternatives": alternatives,
        "control_src": reg["control_src"].copy(),
        "control_dst": reg["control_dst"].copy(),
        "control_dst_auto": reg["control_dst_auto"].copy(),
        "control_diam_mm": np.asarray(reg.get("control_diam_mm", coarse_diam), np.float64),
        "registration_mode_requested": registration_mode,
        "pcb_thickness_mm": float(pcb_thickness_mm),
        "gerber_path": str(gerber_path),
        "opposite_gerber_path": None if not opposite_gerber_path else str(opposite_gerber_path),
        "opposite_gerber_full": opposite_gerber,
        "coarse_H": coarse_reg["H"].copy(),
        "coarse_control_src": coarse_reg["control_src"].copy(),
        "coarse_control_dst": coarse_reg["control_dst"].copy(),
        "coarse_control_diam_mm": coarse_diam.copy(),
        "coarse_registration_model": coarse_reg.get("model", "affine"),
        "coarse_anchor_rms_px": float(coarse_reg.get("rms", 0.0)),
        "px_per_mm": float(px_per_mm),
    }
    _refresh_report(result, reg)
    progress(100, "Готово")
    return result




def analyze_camvu(scan_path: str | Path, cam_path: str | Path, copper_layer: int | str | None = None,
                  pdf_page: int = 0, pdf_dpi: int = 150, registration_mode: str = "auto",
                  hole_tolerance_mm: float = 0.35, progress_cb=None) -> dict[str, Any]:
    """AOI analysis using one CAMvu text database instead of separate Gerber/Excellon files."""
    def progress(value: int, text: str):
        if progress_cb is not None:
            try: progress_cb(int(value), str(text))
            except Exception: pass

    progress(3, "Чтение CAM…")
    cam = parse_camvu(cam_path)
    lid = camvu_layer_id(cam, copper_layer)
    other_lid = camvu_opposite_layer(cam, lid)
    progress(8, "Чтение скана / PDF…")
    img, scan_meta = load_scan(scan_path, pdf_page=pdf_page, pdf_dpi=pdf_dpi, return_meta=True)
    progress(15, "Поиск контура платы…")
    board = pcb_mask(img)
    progress(23, "Поиск монтажных отверстий…")
    scan_pts, hole_mask = detect_large_holes(img, board)
    progress(31, "Выбор механических якорей из CAM…")
    anchors, primary_diam = _cam_primary_anchors(cam)
    if len(anchors) < 4:
        raise RuntimeError("CAM: not enough suitable mechanical drill anchors")
    q = np.asarray([[x, -y] for x,y,d in anchors], np.float64)
    anchor_diam = np.asarray([d for x,y,d in anchors], np.float64)
    progress(42, "Грубая регистрация платы…")
    reg, alternatives = auto_point_registration(q, scan_pts, mode=registration_mode)
    orientation_disambiguated = False
    orientation_scores = []
    # Round boards often have a symmetric ring of mounting holes.  Several rotations can therefore
    # fit the primary anchors equally well even when six or more anchors are visible.  v0.15.1
    # always keeps rotational alternatives and, when they are genuinely distinct, lets the full
    # CAM drill map choose the orientation.
    if len(alternatives) > 1:
        best_count = int(reg.get("count",0))
        cands = [a for a in alternatives[:12] if int(a.get("count",0)) >= max(3, best_count-1)]
        # Only pay for the full-hole check if there is a different rotation or handedness to test.
        sigs = set((bool(a.get("mirror",False)), int(round(registration_angle_deg(a["H"], a.get("mirror",False))/5.0))) for a in cands)
        if len(sigs) > 1:
            progress(48, "Проверка поворота по полной карте отверстий…")
            scored = []
            hh, ww = board.shape[:2]
            touches = bool(np.any(board[0,:]) or np.any(board[-1,:]) or np.any(board[:,0]) or np.any(board[:,-1]))
            for alt in cands:
                try:
                    chk = evaluate_cam_holes(img, board, alt["H"], alt["mirror"], cam,
                                             tolerance_mm=max(0.45, float(hole_tolerance_mm)),
                                             copper_layer_id=lid)
                    cc = chk.get("counts",{})
                    ok=float(cc.get("OK",0)); warn=float(cc.get("WARN",0)); shift=float(cc.get("SHIFT",0)); unk=float(cc.get("UNKNOWN",0)); out=float(cc.get("OUTSIDE",0))
                    vis=max(1.0, ok+warn+shift+unk); total=max(1.0, vis+out)
                    quality=(ok + 0.30*warn - 1.80*shift - 0.45*unk)/vis
                    coverage=vis/total
                    # A full scan should explain almost the complete drill map.  On a clipped scan
                    # OUTSIDE is legitimate, so coverage gets only a small tie-break weight.
                    score=100.0*quality + (30.0 if not touches else 5.0)*coverage
                    score += 0.35*float(alt.get("count",0)) - 0.05*float(alt.get("rms",0.0))
                    angle=registration_angle_deg(alt["H"], alt.get("mirror",False))
                    scored.append((score, alt, cc, angle))
                    orientation_scores.append({"mirror":bool(alt.get("mirror",False)),
                                               "angle_deg":round(float(angle),1),
                                               "anchor_count":int(alt.get("count",0)),
                                               "score":round(score,3), "counts":dict(cc)})
                except Exception:
                    pass
            if scored:
                scored.sort(key=lambda z:(-z[0], -int(z[1].get("count",0)), float(z[1].get("rms",1e9))))
                if scored[0][1] is not reg:
                    reg = scored[0][1]
                    orientation_disambiguated = True
    coarse_reg = reg
    coarse_diam = np.asarray([anchor_diam[i] for i,j in reg["pairs"]], np.float64)
    progress(54, "Уточнение по кромкам монтажных отверстий…")
    reg = refine_registration_to_surface_rims(img, board, reg, coarse_diam, mode=registration_mode)
    H, mirror = reg["H"], reg["mirror"]
    px_per_mm = estimate_registration_px_per_mm(H, reg.get("control_src"))
    progress(65, f"Рендер CAM слоя {cam.get('layers',{}).get(lid,lid)}…")
    copper = render_camvu_layer_to_scan(cam, lid, img.shape, H, mirror)
    # Two display masks: one keeps PTH pads, the other reveals the physical bore.
    # Thermal-relief assistance is display-only; AOI uses the unmodified `copper` mask.
    copper_overlay, thermal_count = apply_cam_thermal_display_assist(copper, cam, H, mirror, lid)
    copper_display = render_camvu_layer_to_scan(
        cam, lid, img.shape, H, mirror, exclude_round_pth_pads=True
    )
    copper_display = suppress_cam_pth_overlay(copper_display, cam, H, mirror)
    copper_display, _ = apply_cam_thermal_display_assist(copper_display, cam, H, mirror, lid)
    opposite = None
    if other_lid is not None:
        progress(73, "Рендер противоположного слоя меди…")
        opposite = render_camvu_layer_to_scan(cam, other_lid, img.shape, H, mirror)
    progress(81, "Проверка отверстий…")
    hole_check = evaluate_cam_holes(img, board, H, mirror, cam, tolerance_mm=hole_tolerance_mm, copper_layer_id=lid)
    progress(90, "Подготовка отображения…")
    prob, copper_evidence_warning = safe_copper_evidence(img, copper, board)
    crop = board_bbox(board, 20)
    board_touches = {
        "top": bool(np.any(board[0,:] > 0)), "bottom": bool(np.any(board[-1,:] > 0)),
        "left": bool(np.any(board[:,0] > 0)), "right": bool(np.any(board[:,-1] > 0)),
    }
    partial_scan = bool(any(board_touches.values()))
    report = {
        "source_type": "camvu",
        "cam_layer_id": int(lid), "cam_layer_name": cam.get("layers",{}).get(lid,str(lid)),
        "cam_other_layer_id": None if other_lid is None else int(other_lid),
        "cam_other_layer_name": None if other_lid is None else cam.get("layers",{}).get(other_lid,str(other_lid)),
        "cam_holes_total": len(cam.get("holes",[])),
        "detected_scan_anchors": int(len(scan_pts)), "drill_anchors": int(len(q)),
        "primary_anchor_diameter_mm": None if primary_diam is None else round(float(primary_diam),3),
        "anchor_policy": "partial-point drill registration",
        "matched_anchors": int(reg["count"]), "coarse_matched_anchors": int(coarse_reg["count"]),
        "orientation_disambiguated": bool(orientation_disambiguated),
        "orientation_scores": orientation_scores,
        "partial_scan": bool(partial_scan), "board_touches_frame": board_touches,
        "rim_refined_anchors": int(reg.get("rim_refined_count",0)),
        "rim_refine_rejected": int(reg.get("rim_refine_rejected",0)),
        "mirrored": bool(mirror), "orientation_angle_deg": round(registration_angle_deg(H, mirror),1), "orientation_candidates_checked": len(orientation_scores), "crop": crop,
        "scan_source_mode": scan_meta.get("source_mode"), "scan_native_shape": scan_meta.get("native_shape"),
        "scan_analysis_shape": scan_meta.get("analysis_shape"), "scan_analysis_scale": round(float(scan_meta.get("analysis_scale",1.0)),4),
        "px_per_mm": round(float(px_per_mm),4),
        "copper_coverage_pct": round(float(np.count_nonzero((copper>127) & (board>0))) / max(1, int(np.count_nonzero(board>0))) * 100.0, 2),
        "copper_evidence_warning": copper_evidence_warning,
        "solder_mask_present": bool(mask_render is not None),
        "solder_mask_path": None if not solder_mask_gerber_path else str(solder_mask_gerber_path),
        "hole_check_counts": hole_check.get("counts",{}),
        "hole_category_counts": hole_check.get("category_counts",{}),
        "thermal_display_count": int(thermal_count),
        "hole_median_shift_mm": None if hole_check.get("median_shift_mm") is None else round(float(hole_check["median_shift_mm"]),3),
        "hole_max_shift_mm": None if hole_check.get("max_shift_mm") is None else round(float(hole_check["max_shift_mm"]),3),
    }
    result = {
        "source_type": "camvu", "scan_full": img, "board_full": board, "gerber_full": copper,
        "gerber_display_full": copper_display, "gerber_overlay_full": copper_overlay,
        "opposite_gerber_full": opposite, "prob_full": prob, "scan_pts": scan_pts, "drill_xy": q,
        "H": H, "mirror": mirror, "crop": crop, "report": report, "alternatives": alternatives,
        "control_src": reg["control_src"].copy(), "control_dst": reg["control_dst"].copy(),
        "control_dst_auto": reg["control_dst_auto"].copy(),
        "control_diam_mm": np.asarray(reg.get("control_diam_mm",coarse_diam),np.float64),
        "registration_mode_requested": registration_mode, "cam_path": str(cam_path), "cam_data": cam,
        "cam_layer_id": int(lid), "cam_other_layer_id": other_lid, "hole_check": hole_check,
        "hole_tolerance_mm": float(hole_tolerance_mm), "coarse_H": coarse_reg["H"].copy(),
        "coarse_control_src": coarse_reg["control_src"].copy(), "coarse_control_dst": coarse_reg["control_dst"].copy(),
        "coarse_control_diam_mm": coarse_diam.copy(), "coarse_registration_model": coarse_reg.get("model","affine"),
        "coarse_anchor_rms_px": float(coarse_reg.get("rms",0.0)), "px_per_mm": float(px_per_mm),
    }
    _refresh_report(result, reg)
    progress(100, "Готово")
    return result


def refit_result(result: dict[str, Any], registration_mode: str = "auto", recompute_copper: bool = True, progress_cb=None) -> dict[str, Any]:
    def progress(value: int, text: str):
        if progress_cb is not None:
            try:
                progress_cb(int(value), str(text))
            except Exception:
                pass

    progress(15, "Пересчёт регистрации…")
    fit = fit_registration_points(result["control_src"], result["control_dst"], mode=registration_mode)
    result["H"] = fit["H"]
    result["registration_mode_requested"] = registration_mode
    result["px_per_mm"] = estimate_registration_px_per_mm(result["H"], result.get("control_src"))
    result.setdefault("report", {})["px_per_mm"] = round(float(result["px_per_mm"]), 4)
    progress(50, "Перерисовка разводки…")
    if result.get("source_type") == "camvu":
        cam = result.get("cam_data") or parse_camvu(result["cam_path"])
        result["cam_data"] = cam
        result["gerber_full"] = render_camvu_layer_to_scan(
            cam, result["cam_layer_id"], result["scan_full"].shape, result["H"], result["mirror"]
        )
        result["gerber_overlay_full"], thermal_count = apply_cam_thermal_display_assist(
            result["gerber_full"], cam, result["H"], result["mirror"], result["cam_layer_id"]
        )
        result["gerber_display_full"] = render_camvu_layer_to_scan(
            cam, result["cam_layer_id"], result["scan_full"].shape, result["H"], result["mirror"],
            exclude_round_pth_pads=True
        )
        result["gerber_display_full"] = suppress_cam_pth_overlay(
            result["gerber_display_full"], cam, result["H"], result["mirror"]
        )
        result["gerber_display_full"], _ = apply_cam_thermal_display_assist(
            result["gerber_display_full"], cam, result["H"], result["mirror"], result["cam_layer_id"]
        )
        result.setdefault("report",{})["thermal_display_count"] = int(thermal_count)
        other_lid = result.get("cam_other_layer_id")
        if other_lid is not None:
            progress(64, "Перерисовка противоположного слоя…")
            result["opposite_gerber_full"] = render_camvu_layer_to_scan(
                cam, other_lid, result["scan_full"].shape, result["H"], result["mirror"]
            )
        progress(70, "Пересчёт положения отверстий…")
        result["hole_check"] = evaluate_cam_holes(
            result["scan_full"], result["board_full"], result["H"], result["mirror"], cam,
            tolerance_mm=float(result.get("hole_tolerance_mm",0.35)), copper_layer_id=result.get("cam_layer_id")
        )
        result.setdefault("report",{})["hole_check_counts"] = result["hole_check"].get("counts",{})
    else:
        raw = render_gerber_to_scan(
            result["gerber_path"], result["scan_full"].shape, result["H"], result["mirror"]
        )
        holes = result.get("gerber_holes") or []
        result["gerber_full"] = subtract_drill_bores_from_copper(
            raw, holes, result["H"], result["mirror"]
        )
        result["gerber_display_full"] = result["gerber_full"]
        result["gerber_overlay_full"] = result["gerber_full"]
        result.setdefault("report", {})["drill_bores_subtracted_from_copper"] = int(len(holes))
        opp_path = result.get("opposite_gerber_path")
        if opp_path:
            progress(64, "Перерисовка противоположного слоя…")
            try:
                opp_raw = render_gerber_to_scan(
                    opp_path, result["scan_full"].shape, result["H"], result["mirror"]
                )
                result["opposite_gerber_full"] = subtract_drill_bores_from_copper(
                    opp_raw, holes, result["H"], result["mirror"]
                )
            except Exception:
                result["opposite_gerber_full"] = None
        if result.get("gerber_holes"):
            progress(70, "Пересчёт положения отверстий…")
            result["hole_check"] = evaluate_gerber_holes(
                result["scan_full"], result["board_full"], result["H"], result["mirror"],
                result["gerber_holes"], tolerance_mm=float(result.get("hole_tolerance_mm",0.35))
            )
            result.setdefault("report",{})["hole_check_counts"] = result["hole_check"].get("counts",{})
    if recompute_copper:
        progress(78, "Пересчёт карты меди…")
        result["prob_full"], warn = safe_copper_evidence(result["scan_full"], result["gerber_full"], result["board_full"])
        result.setdefault("report", {})["copper_evidence_warning"] = warn
    _refresh_report(result, fit)
    progress(100, "Готово")
    return result


def reset_manual_anchors(result: dict[str, Any], registration_mode: str = "auto", progress_cb=None) -> dict[str, Any]:
    result["control_dst"] = result["control_dst_auto"].copy()
    return refit_result(result, registration_mode=registration_mode, recompute_copper=False, progress_cb=progress_cb)


def crop_result(arr: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop
    return arr[y0:y1, x0:x1]
