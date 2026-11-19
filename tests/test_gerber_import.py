"""Input-format regression checks; no customer design files are shipped."""
import tempfile
import unittest
from pathlib import Path

from pcb_aoi.visual_comparator.core import parse_excellon
from pcb_aoi.visual_comparator.app import PCBApp


REF = (-40.0, 0.0, 40.0, 83.0)


class ImportTests(unittest.TestCase):
    def write(self, folder, name, content):
        p = Path(folder) / name
        p.write_text(content, encoding="ascii")
        return p

    def test_compact_excellon_mounting_holes(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.write(d, "Drill.txt", """M48
;FILE_FORMAT=4:3
METRIC,LZ
T01C3.000
%
T01
X-00368Y00032
X00368
Y00793
X-00368
M30
""")
            hits, info = parse_excellon(f, reference_bounds_mm=REF, return_meta=True)
            self.assertEqual(len(hits), 4)
            self.assertEqual(info["fixed_padding"], "right")
            self.assertAlmostEqual(hits[0][0], -36.8)
            self.assertAlmostEqual(hits[0][1], 3.2)
            self.assertAlmostEqual(hits[2][1], 79.3)

    def test_compact_small_npth_column(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.write(d, "NonPlated.txt", """M48
;FILE_FORMAT=4:3
METRIC,LZ
T07C0.700
%
T07
X-000289Y00061
X000289
Y00764
X-000289
M30
""")
            hits, info = parse_excellon(f, reference_bounds_mm=REF, return_meta=True)
            self.assertEqual(info["fixed_padding"], "right")
            self.assertEqual(len(hits), 4)
            self.assertAlmostEqual(hits[0][0], -2.89)
            self.assertAlmostEqual(hits[0][1], 6.1)
            self.assertAlmostEqual(hits[2][1], 76.4)

    def test_routes_are_not_circular_drill_hits(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.write(d, "Slots.txt", """M48
;FILE_FORMAT=4:3
METRIC,LZ
T02C0.600
%
T02
G00X0004325Y000202
M15
G01Y000282
M16
G85X00042Y000282
M30
""")
            hits, info = parse_excellon(f, reference_bounds_mm=REF, return_meta=True)
            self.assertFalse(hits)
            self.assertEqual(info["routed_moves_skipped"], 3)

    def test_outline_comes_from_extension_report(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "board.EXTREP", ".GM27          Board Shape\n.GTL           Top Layer\n.GBL           Bottom Layer\n")
            self.write(d, "board.GM27", "%FSLAX24Y24*%\nM02*\n")
            self.write(d, "board.GTL", "%FSLAX24Y24*%\nM02*\n")
            self.write(d, "board.GBL", "%FSLAX24Y24*%\nM02*\n")
            b = PCBApp._detect_gerber_bundle(PCBApp.__new__(PCBApp), d)
            self.assertEqual(b["profile"].suffix.lower(), ".gm27")
            self.assertEqual(b["top_cu"].suffix.lower(), ".gtl")
            self.assertEqual(b["bottom_cu"].suffix.lower(), ".gbl")


if __name__ == "__main__":
    unittest.main()
