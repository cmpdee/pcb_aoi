#!/usr/bin/env python3
"""PCB AOI launcher.

For v0.15.15 the launcher opens Visual Comparator directly.
Future modules (LiveCam, Component Inspection) can be selected here without
changing the Visual Comparator package.
"""
from pcb_aoi.visual_comparator import main

if __name__ == "__main__":
    main()
