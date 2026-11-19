# pcb_aoi

PCB AOI is an operator-assist toolkit for optical inspection of printed circuit boards.

The project is intentionally structured as a set of inspection modules rather than one monolithic application.

## Current module: Visual Comparator

**Visual Comparator** aligns a PCB scan or PDF page with manufacturing data and overlays copper/drill geometry for operator inspection.

Current capabilities:

- PNG/JPEG/TIFF/BMP and multi-page PDF input with per-page preview/selection;
- Gerber + Excellon folder workflow;
- CAMvu/CAM350 text-database workflow;
- automatic Top/Bottom registration and Bottom-side mirroring;
- profile-assisted rejection of ambiguous 90°/180° orientations;
- full drill-map orientation checking;
- Altium extension report support (`.EXTREP`) for arbitrary `Board Shape`/`.GMnn` profiles;
- robust Excellon fixed-width decoding, including shortened `METRIC,LZ` 4:3/4:4 coordinates;
- slot-routing commands excluded from circular drill counts;
- subpixel drill holes reported as optically unresolved rather than offset defects;
- copper overlay with adjustable opacity;
- PTH / tented VIA / NPTH visualization when solder-mask information is available;
- manual anchor correction;
- export of the current inspection view;
- **Russian / English interface switch without restarting the application**.

> Status: **v0.15.15, pre-1.0**. Visual Comparator is an operator aid, not an automatic production pass/fail system.

## Planned project structure

The repository name is `pcb_aoi`. Visual Comparator is the first module. Planned branches/modules can be added without restructuring the existing comparator:

- **Visual Comparator** — scan/PDF ↔ Gerber/CAM comparison (implemented);
- **LiveCam** — live camera stream with automatic Gerber overlay (planned);
- **Component Inspection** — assembled-board/component inspection (possible future module).

At v0.15.15 the root launcher opens Visual Comparator directly. A module-selection start screen can be added later while keeping the current module isolated under `pcb_aoi/visual_comparator/`.

## Quick start — Windows

1. Install 64-bit Python 3.12 (recommended) or Python 3.13.
2. Run `run_windows.bat`.
3. Choose a scan/PDF. For multi-page PDFs, use **Preview…** to browse pages before selecting one.
4. Leave **Source = Gerber**, select the Gerber/Excellon folder and Top or Bottom.
5. Click **ANALYZE** / **АНАЛИЗ**.

On first launch the script creates `.venv` and installs dependencies.

## Linux / macOS

```sh
sh run_linux_macos.sh
```

A working Tk installation is required.

## Interface language

Use the **Language / Язык** selector in the bottom-right corner of the window. English is the default interface language. Supported languages:

- Русский
- English

The language changes in-place; restarting is not required. Technical layer names such as `Top`, `Bottom`, `Gerber`, `CAM`, `Overlay` and `Anchors` remain unchanged.

## Supported manufacturing data

The Gerber-folder scanner uses file contents, standard extensions and generic naming conventions. It does not depend on a particular PCB/project name or absolute path.

Typical recognition includes:

- Top copper: `.gtl`, X2 `FileFunction=Copper,...,Top`, generic Top/Copper names;
- Bottom copper: `.gbl`, X2 `FileFunction=Copper,...,Bot`, generic Bottom/Copper names;
- solder mask: `.gts` / `.gbs`, X2 metadata and generic mask names;
- board profile: X2 Profile, `profile`, `outline`, `edge_cuts` and similar generic names;
- drill: Excellon `.drl`, `.xln`, `.exc`, `.drd`, plus supported ASCII drill `.txt`.

If the Altium export includes an `.EXTREP` layer report, `Board Shape` profiles with arbitrary `.GMnn` extensions are detected from that report rather than hardcoded filenames.

For Altium-style exports where a binary `.DRL` and readable ASCII `.TXT` with the same stem coexist, the readable ASCII drill file is preferred. Mixed PTH/NPTH tool groups are supported.

CAMvu/CAM350 compatibility includes a legacy aperture-width profile selected by a **content signature**, not by a PCB name or filename.

## Repository layout

```text
pcb_aoi.py                         # current project entry point
visual_comparator.py              # direct Visual Comparator entry point
pcb_aoi/
  __init__.py
  visual_comparator/
    app.py                        # GUI
    core.py                       # registration / Gerber / drill / rendering logic
run_windows.bat
run_linux_macos.sh
build_windows_exe.bat
requirements.txt
```

## Windows EXE

Run:

```bat
build_windows_exe.bat
```

The output is `dist\PCB_AOI.exe`.

## Release hygiene

The public tree contains no test-board images, uploaded scans, board-number-specific branches, absolute development paths, or filename checks tied to the boards used during development.

See `RELEASE_AUDIT.md` for v0.15.15 checks and `PUBLISHING.md` for GitHub publishing instructions.

## License

No open-source license is included yet. Add the chosen license before public release if reuse/modification rights are intended.

## Regression checks

```sh
python -m unittest discover -s tests -v
```

The included tests use generated synthetic CAM inputs; no customer boards or scans are shipped.
