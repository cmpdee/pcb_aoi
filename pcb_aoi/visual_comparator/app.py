#!/usr/bin/env python3
"""PCB AOI Desktop v0.15.15 — operator optical comparator with CAM or Gerber input."""
from __future__ import annotations

import queue
import re
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from .core import (
    analyze_camvu,
    analyze_gerber,
    parse_camvu,
    camvu_copper_layers,
    hole_overlay,
    reclassify_hole_check,
    anchor_overlay,
    crop_result,
    gerber_overlay,
    pdf_page_count,
    load_scan,
    pcb_mask,
    board_bbox,
    refit_result,
    reset_manual_anchors,
)

APP_TITLE = "PCB AOI — Visual Comparator v0.15.15"
PDF_DPI = 150
REGISTRATION_MODE = "auto"


I18N = {
    "ru": {
        "language": "Язык:", "scan": "Скан", "browse": "Обзор…", "pdf_page": "PDF страница:",
        "preview": "Просмотр…", "preview_title": "Предпросмотр страниц скана", "previous": "◀ Назад",
        "next": "Вперёд ▶", "select_page": "Выбрать страницу", "close": "Закрыть",
        "full_page": "Вся страница", "board_preview": "Плата крупным планом",
        "page_of": "Страница {page} из {count}", "preview_failed": "Предпросмотр недоступен: {error}",
        "source": "Источник", "analyze": "АНАЛИЗ", "cu_layer": "Слой Cu", "folder": "Папка",
        "side": "Сторона", "view": "Вид:", "routing": "Разводка:", "holes": "Отверстия",
        "outline": "Контур", "hole_tol": "Допуск отв., мм:", "export": "Экспорт…",
        "edit_anchors": "Правка якорей", "apply": "Применить", "fit": "Вписать", "cancel": "Отмена", "reset_auto": "Сброс авто",
        "anchor_hint": "Перемещайте точки; пересчёт — только после Apply.",
        "control": "Контроль", "hole_colors": "Цвета отверстий",
        "choose_gerber": "Выберите папку с Gerber-файлами",
        "initial_status": "Выберите скан и источник разводки.", "waiting": "Ожидание",
        "not_registered": "Регистрация ещё не выполнена",
        "cam_source_status": "Источник: CAM. Выберите файл и нажмите АНАЛИЗ.",
        "source_changed": "Источник изменён. Нажмите АНАЛИЗ.",
        "open_scan": "Открыть скан PCB / PDF", "scan_pdf": "Скан / PDF", "all_files": "Все файлы",
        "scan_loaded": "Скан загружен. Страниц: {count}", "open_cam": "Открыть CAMvu CAM/PCB",
        "cam_status": "CAM: {holes} отверстий; Cu: {layers}", "not_found_short": "не найден",
        "cam_read_failed": "Не удалось прочитать CAM: {error}", "folder_not_found": "✗ Папка не найдена",
        "missing": "✗ Не найдено: {items}", "gerber_ready": "✓ Gerber-комплект готов: {items}",
        "mask_missing": "⚠ Не найден {side} solder-mask — VIA под маской и открытые PTH не различаются.",
        "opposite_missing": "Второй Cu не найден — текущую сторону анализировать можно.",
        "gerber_folder_dialog": "Папка с Gerber / Excellon",
        "gerber_side_status": "Сторона Gerber: {side}. Нажмите АНАЛИЗ.",
        "select_scan": "Сначала выберите скан/PDF.", "select_cam": "Выберите CAM-файл.",
        "gerber_incomplete": "Gerber-комплект неполный", "starting": "Запуск…",
        "processing": "Обработка идёт. Смотрите строку процесса внизу.", "error": "Ошибка",
        "ready": "Готово", "done_status": "Готово. Для ручной проверки используйте Overlay и прозрачность.",
        "mirrored": "зеркальная", "normal": "обычная", "turn": "поворот", "checked": "проверено вариантов: {count}",
        "profile_orientation": "Контур ориентации: Profile", "copper_area": "Cu по Gerber: {pct:.1f}% площади платы",
        "copper_skipped": "Автоанализ меди: пропущен (Overlay работает)",
        "orientation": "Ориентация", "primary_anchors": "Основные якоря", "matched_anchors": "Совпало якорей",
        "scale": "Масштаб", "holes_summary": "Отверстия", "display_error": "Ошибка отображения: {error}",
        "anchor_on": "Правка якорей включена: переместите точки и нажмите Apply.",
        "anchor_off": "Правка якорей выключена.",
        "anchor_drag": "Якорь #{idx}: dx={dx:+.1f}, dy={dy:+.1f}px — Apply для пересчёта.",
        "anchor_changed": "Якорь #{idx} изменён. Можно править дальше; затем Apply.",
        "apply_correction": "Применение коррекции…", "changes_cancelled": "Неприменённые изменения отменены.",
        "resetting": "Возврат автоматических якорей…", "correction_applied": "Ручная коррекция применена.",
        "auto_restored": "Автоматические якоря восстановлены.", "analyze_first": "Сначала выполните анализ.",
        "save_view": "Сохранить текущий вид", "encode_failed": "Не удалось кодировать изображение",
        "saved": "Сохранено: {path}", "language_changed": "Язык интерфейса изменён.",
    },
    "en": {
        "language": "Language:", "scan": "Scan", "browse": "Browse…", "pdf_page": "PDF page:",
        "preview": "Preview…", "preview_title": "Scan page preview", "previous": "◀ Previous",
        "next": "Next ▶", "select_page": "Use this page", "close": "Close",
        "full_page": "Full page", "board_preview": "Board close-up",
        "page_of": "Page {page} of {count}", "preview_failed": "Preview unavailable: {error}",
        "source": "Source", "analyze": "ANALYZE", "cu_layer": "Cu layer", "folder": "Folder",
        "side": "Side", "view": "View:", "routing": "Copper:", "holes": "Holes",
        "outline": "Outline", "hole_tol": "Hole tol., mm:", "export": "Export…",
        "edit_anchors": "Edit anchors", "apply": "Apply", "fit": "Fit", "cancel": "Cancel", "reset_auto": "Reset auto",
        "anchor_hint": "Move points; recalculation happens only after Apply.",
        "control": "Inspection", "hole_colors": "Hole colors",
        "choose_gerber": "Choose a folder with Gerber files",
        "initial_status": "Choose a scan and copper data source.", "waiting": "Waiting",
        "not_registered": "Registration has not been run yet",
        "cam_source_status": "Source: CAM. Choose a file and click ANALYZE.",
        "source_changed": "Source changed. Click ANALYZE.",
        "open_scan": "Open PCB scan / PDF", "scan_pdf": "Scan / PDF", "all_files": "All files",
        "scan_loaded": "Scan loaded. Pages: {count}", "open_cam": "Open CAMvu CAM/PCB",
        "cam_status": "CAM: {holes} holes; Cu: {layers}", "not_found_short": "not found",
        "cam_read_failed": "Could not read CAM: {error}", "folder_not_found": "✗ Folder not found",
        "missing": "✗ Missing: {items}", "gerber_ready": "✓ Gerber set ready: {items}",
        "mask_missing": "⚠ {side} solder mask not found — tented vias and open PTH cannot be distinguished.",
        "opposite_missing": "Opposite Cu layer not found — the selected side can still be analyzed.",
        "gerber_folder_dialog": "Gerber / Excellon folder",
        "gerber_side_status": "Gerber side: {side}. Click ANALYZE.",
        "select_scan": "Choose a scan/PDF first.", "select_cam": "Choose a CAM file.",
        "gerber_incomplete": "Gerber set is incomplete", "starting": "Starting…",
        "processing": "Processing. See the progress line below.", "error": "Error",
        "ready": "Ready", "done_status": "Ready. Use Overlay and opacity for manual inspection.",
        "mirrored": "mirrored", "normal": "normal", "turn": "rotation", "checked": "candidates checked: {count}",
        "profile_orientation": "Orientation outline: Profile", "copper_area": "Gerber Cu: {pct:.1f}% of board area",
        "copper_skipped": "Automatic copper evidence: skipped (Overlay still works)",
        "orientation": "Orientation", "primary_anchors": "Primary anchors", "matched_anchors": "Matched anchors",
        "scale": "Scale", "holes_summary": "Holes", "display_error": "Display error: {error}",
        "anchor_on": "Anchor editing enabled: move points and click Apply.",
        "anchor_off": "Anchor editing disabled.",
        "anchor_drag": "Anchor #{idx}: dx={dx:+.1f}, dy={dy:+.1f}px — click Apply to recalculate.",
        "anchor_changed": "Anchor #{idx} changed. Continue editing or click Apply.",
        "apply_correction": "Applying correction…", "changes_cancelled": "Unapplied changes cancelled.",
        "resetting": "Restoring automatic anchors…", "correction_applied": "Manual correction applied.",
        "auto_restored": "Automatic anchors restored.", "analyze_first": "Run analysis first.",
        "save_view": "Save current view", "encode_failed": "Could not encode image",
        "saved": "Saved: {path}", "language_changed": "Interface language changed.",
    },
}

PROGRESS_EN = {
    "Подготовка эталонных сканов…": "Preparing reference scans…",
    "Сохранение эталонной модели…": "Saving reference model…",
    "Эталонная модель готова": "Reference model ready",
    "Сравнение с оптическим эталоном…": "Comparing with optical reference…",
    "Оптический эталон применён": "Optical reference applied",
    "Чтение скана / PDF…": "Reading scan / PDF…",
    "Поиск платы…": "Detecting board…",
    "Поиск контура платы…": "Detecting board outline…",
    "Поиск монтажных отверстий…": "Detecting mounting holes…",
    "Поиск отверстий…": "Detecting holes…",
    "Чтение Excellon…": "Reading Excellon…",
    "Чтение Excellon и выбор механических якорей…": "Reading Excellon and selecting mechanical anchors…",
    "Грубая регистрация платы…": "Coarse board registration…",
    "Проверка поворота по контуру и полной карте отверстий…": "Checking orientation using profile and full drill map…",
    "Проверка поворота по полной карте отверстий…": "Checking orientation using full drill map…",
    "Уточнение по кромкам отверстий…": "Refining on hole rims…",
    "Уточнение по кромкам монтажных отверстий…": "Refining on mounting-hole rims…",
    "Рендер Gerber…": "Rendering Gerber…",
    "Рендер Gerber в координаты скана…": "Rendering Gerber into scan coordinates…",
    "Рендер второго слоя…": "Rendering second layer…",
    "Рендер противоположного слоя меди…": "Rendering opposite copper layer…",
    "Рендер solder-mask…": "Rendering solder mask…",
    "Проверка отверстий…": "Checking holes…",
    "Подготовка отображения…": "Preparing view…",
    "Чтение CAM…": "Reading CAM…",
    "Выбор механических якорей из CAM…": "Selecting mechanical anchors from CAM…",
    "Пересчёт регистрации…": "Recalculating registration…",
    "Перерисовка разводки…": "Redrawing copper…",
    "Перерисовка противоположного слоя…": "Redrawing opposite layer…",
    "Пересчёт положения отверстий…": "Recalculating hole positions…",
    "Пересчёт карты меди…": "Recalculating copper map…",
    "Готово": "Ready",
}


def cv_bgr_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class ImageCanvas(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg="#252525", highlightthickness=0)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self._source = None; self._photo = None; self._item = None
        self.zoom = 1.0; self.fit_mode = True
        self._drag_pan = False; self._point_drag_active = False
        self.point_press_cb = None; self.point_move_cb = None; self.point_release_cb = None
        self._drag_preview_start = None; self._drag_preview_items = []
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda e: self._wheel_linux(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._wheel_linux(e, -1))
        self.canvas.bind("<ButtonPress-1>", self._mouse_press)
        self.canvas.bind("<B1-Motion>", self._mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_release)

    def set_point_callbacks(self, press=None, move=None, release=None):
        self.point_press_cb = press; self.point_move_cb = move; self.point_release_cb = release

    def _clear_drag_preview(self):
        for item in self._drag_preview_items:
            try: self.canvas.delete(item)
            except Exception: pass
        self._drag_preview_items = []; self._drag_preview_start = None

    def _draw_drag_preview(self, p_img, start=False):
        if p_img is None: return
        z = max(1e-9, self.current_zoom()); x, y = float(p_img[0])*z, float(p_img[1])*z
        if start or self._drag_preview_start is None:
            self._clear_drag_preview(); self._drag_preview_start = (x, y)
        else:
            while len(self._drag_preview_items) > 1:
                item = self._drag_preview_items.pop()
                try: self.canvas.delete(item)
                except Exception: pass
        sx, sy = self._drag_preview_start
        if not self._drag_preview_items:
            r0=5; self._drag_preview_items.append(self.canvas.create_oval(sx-r0,sy-r0,sx+r0,sy+r0,outline="#00ffff",width=2,tags=("drag_preview",)))
        self._drag_preview_items.append(self.canvas.create_line(sx,sy,x,y,fill="#00ffff",width=2,dash=(4,3),tags=("drag_preview",)))
        r=10; self._drag_preview_items.append(self.canvas.create_oval(x-r,y-r,x+r,y+r,fill="#00d050",outline="#ffff00",width=3,tags=("drag_preview",)))
        self._drag_preview_items.append(self.canvas.create_line(x-r-4,y,x+r+4,y,fill="#ffff00",width=2,tags=("drag_preview",)))
        self._drag_preview_items.append(self.canvas.create_line(x,y-r-4,x,y+r+4,fill="#ffff00",width=2,tags=("drag_preview",)))
        self.canvas.tag_raise("drag_preview")

    def set_image(self, img: np.ndarray | None, fit=True):
        self._source = img
        if fit: self.fit_mode = True
        self._redraw()

    def fit(self): self.fit_mode = True; self._redraw()
    def actual_size(self): self.fit_mode = False; self.zoom = 1.0; self._redraw()

    def _fit_zoom(self):
        if self._source is None: return 1.0
        h,w=self._source.shape[:2]; cw=max(50,self.canvas.winfo_width()-8); ch=max(50,self.canvas.winfo_height()-8)
        return min(cw/w,ch/h,1.0)
    def current_zoom(self): return self._fit_zoom() if self.fit_mode else self.zoom

    def _redraw(self):
        if self._source is None:
            self.canvas.delete("all"); return
        z=max(0.05,min(8.0,self.current_zoom())); h,w=self._source.shape[:2]
        nw,nh=max(1,int(round(w*z))),max(1,int(round(h*z)))
        disp=cv2.resize(self._source,(nw,nh),interpolation=cv2.INTER_AREA if z<1 else cv2.INTER_LINEAR)
        self._photo=ImageTk.PhotoImage(cv_bgr_to_pil(disp)); self.canvas.delete("all")
        self._item=self.canvas.create_image(0,0,image=self._photo,anchor="nw"); self.canvas.configure(scrollregion=(0,0,nw,nh))

    def event_to_image(self,e):
        if self._source is None: return None
        z=max(1e-9,self.current_zoom()); return np.array([self.canvas.canvasx(e.x)/z,self.canvas.canvasy(e.y)/z],np.float64)
    def _wheel(self,e): self._zoom_step(1 if e.delta>0 else -1)
    def _wheel_linux(self,e,step): self._zoom_step(step)
    def _zoom_step(self,step):
        self.fit_mode=False; self.zoom=max(0.05,min(8.0,self.zoom*(1.15 if step>0 else 1/1.15))); self._redraw()
    def _mouse_press(self,e):
        p=self.event_to_image(e); handled=False; preview_p=p
        if p is not None and self.point_press_cb is not None:
            try:
                result=self.point_press_cb(p,self.current_zoom())
                if isinstance(result,dict):
                    handled=bool(result.get("handled",False)); preview_p=np.asarray(result.get("point",p),np.float64)
                else: handled=bool(result)
            except Exception: handled=False
        if handled:
            self._point_drag_active=True; self._drag_pan=False; self.canvas.configure(cursor="fleur"); self._draw_drag_preview(preview_p,start=True)
        else:
            self._point_drag_active=False; self._drag_pan=True; self.canvas.scan_mark(e.x,e.y)
    def _mouse_move(self,e):
        if self._point_drag_active:
            p=self.event_to_image(e)
            if p is not None:
                self._draw_drag_preview(p,start=False)
                if self.point_move_cb is not None:
                    try: self.point_move_cb(p)
                    except Exception: pass
        elif self._drag_pan: self.canvas.scan_dragto(e.x,e.y,gain=1)
    def _mouse_release(self,e):
        if self._point_drag_active and self.point_release_cb is not None:
            p=self.event_to_image(e)
            if p is not None: self.point_release_cb(p)
        self._point_drag_active=False; self._drag_pan=False; self.canvas.configure(cursor=""); self._clear_drag_preview()


class ScanPagePreview(tk.Toplevel):
    """Small, disposable page selector; never loads every PDF page into memory."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.page = max(1, min(parent.page_count, int(parent.page_var.get())))
        self.title(parent.tr("preview_title"))
        self.geometry("1000x700")
        self.minsize(740, 460)
        self._photo_full = None
        self._photo_board = None
        self.bind("<Left>", lambda e:self.navigate(-1))
        self.bind("<Right>", lambda e:self.navigate(1))
        self.bind("<Escape>", lambda e:self.destroy())

        bar = ttk.Frame(self,padding=8); bar.pack(side="top",fill="x")
        ttk.Button(bar,text=parent.tr("previous"),command=lambda:self.navigate(-1)).pack(side="left")
        self.page_label = ttk.Label(bar,text="",width=22,anchor="center")
        self.page_label.pack(side="left",padx=8)
        ttk.Button(bar,text=parent.tr("next"),command=lambda:self.navigate(1)).pack(side="left")
        ttk.Button(bar,text=parent.tr("close"),command=self.destroy).pack(side="right",padx=(8,0))
        ttk.Button(bar,text=parent.tr("select_page"),command=self.accept).pack(side="right")
        body=ttk.Frame(self,padding=(8,2,8,8));body.pack(fill="both",expand=True)
        body.columnconfigure(0,weight=2);body.columnconfigure(1,weight=3)
        body.rowconfigure(1,weight=1)
        ttk.Label(body,text=parent.tr("full_page")).grid(row=0,column=0,pady=5)
        ttk.Label(body,text=parent.tr("board_preview")).grid(row=0,column=1,pady=5)
        self.full_label=ttk.Label(body,anchor="center",relief="sunken")
        self.full_label.grid(row=1,column=0,sticky="nsew",padx=(0,8))
        self.board_label=ttk.Label(body,anchor="center",relief="sunken")
        self.board_label.grid(row=1,column=1,sticky="nsew")
        self.after(75,self.refresh)

    @staticmethod
    def _photo(img, max_w, max_h):
        h,w=img.shape[:2]
        scale=min(float(max_w)/max(1,w),float(max_h)/max(1,h),1.0)
        small=cv2.resize(img,(max(1,int(w*scale)),max(1,int(h*scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        return ImageTk.PhotoImage(cv_bgr_to_pil(small))

    def refresh(self):
        try:
            self.page_label.config(text=self.parent.tr("page_of",page=self.page,count=self.parent.page_count))
            img=load_scan(self.parent.scan_path.get(),pdf_page=self.page-1,
                          pdf_dpi=95,normalize_max_side=1600)
            self._photo_full=self._photo(img,350,540)
            self.full_label.config(image=self._photo_full,text="")
            try:
                mask=pcb_mask(img)
                x0,y0,x1,y1=board_bbox(mask,30)
                cropped=img[y0:y1,x0:x1]
            except Exception:
                cropped=img
            self._photo_board=self._photo(cropped,560,540)
            self.board_label.config(image=self._photo_board,text="")
        except Exception as e:
            self.full_label.config(text=self.parent.tr("preview_failed",error=e),image="")
            self.board_label.config(image="")

    def navigate(self,delta):
        next_page=max(1,min(self.parent.page_count,self.page+delta))
        if next_page != self.page:
            self.page=next_page;self.refresh()

    def accept(self):
        self.parent.page_var.set(self.page)
        self.parent._page_changed()
        self.destroy()


class PCBApp(tk.Tk):
    def __init__(self):
        super().__init__(); self.language="en"; self.lang_var=tk.StringVar(value="English")
        self.title(APP_TITLE); self.geometry("1460x860"); self.minsize(1080,650)
        self.scan_path=tk.StringVar(); self.source_mode=tk.StringVar(value="Gerber")
        self.cam_path=tk.StringVar(); self.cam_layer_var=tk.StringVar(); self.cam_layer_choices=[]
        self.gerber_dir_var=tk.StringVar(); self.gerber_side_var=tk.StringVar(value="Top")
        self._last_gerber_folder=""
        self.gerber_bundle={}; self.gerber_bundle_status=tk.StringVar(value=self.tr("choose_gerber"))
        self.page_var=tk.IntVar(value=1); self.page_count=1
        self.view_var=tk.StringVar(value="Overlay"); self.alpha_var=tk.DoubleVar(value=45)
        self.show_holes_var=tk.BooleanVar(value=True); self.show_outline_var=tk.BooleanVar(value=False)
        self.hole_tol_var=tk.DoubleVar(value=0.50); self.edit_anchors_var=tk.BooleanVar(value=False)
        self.status_var=tk.StringVar(value=self.tr("initial_status"))
        self.progress_text_var=tk.StringVar(value=self.tr("waiting")); self.progress_pct_var=tk.StringVar(value="0%")
        self._progress_state=("key","waiting")
        self.metrics_var=tk.StringVar(value=self.tr("not_registered"))
        self.legend_var=tk.StringVar()
        self.result=None; self.pending_control_dst=None; self.busy=False; self._selected_anchor=None
        self._worker_queue=queue.Queue()
        self._preview_after=None
        self._build_ui(); self._update_source_ui(); self.after(80,self._poll_worker_queue)

    def tr(self, key, **kwargs):
        text=I18N.get(self.language,I18N["ru"]).get(key,I18N["ru"].get(key,key))
        try: return text.format(**kwargs)
        except Exception: return text

    def _translate_progress(self, text):
        if self.language != "en": return str(text)
        s=str(text)
        if s in PROGRESS_EN: return PROGRESS_EN[s]
        m=re.fullmatch(r"Совмещение эталона (\d+)/(\d+)…",s)
        if m: return f"Aligning reference {m.group(1)}/{m.group(2)}…"
        m=re.fullmatch(r"Оптическая модель (\d+)/(\d+)…",s)
        if m: return f"Building optical model {m.group(1)}/{m.group(2)}…"
        m=re.fullmatch(r"Рендер CAM слоя (.+)…",s)
        if m: return f"Rendering CAM layer {m.group(1)}…"
        return s

    def _language_changed(self, _event=None):
        if self.busy:
            self.lang_var.set("English" if self.language=="en" else "Русский")
            return
        self.language="en" if self.lang_var.get()=="English" else "ru"
        for child in list(self.winfo_children()):
            child.destroy()
        self.title(APP_TITLE)
        self._build_ui(); self._update_source_ui()
        self.metrics_var.set(self._metrics_text()); self.legend_var.set(self._legend_text())
        try: self.progress["value"]=int(str(self.progress_pct_var.get()).rstrip("%") or 0)
        except Exception: pass
        self._refresh_progress_language()
        if self.result is not None:
            self.refresh_view(fit=True)
        elif self.scan_path.get():
            self._preview_current_page()
        if self.view_var.get()=="Anchors": self._view_changed()
        self.status_var.set(self.tr("language_changed"))

    def _refresh_progress_language(self):
        state=getattr(self,"_progress_state",("key","waiting"))
        kind,value=state
        if kind=="key":
            self.progress_text_var.set(self.tr(value))
        else:
            self.progress_text_var.set(self._translate_progress(value))

    def _build_ui(self):
        top=ttk.Frame(self,padding=6); top.pack(side="top",fill="x"); top.columnconfigure(1,weight=1)
        ttk.Label(top,text=self.tr("scan"),width=9).grid(row=0,column=0,sticky="w",padx=(0,4),pady=2)
        ttk.Entry(top,textvariable=self.scan_path).grid(row=0,column=1,sticky="ew",pady=2)
        ttk.Button(top,text=self.tr("browse"),command=self.pick_scan).grid(row=0,column=2,padx=4,pady=2)
        pagebox=ttk.Frame(top); pagebox.grid(row=0,column=3,padx=(10,0),sticky="e")
        ttk.Label(pagebox,text=self.tr("pdf_page")).pack(side="left")
        self.page_spin=ttk.Spinbox(pagebox,from_=1,to=max(1,self.page_count),width=4,
                                   textvariable=self.page_var,command=self._page_changed)
        self.page_spin.pack(side="left",padx=(4,0))
        self.page_spin.bind("<Return>",lambda e:self._page_changed())
        self.page_spin.bind("<FocusOut>",lambda e:self._page_changed())
        ttk.Button(pagebox,text=self.tr("preview"),command=self.open_scan_preview).pack(side="left",padx=(6,0))

        ttk.Label(top,text=self.tr("source"),width=9).grid(row=1,column=0,sticky="w",padx=(0,4),pady=2)
        self.source_combo=ttk.Combobox(top,state="readonly",width=12,textvariable=self.source_mode,values=["CAM","Gerber"])
        self.source_combo.grid(row=1,column=1,sticky="w",pady=2); self.source_combo.bind("<<ComboboxSelected>>",lambda e:self._source_changed())
        self.analysis_btn=ttk.Button(top,text=self.tr("analyze"),command=self.start_analysis); self.analysis_btn.grid(row=1,column=3,rowspan=4,padx=(10,0),sticky="nsew")

        self.source_holder=ttk.Frame(top); self.source_holder.grid(row=2,column=0,columnspan=3,sticky="ew"); self.source_holder.columnconfigure(1,weight=1)
        self.cam_frame=ttk.Frame(self.source_holder); self.cam_frame.columnconfigure(1,weight=1)
        ttk.Label(self.cam_frame,text="CAM",width=9).grid(row=0,column=0,sticky="w",padx=(0,4),pady=2)
        ttk.Entry(self.cam_frame,textvariable=self.cam_path).grid(row=0,column=1,sticky="ew",pady=2)
        ttk.Button(self.cam_frame,text=self.tr("browse"),command=self.pick_cam).grid(row=0,column=2,padx=4,pady=2)
        ttk.Label(self.cam_frame,text=self.tr("cu_layer"),width=9).grid(row=1,column=0,sticky="w",padx=(0,4),pady=2)
        self.cam_layer_combo=ttk.Combobox(self.cam_frame,state="readonly",textvariable=self.cam_layer_var,values=[])
        self.cam_layer_combo.grid(row=1,column=1,sticky="ew",pady=2); self.cam_layer_combo.bind("<<ComboboxSelected>>",lambda e:self._source_data_changed())

        self.gerber_frame=ttk.Frame(self.source_holder); self.gerber_frame.columnconfigure(1,weight=1)
        ttk.Label(self.gerber_frame,text=self.tr("folder"),width=9).grid(row=0,column=0,sticky="w",padx=(0,4),pady=2)
        self.gerber_dir_entry=ttk.Entry(self.gerber_frame,textvariable=self.gerber_dir_var)
        self.gerber_dir_entry.grid(row=0,column=1,sticky="ew",pady=2)
        self.gerber_dir_entry.bind("<Return>",lambda e:self._scan_gerber_folder())
        self.gerber_dir_entry.bind("<FocusOut>",lambda e:self._scan_gerber_folder())
        ttk.Button(self.gerber_frame,text=self.tr("browse"),command=self.pick_gerber_folder).grid(row=0,column=2,padx=4,pady=2)
        ttk.Label(self.gerber_frame,text=self.tr("side"),width=9).grid(row=1,column=0,sticky="w",padx=(0,4),pady=2)
        self.gerber_side_combo=ttk.Combobox(self.gerber_frame,state="readonly",width=12,textvariable=self.gerber_side_var,values=["Top","Bottom"])
        self.gerber_side_combo.grid(row=1,column=1,sticky="w",pady=2)
        self.gerber_side_combo.bind("<<ComboboxSelected>>",lambda e:self._gerber_side_changed())
        self.gerber_ready_label=tk.Label(self.gerber_frame,textvariable=self.gerber_bundle_status,anchor="w",justify="left",fg="#666666",bg=self.cget("bg"))
        self.gerber_ready_label.grid(row=2,column=1,columnspan=2,sticky="ew",pady=(2,0))

        controls=ttk.Frame(self,padding=(6,0,6,5)); controls.pack(side="top",fill="x")
        ttk.Label(controls,text=self.tr("view")).pack(side="left")
        self.view_combo=ttk.Combobox(controls,state="readonly",width=11,textvariable=self.view_var,values=["Scan","Overlay","Anchors"])
        self.view_combo.pack(side="left",padx=(4,12)); self.view_combo.bind("<<ComboboxSelected>>",self._view_changed)
        ttk.Label(controls,text=self.tr("routing")).pack(side="left")
        ttk.Scale(controls,from_=0,to=100,length=190,variable=self.alpha_var,command=lambda v:self.refresh_view()).pack(side="left",padx=(4,12))
        ttk.Checkbutton(controls,text=self.tr("holes"),variable=self.show_holes_var,command=self.refresh_view).pack(side="left",padx=(6,4))
        ttk.Checkbutton(controls,text=self.tr("outline"),variable=self.show_outline_var,command=self.refresh_view).pack(side="left",padx=(4,8))
        ttk.Label(controls,text=self.tr("hole_tol")).pack(side="left")
        self.hole_tol_spin=ttk.Spinbox(controls,from_=0.10,to=1.50,increment=0.05,width=5,textvariable=self.hole_tol_var,command=self._hole_tolerance_changed)
        self.hole_tol_spin.pack(side="left",padx=(4,10)); self.hole_tol_spin.bind("<Return>",self._hole_tolerance_changed); self.hole_tol_spin.bind("<FocusOut>",self._hole_tolerance_changed)
        ttk.Button(controls,text=self.tr("fit"),command=lambda:self.viewer.fit()).pack(side="left",padx=2)
        ttk.Button(controls,text="100%",command=lambda:self.viewer.actual_size()).pack(side="left",padx=2)
        ttk.Button(controls,text=self.tr("export"),command=self.export_view).pack(side="left",padx=(8,2))

        self.anchor_tools=ttk.Frame(self,padding=(6,0,6,5))
        ttk.Checkbutton(self.anchor_tools,text=self.tr("edit_anchors"),variable=self.edit_anchors_var,command=self.toggle_anchor_edit).pack(side="left")
        ttk.Button(self.anchor_tools,text=self.tr("apply"),command=self.apply_edits).pack(side="left",padx=(8,2))
        ttk.Button(self.anchor_tools,text=self.tr("cancel"),command=self.cancel_edits).pack(side="left",padx=2)
        ttk.Button(self.anchor_tools,text=self.tr("reset_auto"),command=self.reset_anchors).pack(side="left",padx=2)
        ttk.Label(self.anchor_tools,text=self.tr("anchor_hint")).pack(side="left",padx=(12,0))

        middle=ttk.Panedwindow(self,orient="horizontal"); middle.pack(fill="both",expand=True)
        self.viewer=ImageCanvas(middle); self.viewer.set_point_callbacks(self._anchor_press,self._anchor_move,self._anchor_release); middle.add(self.viewer,weight=6)
        side=ttk.Frame(middle,padding=8,width=300); middle.add(side,weight=1)
        ttk.Label(side,text=self.tr("control"),font=("TkDefaultFont",11,"bold")).pack(anchor="w")
        ttk.Label(side,textvariable=self.metrics_var,justify="left",wraplength=290).pack(anchor="w",fill="x",pady=(6,12))
        ttk.Separator(side).pack(fill="x",pady=4)
        ttk.Label(side,text=self.tr("hole_colors"),font=("TkDefaultFont",10,"bold")).pack(anchor="w",pady=(7,4))
        ttk.Label(side,textvariable=self.legend_var,justify="left",wraplength=290).pack(anchor="w",fill="x")

        status=ttk.Frame(self,padding=(6,4)); status.pack(side="bottom",fill="x")
        langbox=ttk.Frame(status); langbox.pack(side="right",padx=(14,0))
        ttk.Label(langbox,text=self.tr("language")).pack(side="left")
        self.lang_combo=ttk.Combobox(langbox,state="readonly",width=8,textvariable=self.lang_var,values=["English","Русский"])
        self.lang_combo.pack(side="left",padx=(5,0)); self.lang_combo.bind("<<ComboboxSelected>>",self._language_changed)
        self.progress=ttk.Progressbar(status,mode="determinate",maximum=100,length=260); self.progress.pack(side="left",padx=(0,6))
        ttk.Label(status,textvariable=self.progress_pct_var,width=5).pack(side="left",padx=(0,8))
        ttk.Label(status,textvariable=self.progress_text_var,width=42).pack(side="left",padx=(0,10))
        ttk.Separator(status,orient="vertical").pack(side="left",fill="y",padx=(0,10))
        ttk.Label(status,textvariable=self.status_var).pack(side="left",fill="x",expand=True)

    @staticmethod
    def _path_row(frame,row,label,var,command):
        ttk.Label(frame,text=label,width=9).grid(row=row,column=0,sticky="w",padx=(0,4),pady=2)
        ttk.Entry(frame,textvariable=var).grid(row=row,column=1,sticky="ew",pady=2)
        ttk.Button(frame,text="Обзор…",command=command).grid(row=row,column=2,padx=4,pady=2)

    def _source_changed(self):
        self.result=None; self._update_source_ui()
        if self.source_mode.get()=="Gerber": self._scan_gerber_folder()
        else: self.status_var.set(self.tr("cam_source_status"))
    def _source_data_changed(self): self.result=None; self.status_var.set(self.tr("source_changed"))

    def _update_source_ui(self):
        self.cam_frame.grid_forget(); self.gerber_frame.grid_forget()
        if self.source_mode.get()=="CAM": self.cam_frame.grid(row=0,column=0,sticky="ew")
        else:
            self.gerber_frame.grid(row=0,column=0,sticky="ew")
            self.after_idle(self._scan_gerber_folder)
        self.legend_var.set(self._legend_text())

    def _legend_text(self):
        b=getattr(self,"gerber_bundle",{}) or {}; side=self.gerber_side_var.get()
        mask=b.get("top_mask") if side=="Top" else b.get("bottom_mask")
        if self.language=="en":
            if self.source_mode.get()=="CAM" or mask:
                return ("Green — open PTH\nViolet — tented VIA\nCyan — NPTH\n"
                        "Orange — borderline offset\nRed — significant offset\n"
                        "Yellow — optical result uncertain\nOUT — outside frame")
            return ("Green — plated/PTH\nCyan — NPTH\nOrange — borderline offset\n"
                    "Red — significant offset\nYellow — optical result uncertain\nOUT — outside frame\n"
                    "⚠ Solder mask not found: PTH and tented VIA cannot be distinguished")
        if self.source_mode.get()=="CAM" or mask:
            return ("Зелёный — PTH, маска открыта\nФиолетовый — VIA под маской\nГолубой — NPTH\n"
                    "Оранжевый — пограничное смещение\nКрасный — заметное смещение\n"
                    "Жёлтый — оптика неуверенна\nOUT — вне кадра")
        return ("Зелёный — plated/PTH\nГолубой — NPTH\nОранжевый — пограничное смещение\n"
                "Красный — заметное смещение\nЖёлтый — оптика неуверенна\nOUT — вне кадра\n"
                "⚠ Solder-mask не найден: PTH/VIA под маской не различаются")

    def pick_scan(self):
        p=filedialog.askopenfilename(title=self.tr("open_scan"),filetypes=[(self.tr("scan_pdf"),"*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.pdf"),(self.tr("all_files"),"*.*")])
        if not p:return
        self.scan_path.set(p)
        try:self.page_count=pdf_page_count(p)
        except Exception:self.page_count=1
        self.page_spin.configure(to=max(1,self.page_count)); self.page_var.set(1); self.result=None
        self.status_var.set(self.tr("scan_loaded",count=self.page_count)); self._set_progress(0,self.tr("waiting"))
        self._preview_current_page()

    def open_scan_preview(self):
        if self.busy or not self.scan_path.get() or not Path(self.scan_path.get()).is_file():
            messagebox.showerror(APP_TITLE,self.tr("select_scan"));return
        ScanPagePreview(self)

    def _page_changed(self):
        if self.busy or not self.scan_path.get():return
        try: page=int(self.page_var.get())
        except (ValueError,tk.TclError): return
        if not 1 <= page <= max(1,self.page_count): return
        self.result=None
        self.metrics_var.set(self.tr("not_registered"))
        if self._preview_after is not None:
            try:self.after_cancel(self._preview_after)
            except Exception:pass
        self._preview_after=self.after(140,self._preview_current_page)

    def _preview_current_page(self):
        self._preview_after=None
        path=self.scan_path.get().strip()
        if self.busy or not path or not Path(path).is_file(): return
        try:
            page=int(self.page_var.get())-1
            if not 0 <= page < self.page_count: return
            image=load_scan(path,pdf_page=page,pdf_dpi=90,normalize_max_side=1500)
            try:
                x0,y0,x1,y1=board_bbox(pcb_mask(image),24)
                image=image[y0:y1,x0:x1]
            except Exception:pass
            self.viewer.set_image(image,fit=True)
        except Exception as e:
            self.status_var.set(self.tr("preview_failed",error=e))

    def pick_cam(self):
        p=filedialog.askopenfilename(title=self.tr("open_cam"),filetypes=[("CAMvu CAM/PCB","*.cam *.pcb"),(self.tr("all_files"),"*.*")])
        if not p:return
        self.cam_path.set(p); self._populate_cam_layers(p); self.result=None
    def _populate_cam_layers(self,path):
        try:
            cam=parse_camvu(path); layers=camvu_copper_layers(cam); names=[name for _,name in layers]
            self.cam_layer_choices=layers; self.cam_layer_combo.configure(values=names)
            if names and self.cam_layer_var.get() not in names:self.cam_layer_var.set(names[0])
            self.status_var.set(self.tr("cam_status",holes=len(cam.get("holes",[])),layers=", ".join(names) or self.tr("not_found_short")))
        except Exception as e:
            self.cam_layer_choices=[]; self.cam_layer_combo.configure(values=[]); self.status_var.set(self.tr("cam_read_failed",error=e))
    @staticmethod
    def _read_header(path: Path, limit: int = 131072) -> str:
        try:
            return path.read_bytes()[:limit].decode("latin-1", errors="ignore").upper()
        except Exception:
            return ""

    @staticmethod
    def _looks_like_top_cu(path: Path, text: str) -> bool:
        n=path.name.lower(); stem=path.stem.lower(); ext=path.suffix.lower()
        if "TF.FILEFUNCTION,COPPER" in text and ",TOP" in text: return True
        if ext==".gtl": return True
        if any(k in n for k in ("mask","silk","paste","mskg","legend")): return False
        if re.search(r"(^|[_\-.])(f[_\-.]?cu|top[_\-.]?(cu|copper))([_\-.]|$)", n): return True
        if "copper" in n and ("_top" in n or "-top" in n or n.endswith("top.gbr")): return True
        if re.fullmatch(r"top", stem): return True
        if re.fullmatch(r"top[_\-.]?layer(?:[_\-.]?1)?", stem): return True
        return False

    @staticmethod
    def _looks_like_bottom_cu(path: Path, text: str) -> bool:
        n=path.name.lower(); stem=path.stem.lower(); ext=path.suffix.lower()
        if "TF.FILEFUNCTION,COPPER" in text and (",BOT" in text or ",BOTTOM" in text): return True
        if ext==".gbl": return True
        if any(k in n for k in ("mask","silk","paste","mskg","legend")): return False
        if re.search(r"(^|[_\-.])(b[_\-.]?cu|bot(tom)?[_\-.]?(cu|copper))([_\-.]|$)", n): return True
        if "copper" in n and ("_bot" in n or "_bottom" in n or "-bot" in n or n.endswith("bot.gbr") or n.endswith("bottom.gbr")): return True
        if re.fullmatch(r"bottom", stem): return True
        if re.fullmatch(r"bottom[_\-.]?layer(?:[_\-.]?2)?", stem): return True
        return False

    def _detect_gerber_bundle(self, folder: str | Path) -> dict:
        root=Path(folder)
        out={"top_cu":None,"bottom_cu":None,"pth":[],"npth":[],"drill_mixed":[],"drill_unknown":[],
             "top_mask":None,"bottom_mask":None,"profile":None,"job":None,"files":[]}
        if not root.is_dir(): return out
        files=[p for p in root.rglob("*") if p.is_file()]
        out["files"]=files
        gerber_ext={".gbr",".ger",".gtl",".gbl",".gts",".gbs",".pho",".art"}
        drill_ext={".drl",".xln",".exc",".drd"}
        # Legacy Altium output can use arbitrary .GMnn for Board Shape.
        # Read the exporter's layer-extension report instead of hardcoding GM14.
        extension_roles={}
        for report in files:
            if report.suffix.lower() != ".extrep": continue
            text=report.read_text(encoding="latin-1",errors="replace")
            for line in text.splitlines():
                match=re.match(r"\s*(\.[A-Za-z][A-Za-z0-9]+)\s+(.+?)\s*$",line)
                if match: extension_roles[match.group(1).lower()]=match.group(2).strip().lower()

        # Altium often emits both a binary EIA .DRL and an ASCII round-hole .TXT
        # with the same stem.  Prefer the ASCII file because it retains tool sizes
        # and plated/non-plated grouping in a form we can inspect reliably.
        ascii_drill_txt={}
        for p in files:
            if p.suffix.lower() != ".txt": continue
            text=self._read_header(p)
            if text.startswith("M48") and ("FILE_FORMAT" in text or re.search(r"T\d+.*C[0-9.]",text)):
                ascii_drill_txt[p.stem.lower()]=p

        for p in files:
            ext=p.suffix.lower(); n=p.name.lower()
            if ext==".gbrjob" and out["job"] is None:
                out["job"]=p; continue
            if ext in gerber_ext or ext in extension_roles:
                text=self._read_header(p)
                role=extension_roles.get(ext,"")
                if any(word in role for word in ("board shape","board outline","board profile","board edge")):
                    if out["profile"] is None: out["profile"]=p
                    continue
                if self._looks_like_top_cu(p,text) and out["top_cu"] is None: out["top_cu"]=p
                elif self._looks_like_bottom_cu(p,text) and out["bottom_cu"] is None: out["bottom_cu"]=p
                elif (("TF.FILEFUNCTION,SOLDERMASK" in text and ",TOP" in text) or ext==".gts" or
                      "f_mask" in n or "mskgtp" in n or "topmask" in n or "top_mask" in n or
                      "soldermask_top" in n or "soldermask-top" in n):
                    if out["top_mask"] is None: out["top_mask"]=p
                elif (("TF.FILEFUNCTION,SOLDERMASK" in text and (",BOT" in text or ",BOTTOM" in text)) or ext==".gbs" or
                      "b_mask" in n or "mskgbt" in n or "bottommask" in n or "bottom_mask" in n or
                      "soldermask_bot" in n or "soldermask_bottom" in n or "soldermask-bot" in n):
                    if out["bottom_mask"] is None: out["bottom_mask"]=p
                elif ("TF.FILEFUNCTION,PROFILE" in text or "edge_cuts" in n or "outline" in n or
                      "border_cor" in n or "board_cor" in n or n.startswith("border") or n.startswith("board_cor") or "profile" in n):
                    if out["profile"] is None: out["profile"]=p
                continue
            if ext in drill_ext or ext==".txt":
                # Skip Altium binary EIA DRL when the matching ASCII TXT exists.
                if ext==".drl" and p.stem.lower() in ascii_drill_txt:
                    continue
                text=self._read_header(p); upname=p.name.upper()
                if ext==".txt" and not (text.startswith("M48") or "TF.FILEFUNCTION" in text or re.search(r"\bT\d+.*C[0-9.]",text)):
                    continue
                mixed=("TYPE=PLATED" in text and ("TYPE=NON_PLATED" in text or "TYPE=NON-PLATED" in text))
                nonplated=(not mixed and ("FILEFUNCTION,NONPLATED" in text or "NPTH" in upname or ("NON" in upname and "PLAT" in upname)))
                plated=(not mixed and ("FILEFUNCTION,PLATED" in text or "PLATED,PTH" in text or
                                     ("PLATED" in upname and not nonplated) or ("PTH" in upname and not nonplated)))
                if mixed: out["drill_mixed"].append(p)
                elif nonplated: out["npth"].append(p)
                elif plated: out["pth"].append(p)
                else: out["drill_unknown"].append(p)
        return out

    def _gerber_requirements(self):
        b=self.gerber_bundle or {}; side=self.gerber_side_var.get()
        primary=b.get("top_cu") if side=="Top" else b.get("bottom_cu")
        opposite=b.get("bottom_cu") if side=="Top" else b.get("top_cu")
        drills=list(b.get("pth",[]))+list(b.get("npth",[]))+list(b.get("drill_mixed",[]))+list(b.get("drill_unknown",[]))
        missing=[]
        if primary is None: missing.append("Top Cu" if side=="Top" else "Bottom Cu")
        if not drills: missing.append("Excellon drill")
        return primary, opposite, drills, missing

    def _scan_gerber_folder(self):
        folder=self.gerber_dir_var.get().strip()
        # If the operator points the app at another Gerber package, start from Top.
        # Keep the current side when this function is merely re-run for the same folder
        # (for example after choosing Bottom).
        try:
            folder_key=str(Path(folder).resolve()) if folder else ""
        except Exception:
            folder_key=folder
        if folder_key and self._last_gerber_folder and folder_key != self._last_gerber_folder:
            self.gerber_side_var.set("Top")
        if folder_key:
            self._last_gerber_folder=folder_key
        self.gerber_bundle=self._detect_gerber_bundle(folder) if folder else {}
        primary,opposite,drills,missing=self._gerber_requirements()
        b=self.gerber_bundle
        if not folder:
            text=self.tr("choose_gerber"); color="#666666"
        elif not Path(folder).is_dir():
            text=self.tr("folder_not_found"); color="#b00020"
        elif missing:
            text=self.tr("missing",items=", ".join(missing)); color="#b00020"
        else:
            parts=[]
            parts.append("Top Cu" if b.get("top_cu") else "")
            parts.append("Bottom Cu" if b.get("bottom_cu") else "")
            if b.get("pth"): parts.append(f"PTH drill ×{len(b['pth'])}")
            if b.get("npth"): parts.append(f"NPTH drill ×{len(b['npth'])}")
            if b.get("drill_mixed"): parts.append(f"PTH+NPTH drill ×{len(b['drill_mixed'])}")
            if b.get("drill_unknown"): parts.append(f"drill ×{len(b['drill_unknown'])}")
            text=self.tr("gerber_ready",items=" | ".join(x for x in parts if x))
            # Fully classified drill set gets green; optional omissions are shown as warnings.
            color="#138a2e" if (b.get("pth") or b.get("npth") or b.get("drill_mixed")) else "#a06000"
            side=self.gerber_side_var.get(); mask=b.get("top_mask") if side=="Top" else b.get("bottom_mask")
            if mask:
                text += f" | {side} mask"
            else:
                text += "\n" + self.tr("mask_missing",side=side)
                color="#a06000"
            if not opposite:
                text += "\n" + self.tr("opposite_missing")
                color="#a06000"
        try:self.gerber_ready_label.configure(fg=color)
        except Exception:pass
        self.gerber_bundle_status.set(text)
        self.result=None
        if folder and Path(folder).is_dir():
            self.status_var.set(text.replace("\n"," "))
        return primary,opposite,drills,missing

    def pick_gerber_folder(self):
        initial=self.gerber_dir_var.get().strip() or str(Path(self.scan_path.get()).parent if self.scan_path.get() else Path.cwd())
        p=filedialog.askdirectory(title=self.tr("gerber_folder_dialog"),initialdir=initial)
        if not p:return
        # A newly selected Gerber set always opens on the component/top side.
        # This avoids carrying a previous Bottom selection into a different board.
        self.gerber_side_var.set("Top")
        self.gerber_dir_var.set(p); self._scan_gerber_folder()

    def _gerber_side_changed(self):
        self.result=None; self._scan_gerber_folder(); self.status_var.set(self.tr("gerber_side_status",side=self.gerber_side_var.get()))

    def _view_changed(self,_event=None):
        if self.view_var.get()=="Anchors": self.anchor_tools.pack(side="top",fill="x",before=self.viewer.master)
        else:
            self.edit_anchors_var.set(False);self._selected_anchor=None;self.anchor_tools.pack_forget()
        self.refresh_view()
    def _set_progress(self,pct,text):
        pct=max(0,min(100,int(pct)))
        self.progress["value"]=pct; self.progress_pct_var.set(f"{pct}%")
        raw=str(text)
        semantic=None
        for key in ("waiting","starting","error","ready","apply_correction","resetting"):
            if raw in (I18N["ru"].get(key),I18N["en"].get(key)):
                semantic=key; break
        self._progress_state=("key",semantic) if semantic else ("raw",raw)
        self._refresh_progress_language()
    def _set_busy(self,value):
        self.busy=bool(value);self.analysis_btn.state(["disabled"] if self.busy else ["!disabled"]); self.lang_combo.configure(state="disabled" if self.busy else "readonly")

    def start_analysis(self):
        if self.busy:return
        if not self.scan_path.get() or not Path(self.scan_path.get()).exists():
            messagebox.showerror(APP_TITLE,self.tr("select_scan"));return
        if self.source_mode.get()=="CAM":
            if not self.cam_path.get() or not Path(self.cam_path.get()).exists():messagebox.showerror(APP_TITLE,self.tr("select_cam"));return
        else:
            primary,opposite,ds,missing=self._scan_gerber_folder()
            if missing:
                messagebox.showerror(APP_TITLE,self.tr("gerber_incomplete")+":\n• " + "\n• ".join(missing));return
        self._set_busy(True);self._set_progress(0,self.tr("starting"));self.status_var.set(self.tr("processing"))
        page=max(0,int(self.page_var.get())-1);threading.Thread(target=self._analysis_worker,args=(page,),daemon=True).start()

    def _analysis_worker(self,page):
        try:
            def progress(pct,text):self._worker_queue.put(("progress",int(pct),str(text)))
            if self.source_mode.get()=="CAM":
                result=analyze_camvu(self.scan_path.get(),self.cam_path.get(),self.cam_layer_var.get() or None,pdf_page=page,pdf_dpi=PDF_DPI,registration_mode=REGISTRATION_MODE,hole_tolerance_mm=float(self.hole_tol_var.get()),progress_cb=progress)
            else:
                primary,opposite,drills,missing=self._gerber_requirements()
                if missing: raise RuntimeError(self.tr("gerber_incomplete") + ": " + ", ".join(missing))
                b=self.gerber_bundle or {}; side=self.gerber_side_var.get()
                mask=b.get("top_mask") if side=="Top" else b.get("bottom_mask")
                result=analyze_gerber(self.scan_path.get(),str(primary),[str(p) for p in drills],pdf_page=page,pdf_dpi=PDF_DPI,registration_mode=REGISTRATION_MODE,hole_tolerance_mm=float(self.hole_tol_var.get()),progress_cb=progress,opposite_gerber_path=(str(opposite) if opposite else None),solder_mask_gerber_path=(str(mask) if mask else None),profile_gerber_path=(str(b.get("profile")) if b.get("profile") else None))
                result["gerber_folder"]=self.gerber_dir_var.get().strip(); result["gerber_side"]=self.gerber_side_var.get()
            self._worker_queue.put(("analysis_done",result,None))
        except Exception as e:self._worker_queue.put(("analysis_done",None,(e,traceback.format_exc())))

    def _refit_worker(self,operation):
        try:
            def progress(pct,text):self._worker_queue.put(("progress",int(pct),str(text)))
            if operation=="apply":
                self.result["control_dst"]=self.pending_control_dst.copy();refit_result(self.result,registration_mode=REGISTRATION_MODE,recompute_copper=False,progress_cb=progress)
            else:reset_manual_anchors(self.result,registration_mode=REGISTRATION_MODE,progress_cb=progress)
            self._worker_queue.put(("refit_done",operation,None))
        except Exception as e:self._worker_queue.put(("refit_done",operation,(e,traceback.format_exc())))

    def _poll_worker_queue(self):
        try:
            while True:
                msg=self._worker_queue.get_nowait();kind=msg[0]
                if kind=="progress":self._set_progress(msg[1],msg[2])
                elif kind=="analysis_done":self._analysis_done(msg[1],msg[2])
                elif kind=="refit_done":self._refit_done(msg[1],msg[2])
        except queue.Empty:pass
        self.after(80,self._poll_worker_queue)

    def _analysis_done(self,result,error):
        self._set_busy(False)
        if error:
            e,tb=error;self._set_progress(0,self.tr("error"));messagebox.showerror(APP_TITLE,f"{e}\n\n{tb[-2600:]}");return
        self.result=result;self.pending_control_dst=result.get("control_dst").copy();self.metrics_var.set(self._metrics_text());self.legend_var.set(self._legend_text())
        self._set_progress(100,self.tr("ready"));self.status_var.set(self.tr("done_status"));self.refresh_view(fit=True)

    def _metrics_text(self):
        if self.result is None:return self.tr("not_registered")
        r=self.result.get("report",{});source=self.result.get("source_type","camvu")
        title=(f"CAM: {r.get('cam_layer_name','—')}" if source=="camvu" else f"Gerber: {r.get('layer_name',Path(self.result.get('gerber_path','')).name)}")
        side=self.tr("mirrored") if r.get("mirrored") else self.tr("normal");dia=r.get("primary_anchor_diameter_mm");dia_s="—" if dia is None else f"Ø{dia} mm"
        hc=(self.result.get("hole_check") or {}).get("counts",{});groups=(self.result.get("hole_check") or {}).get("groups",[])
        lines=[]
        for g in groups:
            cat={"PTH_OPEN":"PTH","VIA_TENTED":"VIA","NPTH":"NPTH"}.get(g.get("category"),str(g.get("category")))
            vis=int(g.get("visible_expected",g.get("expected",0)));found=int(g.get("found",0));out=int(g.get("OUTSIDE",0))
            warn=int(g.get("WARN",0));shift=int(g.get("SHIFT",0));unk=int(g.get("UNKNOWN",0))
            tail="OK" if warn+shift+unk==0 and found==vis else f"W{warn} S{shift} ?{unk}"
            lines.append(f"{cat} Ø{float(g.get('diameter_mm',0)):.2f}: {found}/{vis} {tail}"+(f" +{out} OUT" if out else ""))
        ang=r.get("orientation_angle_deg"); ang_s="—" if ang is None else f"{float(ang):.1f}°"
        checked=int(r.get("orientation_candidates_checked",0) or 0)
        ori=f"{side}, {self.tr('turn')} {ang_s}" + (f" ({self.tr('checked',count=checked)})" if checked>1 else "")
        prof_s=("\n"+self.tr("profile_orientation") if r.get("profile_orientation_used") else "")
        cov=r.get("copper_coverage_pct")
        cov_s=("" if cov is None else "\n"+self.tr("copper_area",pct=float(cov)))
        auto_warn=r.get("copper_evidence_warning")
        warn_s=("" if not auto_warn else "\n"+self.tr("copper_skipped"))
        return (f"{title}\n{self.tr('orientation')}: {ori}\n{self.tr('primary_anchors')}: {dia_s}\n{self.tr('matched_anchors')}: {r.get('matched_anchors','—')}\n"
                f"{self.tr('scale')}: {r.get('px_per_mm','—')} px/mm{prof_s}{cov_s}{warn_s}\n{self.tr('holes_summary')}: OK {hc.get('OK',0)} | WARN {hc.get('WARN',0)} | SHIFT {hc.get('SHIFT',0)} | ? {hc.get('UNKNOWN',0)} | OUT {hc.get('OUTSIDE',0)}"
                +("\n\n"+"\n".join(lines) if lines else ""))

    def _make_current_view(self):
        if self.result is None:return None
        R=self.result;crop=R["crop"];scan=crop_result(R["scan_full"],crop);mode=self.view_var.get()
        if mode=="Scan":return scan.copy()
        if mode=="Overlay":
            overlay_mask=R.get("gerber_overlay_full",R["gerber_full"])
            full=gerber_overlay(R["scan_full"],overlay_mask,alpha=float(self.alpha_var.get())/100.0)
            if self.show_outline_var.get():
                mm=(overlay_mask>32).astype(np.uint8)*255;edge=cv2.morphologyEx(mm,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8));full[edge>0]=(0,0,255)
            if self.show_holes_var.get() and R.get("hole_check") is not None:full=hole_overlay(full,R["hole_check"])
            return crop_result(full,crop)
        if mode=="Anchors":
            full=anchor_overlay(R["scan_full"],R["scan_pts"],R["drill_xy"],R["H"],R["mirror"],R.get("control_src"),self.pending_control_dst if self.pending_control_dst is not None else R.get("control_dst"))
            return crop_result(full,crop)
        return scan.copy()

    def refresh_view(self,fit=False):
        if self.result is None or self.busy:return
        try:self.viewer.set_image(self._make_current_view(),fit=fit)
        except Exception as e:self.status_var.set(self.tr("display_error",error=e))

    def _hole_tolerance_changed(self,_event=None):
        if self.result is None or self.result.get("hole_check") is None:return
        try:tol=float(self.hole_tol_var.get())
        except Exception:return
        reclassify_hole_check(self.result["hole_check"],tol);self.result["hole_tolerance_mm"]=tol;self.metrics_var.set(self._metrics_text());self.refresh_view()

    def toggle_anchor_edit(self):
        if self.result is None:self.edit_anchors_var.set(False);return
        self._selected_anchor=None;self.status_var.set(self.tr("anchor_on") if self.edit_anchors_var.get() else self.tr("anchor_off"))
    def _anchor_press(self,p_crop,zoom):
        if self.busy or not self.edit_anchors_var.get() or self.result is None or self.view_var.get()!="Anchors":return False
        x0,y0,_,_=self.result["crop"];p_full=p_crop+np.array([x0,y0],np.float64);dst=self.pending_control_dst
        if dst is None or len(dst)==0:return False
        radius=18.0/max(0.08,zoom);d=np.linalg.norm(dst-p_full[None,:],axis=1);idx=int(np.argmin(d))
        if d[idx]>radius:return False
        self._selected_anchor=idx;self._drag_anchor_old=dst[idx].copy();return {"handled":True,"point":dst[idx]-np.array([x0,y0],np.float64)}
    def _anchor_move(self,p_crop):
        if self._selected_anchor is None or self.result is None:return
        x0,y0,_,_=self.result["crop"];p_full=p_crop+np.array([x0,y0],np.float64);delta=p_full-self._drag_anchor_old
        self.status_var.set(self.tr("anchor_drag",idx=self._selected_anchor+1,dx=delta[0],dy=delta[1]))
    def _anchor_release(self,p_crop):
        if self._selected_anchor is None or self.result is None:return
        x0,y0,_,_=self.result["crop"];p_full=p_crop+np.array([x0,y0],np.float64);idx=self._selected_anchor;self._selected_anchor=None
        if self.pending_control_dst is not None:self.pending_control_dst[idx]=p_full
        self.status_var.set(self.tr("anchor_changed",idx=idx+1));self.refresh_view()
    def apply_edits(self):
        if self.busy or self.result is None or self.pending_control_dst is None:return
        self._set_busy(True);self._set_progress(0,self.tr("apply_correction"));threading.Thread(target=self._refit_worker,args=("apply",),daemon=True).start()
    def cancel_edits(self):
        if self.busy or self.result is None:return
        self.pending_control_dst=self.result.get("control_dst").copy();self.status_var.set(self.tr("changes_cancelled"));self.refresh_view()
    def reset_anchors(self):
        if self.busy or self.result is None:return
        self._set_busy(True);self._set_progress(0,self.tr("resetting"));threading.Thread(target=self._refit_worker,args=("reset",),daemon=True).start()
    def _refit_done(self,operation,error):
        self._set_busy(False)
        if error:
            e,tb=error;self._set_progress(0,self.tr("error"));messagebox.showerror(APP_TITLE,f"{e}\n\n{tb[-2200:]}");return
        self.pending_control_dst=self.result["control_dst"].copy();self.metrics_var.set(self._metrics_text());self._set_progress(100,self.tr("ready"));self.status_var.set(self.tr("correction_applied") if operation=="apply" else self.tr("auto_restored"));self.refresh_view()

    def export_view(self):
        img=self._make_current_view()
        if img is None:messagebox.showinfo(APP_TITLE,self.tr("analyze_first"));return
        p=filedialog.asksaveasfilename(title=self.tr("save_view"),defaultextension=".png",filetypes=[("PNG","*.png"),("JPEG","*.jpg *.jpeg")])
        if not p:return
        try:
            ext=Path(p).suffix.lower() or ".png";ok,buf=cv2.imencode(ext,img)
            if not ok:raise RuntimeError(self.tr("encode_failed"))
            Path(p).write_bytes(buf.tobytes());self.status_var.set(self.tr("saved",path=p))
        except Exception as e:messagebox.showerror(APP_TITLE,str(e))


def main():
    app=PCBApp(); app.mainloop()

if __name__ == "__main__":
    main()
