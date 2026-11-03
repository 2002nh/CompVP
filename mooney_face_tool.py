"""Mooney Face Tool — Task 2 (desktop GUI, no GPU required).

INSTALL: python -m pip install opencv-python numpy pillow
RUN:     python mooney_face_tool.py
Ubuntu system Python may also need: sudo apt install python3-tk

1. Open an image or a folder of extracted images (not a ZIP).
2. Adjust Gaussian blur sigma and target white-pixel percentage.
   Changes are remembered per image during this session.
   'Apply settings to all' sets the current values for every loaded image.
3. Click 'Batch convert all' and choose a destination parent folder.
   A NEW run folder is created; source images are never overwritten.
4. Review the images and mark your own top 10. Export them after conversion.

Output: Original/*.png, Mooney/*.png, settings.csv, and (after selection)
Subset/Original/*.png + Subset/Mooney/*.png + selection.csv.
Original exports preserve the photograph's RGB pixels at its original size,
with EXIF orientation applied. Matching filenames are assigned in list order.

White percentage is a target: equal-brightness pixels can prevent an exact
match. The actual percentage is displayed and recorded. No artificial noise
or tie-breaking patterns are added. A constant image cannot yield meaningful
Mooney structure. 'Good' images and the top 10 still require your judgement.
"""

from pathlib import Path
from datetime import datetime
import csv
import math
import queue
import shutil
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps

from rembg import remove


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_rgb(path):
    """Read one photograph, apply its orientation, and release the file handle."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
    """Some photographs—especially phone photos—store instructions about their orientation 
    in EXIF metadata. This reads those instructions and rotates or flips the pixels when necessary,
      so the photograph appears correctly."""


def make_mooney(image, blur_sigma=3.0, white_percent=50.0):
    """Return (blurred_float_array, binary_uint8_array, threshold, actual_white%)."""
    if not math.isfinite(blur_sigma) or blur_sigma < 0:
        raise ValueError("Blur sigma must be a finite number >= 0.")
    if not math.isfinite(white_percent) or not 0 <= white_percent <= 100:
        raise ValueError("White percentage must be between 0 and 100.")


    # STEP 1: Convert RGB to greyscale; float pixels avoid blur quantization.
    grey = np.asarray(image.convert("L"), dtype=np.float32)

    # STEP 2: Apply Gaussian blur. Sigma=0 explicitly means no blur.
    blurred = grey if blur_sigma == 0 else cv2.GaussianBlur(
        grey, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma
    )

    # STEP 3: Compute the brightness threshold from the requested white ratio.
    # For 40% white, approximately the brightest 40% should become white.
    if white_percent == 0:
        threshold = float(blurred.max())
        binary = np.zeros(grey.shape, dtype=np.uint8)
    elif white_percent == 100:
        threshold = float(blurred.min()) - 1.0
        binary = np.full(grey.shape, 255, dtype=np.uint8)
    else:
        threshold = float(np.percentile(blurred, 100.0 - white_percent))
        binary = np.where(blurred > threshold, 255, 0).astype(np.uint8)

    actual = float(np.count_nonzero(binary)) / binary.size * 100.0
    return blurred, binary, threshold, actual


def batch_convert(paths, destination, defaults=(3.0, 50.0), overrides=None, progress=None):
    """Process arbitrary-size batches, one image at a time, into a NEW directory.

    overrides maps string source paths to (sigma, white percentage).
    Returns (successful-file records, error records). Does not skip failures silently.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("No input images supplied.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "Original").mkdir()
    (destination / "Mooney").mkdir()
    overrides = overrides or {}
    saved, errors = {}, []
    columns = ["image_id", "source_file", "blur_sigma", "target_white_percent",
               "actual_white_percent", "threshold", "status", "error"]
    with (destination / "settings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i, path in enumerate(paths, 1):
            path = Path(path)
            key = str(path)
            image_id = f"{i:03d}.png"
            sigma, white = overrides.get(key, defaults)
            row = dict(image_id=image_id, source_file=key, blur_sigma=sigma,
                       target_white_percent=white)
            original_path = destination / "Original" / image_id
            mooney_path = destination / "Mooney" / image_id
            try:
                rgb = load_rgb(path)
                _, binary, threshold, actual = make_mooney(rgb, sigma, white)
                rgb.save(original_path)
                Image.fromarray(binary).save(mooney_path)
                row.update(actual_white_percent=actual, threshold=threshold, status="ok", error="")
                saved[key] = dict(name=image_id, settings=(sigma, white),
                                  original=original_path, mooney=mooney_path)
            except Exception as exc:
                original_path.unlink(missing_ok=True)
                mooney_path.unlink(missing_ok=True)
                row.update(status="error", error=str(exc))
                errors.append((key, str(exc)))
            writer.writerow(row)
            stream.flush()
            if progress:
                progress(i, len(paths))
    return saved, errors


def export_subset(selected, saved, destination):
    """Copy exactly 10 manually chosen, successfully converted pairs."""
    selected = list(selected)
    if len(selected) != 10 or len(set(selected)) != 10:
        raise ValueError("Select exactly 10 images manually before exporting.")
    if any(key not in saved for key in selected):
        raise ValueError("Some selected images were not converted successfully.")
    subset = Path(destination) / "Subset"
    # A previous selection is never silently replaced.
    subset.mkdir(exist_ok=False)
    (subset / "Original").mkdir()
    (subset / "Mooney").mkdir()
    with (subset / "selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "source_file"])
        for key in selected:
            rec = saved[key]
            shutil.copy2(rec["original"], subset / "Original" / rec["name"])
            shutil.copy2(rec["mooney"], subset / "Mooney" / rec["name"])
            writer.writerow([rec["name"], key])
    return subset


def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import ImageTk

    class MooneyApp:
        def __init__(self, root):
            self.root = root
            root.title("Mooney Face Tool — Gaussian blur + white-pixel control")
            root.geometry("1180x760")
            root.minsize(980, 650)
            self.paths, self.overrides, self.saved, self.top10 = [], {}, {}, set()
            self.defaults = (3.0, 50.0)
            self.index, self.rgb, self.pending = None, None, None
            self.destination, self.busy, self.loading = None, False, False
            self.events = queue.Queue()
            self.widgets = []
            self.sigma = tk.DoubleVar(value=3.0)
            self.white = tk.DoubleVar(value=50.0)
            self.marked = tk.BooleanVar(value=False)

            toolbar = ttk.Frame(root, padding=10)
            toolbar.pack(fill="x")
            for label, action in [("Open image(s)", self.open_images),
                                  ("Open folder", self.open_folder),
                                  ("Batch convert all", self.start_batch),
                                  ("Export selected top 10", self.save_top10)]:
                button = ttk.Button(toolbar, text=label, command=action)
                button.pack(side="left", padx=4)
                self.widgets.append(button)
            self.count_label = ttk.Label(toolbar, text="0 images loaded")
            self.count_label.pack(side="right")

            body = ttk.Frame(root, padding=(10, 0))
            body.pack(fill="both", expand=True)
            sidebar = ttk.Frame(body)
            sidebar.pack(side="left", fill="y", padx=(0, 10))
            ttk.Label(sidebar, text="Browse / manually choose top 10").pack(anchor="w")
            self.files = tk.Listbox(sidebar, width=30, exportselection=False)
            self.widgets.append(self.files)
            self.files.pack(side="left", fill="both", expand=True)
            scrollbar = ttk.Scrollbar(sidebar, command=self.files.yview)
            scrollbar.pack(side="right", fill="y")
            self.files.configure(yscrollcommand=scrollbar.set)
            self.files.bind("<<ListboxSelect>>", self.on_select)

            main = ttk.Frame(body)
            main.pack(side="left", fill="both", expand=True)
            controls = ttk.LabelFrame(main, text="Conversion settings", padding=10)
            controls.pack(fill="x")
            for label, variable, upper, step in [
                ("Gaussian blur sigma (pixels)", self.sigma, 20, 0.25),
                ("Target white pixels (%)", self.white, 100, 1)]:
                ttk.Label(controls, text=label).pack(anchor="w")
                slider = tk.Scale(controls, from_=0, to=upper, resolution=step,
                                  orient="horizontal", variable=variable,
                                  command=self.settings_changed)
                slider.pack(fill="x")
                self.widgets.append(slider)
            button = ttk.Button(controls, text="Apply settings to all images", command=self.apply_all)
            button.pack(anchor="w", pady=4)
            self.widgets.append(button)
            ttk.Label(controls, text="Slider changes are remembered for the current image during this session.").pack(anchor="w")

            previews = ttk.Frame(main)
            previews.pack(fill="both", expand=True, pady=10)
            self.preview_labels = []
            for col, title in enumerate(["Original", "Gaussian blur", "Mooney (black / white)"]):
                previews.columnconfigure(col, weight=1, uniform="preview")
                pane = ttk.LabelFrame(previews, text=title, padding=4)
                pane.grid(row=0, column=col, sticky="nsew", padx=3)
                label = ttk.Label(pane, anchor="center")
                label.pack(fill="both", expand=True)
                self.preview_labels.append(label)
            previews.rowconfigure(0, weight=1)
            self.info = ttk.Label(main, text="Open a photograph or a folder of images.", wraplength=800)
            self.info.pack(fill="x", pady=4)
            check = ttk.Checkbutton(main, text="Mark this image for my top 10", variable=self.marked,
                                    command=self.toggle_top10)
            check.pack(anchor="w", pady=5)
            self.widgets.append(check)
            self.selection_label = ttk.Label(main, text="Selected: 0 / 10")
            self.selection_label.pack(anchor="w")
            self.bar = ttk.Progressbar(root)
            self.bar.pack(fill="x", padx=14, pady=(8, 0))
            self.status = ttk.Label(root, text="No GPU needed. Use PNG outputs for pure black and white.", padding=10)
            self.status.pack(fill="x")
            root.protocol("WM_DELETE_WINDOW", self.close)
            root.after(100, self.poll)

        def set_paths(self, paths):
            if not paths:
                return
            self.paths = sorted({Path(p).resolve() for p in paths})
            self.overrides, self.saved, self.top10 = {}, {}, set()
            self.destination, self.index, self.rgb = None, None, None
            self.files.delete(0, "end")
            for i, path in enumerate(self.paths, 1):
                self.files.insert("end", f"{i:03d}  {path.name}")
            self.count_label.config(text=f"{len(self.paths)} images loaded")
            self.selection_label.config(text="Selected: 0 / 10")
            self.files.selection_set(0)
            self.on_select()

        def open_images(self):
            paths = filedialog.askopenfilenames(title="Select photographs", filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
            self.set_paths(paths)

        def open_folder(self):
            folder = filedialog.askdirectory(title="Folder containing your extracted images")
            if folder:
                paths = [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED]
                if not paths:
                    messagebox.showinfo("No images", "No supported images in this folder. Open the folder containing the actual image files.")
                self.set_paths(paths)

        def on_select(self, event=None):
            if self.busy or not self.files.curselection():
                return
            self.index = self.files.curselection()[0]
            path = self.paths[self.index]
            self.loading = True
            sigma, white = self.overrides.get(str(path), self.defaults)
            self.sigma.set(sigma)
            self.white.set(white)
            self.marked.set(str(path) in self.top10)
            try:
                self.rgb = load_rgb(path)
            except Exception as exc:
                self.rgb = None
                self.info.config(text=f"Cannot read {path.name}: {exc}")
                for label in self.preview_labels:
                    label.config(image="", text="Cannot read image")
            self.loading = False
            self.render()

        def current_settings(self):
            return (float(self.sigma.get()), float(self.white.get()))

        def settings_changed(self, value=None):
            if self.loading or self.index is None or self.busy:
                return
            self.overrides[str(self.paths[self.index])] = self.current_settings()
            if self.pending:
                self.root.after_cancel(self.pending)
            self.pending = self.root.after(100, self.render)

        def apply_all(self):
            if not self.paths:
                return
            self.defaults = self.current_settings()
            self.overrides.clear()
            self.render()
            self.status.config(text="Current settings applied to all loaded images. Batch convert to save them.")

        def render(self):
            self.pending = None
            if self.rgb is None:
                return
            blurred, binary, threshold, actual = make_mooney(self.rgb, *self.current_settings())
            images = [self.rgb, Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8)),
                      Image.fromarray(binary)]
            width = max(160, min(300, (self.root.winfo_width() - 330) // 3))
            for i, (label, image) in enumerate(zip(self.preview_labels, images)):
                display = image.copy()
                display.thumbnail((width, 320), Image.Resampling.NEAREST if i == 2 else Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display)
                label.configure(image=photo, text="")
                label.image = photo
            note = " Equal-brightness pixels prevent a closer match." if abs(actual - self.white.get()) > 1 else ""
            path = self.paths[self.index]
            self.info.config(text=f"{self.index + 1}/{len(self.paths)} — {path.name} | "
                                  f"Threshold: {threshold:.2f} | Actual white: {actual:.2f}%.{note}")

        def toggle_top10(self):
            if self.index is None:
                self.marked.set(False)
                return
            key = str(self.paths[self.index])
            if self.marked.get():
                if len(self.top10) == 10:
                    self.marked.set(False)
                    messagebox.showinfo("Top 10", "Unmark another image before selecting a replacement.")
                    return
                self.top10.add(key)
            else:
                self.top10.discard(key)
            self.selection_label.config(text=f"Selected: {len(self.top10)} / 10")
            self.files.delete(self.index)
            prefix = "★ " if key in self.top10 else ""
            self.files.insert(self.index, f"{prefix}{self.index + 1:03d}  {self.paths[self.index].name}")
            self.files.selection_set(self.index)

        def start_batch(self):
            if not self.paths:
                messagebox.showinfo("No images", "Open an image folder first.")
                return
            parent = filedialog.askdirectory(title="Choose a parent folder for a NEW output run")
            if not parent:
                return
            destination = Path(parent) / datetime.now().strftime("Mooney_Run_%Y%m%d_%H%M%S_%f")
            self.busy = True
            for widget in self.widgets:
                widget.configure(state="disabled")
            self.bar.configure(maximum=len(self.paths), value=0)
            paths, defaults, overrides = list(self.paths), self.defaults, dict(self.overrides)
            def work():
                try:
                    result = batch_convert(paths, destination, defaults, overrides,
                                           lambda i, n: self.events.put(("progress", i, n)))
                    self.events.put(("done", destination, result))
                except Exception as exc:
                    self.events.put(("error", str(exc)))
            threading.Thread(target=work, daemon=True).start()

        def poll(self):
            try:
                while True:
                    event = self.events.get_nowait()
                    if event[0] == "progress":
                        self.bar.configure(value=event[1])
                        self.status.config(text=f"Converting {event[1]} / {event[2]} images…")
                    else:
                        self.busy = False
                        for widget in self.widgets:
                            widget.configure(state="normal")
                        if event[0] == "done":
                            self.destination = event[1]
                            self.saved, errors = event[2]
                            self.status.config(text=f"Saved {len(self.saved)} pairs; {len(errors)} failures. {self.destination}")
                            messagebox.showinfo("Conversion complete", f"Saved {len(self.saved)} pairs.\nFailures: {len(errors)}\n"
                                                f"See settings.csv for details.\n\n{self.destination}\n\nReview and mark your top 10 manually.")
                        else:
                            self.status.config(text="Batch conversion failed.")
                            messagebox.showerror("Batch error", event[1])
            except queue.Empty:
                pass
            self.root.after(100, self.poll)

        def save_top10(self):
            if self.destination is None:
                messagebox.showinfo("Convert first", "Batch convert your images before exporting the subset.")
                return
            for key in self.top10:
                if key in self.saved and self.saved[key]["settings"] != self.overrides.get(key, self.defaults):
                    messagebox.showinfo("Settings changed", "You changed a selected image since conversion. Batch convert again first.")
                    return
            try:
                selected = [str(p) for p in self.paths if str(p) in self.top10]
                subset = export_subset(selected, self.saved, self.destination)
                messagebox.showinfo("Top 10 saved", f"Saved 10 original/Mooney pairs:\n{subset}")
            except FileExistsError:
                messagebox.showinfo("Subset already exists", "This run already contains an exported subset. Use a new batch run for a new selection.")
            except Exception as exc:
                messagebox.showerror("Cannot export", str(exc))

        def close(self):
            if self.busy:
                messagebox.showinfo("Conversion running", "Wait for the batch conversion to finish before closing.")
            else:
                self.root.destroy()

    root = tk.Tk()
    MooneyApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
