"""
IMAGE PROCESSING DESKTOP APPLICATION
Team Project – Python + OpenCV + Tkinter
HOW TO RUN:
  1. pip install opencv-python pillow
  2. python Final.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


# ── Helper: OpenCV image → Tkinter PhotoImage ─────────────────────────────────
def cv2_to_tk(img, max_w=520, max_h=420):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if len(img.shape) == 2 else cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    pil.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil)


# ── Main App ──────────────────────────────────────────────────────────────────
class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1200x780")
        self.root.minsize(900, 650)
        self.original_img = self.current_img = None
        self.tk_img_left  = self.tk_img_right = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title
        tf = tk.Frame(self.root, bg="#16213e", pady=8)
        tf.pack(fill="x")
        tk.Label(tf, text="🖼  Image Processing Studio",
                font=("Courier New", 18, "bold"), fg="#e94560", bg="#16213e").pack()

        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill="both", expand=True, padx=10, pady=6)
        self._build_left_panel(main)
        self._build_right_panel(main)

        # Status bar
        self.status_var = tk.StringVar(value="Load an image to begin.")
        tk.Label(self.root, textvariable=self.status_var,
                font=("Courier New", 10), fg="#a0a0b0", bg="#0f3460",
                anchor="w", padx=12, pady=4).pack(fill="x", side="bottom")

    def _build_left_panel(self, parent):
        container = tk.Frame(parent, bg="#16213e", width=230)
        container.pack(side="left", fill="y")
        container.pack_propagate(False)

        cs = tk.Canvas(container, bg="#16213e", width=220, highlightthickness=0)
        sb = ttk.Scrollbar(container, orient="vertical", command=cs.yview)
        cs.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cs.pack(side="left", fill="both", expand=True)

        panel = tk.Frame(cs, bg="#16213e", padx=10, pady=10)
        pw = cs.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda e: cs.configure(scrollregion=cs.bbox("all")))
        cs.bind("<Configure>",    lambda e: cs.itemconfig(pw, width=e.width))
        cs.bind_all("<MouseWheel>", lambda e: cs.yview_scroll(int(-1*(e.delta/120)), "units"))

        BTN = dict(font=("Courier New", 10, "bold"), bg="#0f3460", fg="#e0e0e0",
                activebackground="#e94560", activeforeground="white",
                relief="flat", bd=0, padx=8, pady=6, cursor="hand2")
        HDR = dict(font=("Courier New", 9, "bold"), fg="#e94560", bg="#16213e", anchor="w")

        def section(text):
            tk.Label(panel, text=text, **HDR).pack(fill="x", pady=(10, 2))
            tk.Frame(panel, bg="#e94560", height=1).pack(fill="x")

        def btn(text, cmd):
            b = tk.Button(panel, text=text, command=cmd, **BTN)
            b.pack(fill="x", pady=2)
            b.bind("<Enter>", lambda e: b.config(bg="#e94560"))
            b.bind("<Leave>", lambda e: b.config(bg="#0f3460"))

        def slider(var, from_, to, default, label_w, update_fn):
            row = tk.Frame(panel, bg="#16213e")
            row.pack(fill="x")
            s = ttk.Scale(row, from_=from_, to=to, variable=var, orient="horizontal")
            s.pack(side="left", fill="x", expand=True)
            lbl = tk.Label(row, text=str(default), font=("Courier New", 9),
                        fg="#a0c4ff", bg="#16213e", width=label_w)
            lbl.pack(side="left")
            s.config(command=lambda v: update_fn(lbl))
            return s

        # FILE
        section("📁  FILE")
        btn("Open Image", self.open_image)
        btn("Save Image", self.save_image)
        btn("⟳  Reset",  self.reset_image)

        # COLOUR
        section("🎨  COLOUR")
        btn("Show Red Channel",   lambda: self.show_channel("R"))
        btn("Show Green Channel", lambda: self.show_channel("G"))
        btn("Show Blue Channel",  lambda: self.show_channel("B"))
        btn("Grayscale",          self.to_grayscale)

        # GEOMETRY
        section("📐  GEOMETRY")
        btn("Rotate 90° CW",  lambda: self.rotate(cv2.ROTATE_90_CLOCKWISE))
        btn("Rotate 90° CCW", lambda: self.rotate(cv2.ROTATE_90_COUNTERCLOCKWISE))
        btn("Rotate 180°",    lambda: self.rotate(cv2.ROTATE_180))
        btn("Flip Horizontal", lambda: self.flip(1))
        btn("Flip Vertical",   lambda: self.flip(0))

        # ZOOM
        section("🔍  ZOOM  (2×)")
        tk.Label(panel, text="Nearest: blocky  |  Bilinear: smooth",
                font=("Courier New", 8), fg="#7090b0", bg="#16213e").pack(anchor="w", pady=(2,4))
        btn("Nearest Neighbor", lambda: self.zoom("nearest"))
        btn("Bilinear",         lambda: self.zoom("bilinear"))

        # ENHANCE
        section("✨  ENHANCE")
        tk.Label(panel, text="Redistributes pixel intensities\nfor better contrast.",
                font=("Courier New", 8), fg="#7090b0", bg="#16213e").pack(anchor="w", pady=(2,2))
        btn("Histogram Equalize", self.hist_equalize)

        tk.Label(panel, text="Gamma  (γ > 1 = brighter, γ < 1 = darker)",
                font=("Courier New", 8), fg="#7090b0", bg="#16213e",
                wraplength=190, justify="left").pack(anchor="w", pady=(6,0))
        self.gamma_var = tk.DoubleVar(value=1.0)
        slider(self.gamma_var, 0.1, 3.0, "1.00", 4,
            lambda lbl: lbl.config(text=f"{self.gamma_var.get():.2f}"))
        btn("Apply Gamma", self.apply_gamma)

        # FILTERS
        section("🌀  FILTERS")
        tk.Label(panel, text="Blur kernel size (odd, 3–21):",
                font=("Courier New", 9), fg="#a0b0c0", bg="#16213e").pack(anchor="w", pady=(4,0))
        self.blur_var = tk.IntVar(value=5)
        slider(self.blur_var, 3, 21, "5", 3,
            lambda lbl: lbl.config(text=str(int(self.blur_var.get()) | 1)))
        btn("Gaussian Blur", self.gaussian_blur)

        # EDGE DETECTION
        section("🔲  EDGE DETECTION")
        tk.Label(panel, text="Low threshold:",
                font=("Courier New", 9), fg="#a0b0c0", bg="#16213e").pack(anchor="w")
        self.canny_low_var = tk.IntVar(value=100)
        self.canny_low_lbl = tk.Label(panel)  # placeholder replaced by slider()
        low_row = tk.Frame(panel, bg="#16213e"); low_row.pack(fill="x")
        low_s = ttk.Scale(low_row, from_=0, to=255, variable=self.canny_low_var, orient="horizontal")
        low_s.pack(side="left", fill="x", expand=True)
        self.canny_low_lbl = tk.Label(low_row, text="100", font=("Courier New", 9),
                                    fg="#a0c4ff", bg="#16213e", width=4)
        self.canny_low_lbl.pack(side="left")

        tk.Label(panel, text="High threshold:",
                font=("Courier New", 9), fg="#a0b0c0", bg="#16213e").pack(anchor="w", pady=(4,0))
        self.canny_high_var = tk.IntVar(value=200)
        high_row = tk.Frame(panel, bg="#16213e"); high_row.pack(fill="x")
        high_s = ttk.Scale(high_row, from_=0, to=255, variable=self.canny_high_var, orient="horizontal")
        high_s.pack(side="left", fill="x", expand=True)
        self.canny_high_lbl = tk.Label(high_row, text="200", font=("Courier New", 9),
                                    fg="#a0c4ff", bg="#16213e", width=4)
        self.canny_high_lbl.pack(side="left")

        def upd_low(v=None):
            v = int(self.canny_low_var.get())
            self.canny_low_lbl.config(text=str(v))
            if v >= self.canny_high_var.get():
                self.canny_high_var.set(min(v+10, 255))
                self.canny_high_lbl.config(text=str(int(self.canny_high_var.get())))

        def upd_high(v=None):
            v = int(self.canny_high_var.get())
            self.canny_high_lbl.config(text=str(v))
            if v <= self.canny_low_var.get():
                self.canny_low_var.set(max(v-10, 0))
                self.canny_low_lbl.config(text=str(int(self.canny_low_var.get())))

        low_s.config(command=upd_low)
        high_s.config(command=upd_high)
        btn("Canny Edge Detect", self.canny_edges)

        # COMPARE
        section("📊  COMPARE")
        btn("Show Histogram", self.show_histogram)

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg="#1a1a2e")
        right.pack(side="left", fill="both", expand=True, padx=(8,0))

        lbl_s = dict(font=("Courier New", 11, "bold"), bg="#1a1a2e", fg="#a0c4ff", pady=4)
        tk.Label(right, text="◀  Original", **lbl_s).grid(row=0, column=0, sticky="w", padx=8)
        tk.Label(right, text="Result  ▶",   **lbl_s).grid(row=0, column=1, sticky="e", padx=8)

        cfg = dict(bg="#0d1117", bd=2, relief="ridge", highlightthickness=2, highlightbackground="#e94560")
        self.canvas_orig   = tk.Canvas(right, width=520, height=420, **cfg)
        self.canvas_result = tk.Canvas(right, width=520, height=420, **cfg)
        self.canvas_orig.grid(  row=1, column=0, padx=8, pady=4, sticky="nsew")
        self.canvas_result.grid(row=1, column=1, padx=8, pady=4, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)

        self._placeholder(self.canvas_orig,   "Open an image to start")
        self._placeholder(self.canvas_result, "Result appears here")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _placeholder(self, canvas, text):
        canvas.delete("all")
        canvas.create_text(260, 210, text=text, fill="#3a3a5c",
                        font=("Courier New", 13, "bold"))

    def _require_image(self):
        if self.current_img is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            return False
        return True

    def _show_orig(self):
        tk_img = cv2_to_tk(self.original_img)
        self.tk_img_left = tk_img
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(260, 210, anchor="center", image=tk_img)

    def _display_result(self, img, update_current=True):
        if update_current:
            self.current_img = img
        tk_img = cv2_to_tk(img)
        self.tk_img_right = tk_img
        self.canvas_result.delete("all")
        self.canvas_result.create_image(260, 210, anchor="center", image=tk_img)

    def _status(self, msg):
        self.status_var.set(f"  {msg}")

    # ── Feature 1 – Open ──────────────────────────────────────────────────────
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")])
        if not path: return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read the file."); return
        self.original_img = self.current_img = img.copy()
        self._show_orig()
        self._placeholder(self.canvas_result, "Apply an operation →")
        ch = f"Channels: {img.shape[2]}" if len(img.shape) == 3 else "Channels: 1 (grayscale)"
        self._status(f"Loaded: {path}  |  Size: {img.shape[1]}×{img.shape[0]}  |  {ch}")

    # ── Feature 11 – Save ─────────────────────────────────────────────────────
    def save_image(self):
        if not self._require_image(): return
        path = filedialog.asksaveasfilename(
            title="Save image as", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path: return
        cv2.imwrite(path, self.current_img)
        self._status(f"Saved to: {path}")

    # ── Feature 10 – Reset ────────────────────────────────────────────────────
    def reset_image(self):
        if not self._require_image(): return
        self.current_img = self.original_img.copy()
        self._display_result(self.current_img)
        self._status("Reset to original image.")

    # ── Feature 2 – Channels ──────────────────────────────────────────────────
    def show_channel(self, ch):
        if not self._require_image(): return
        if len(self.current_img.shape) == 2:
            messagebox.showinfo("Info", "Image is already grayscale."); return
        b, g, r = cv2.split(self.current_img)
        z = np.zeros_like(b)
        result, label = {
            "R": (cv2.merge([z, z, r]), "Red Channel"),
            "G": (cv2.merge([z, g, z]), "Green Channel"),
            "B": (cv2.merge([b, z, z]), "Blue Channel"),
        }[ch]
        self._display_result(result)
        self._status(f"Showing {label}.")

    # ── Feature 3 – Grayscale ─────────────────────────────────────────────────
    def to_grayscale(self):
        if not self._require_image(): return
        if len(self.current_img.shape) == 2:
            self._status("Image is already grayscale."); return
        self._display_result(cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY))
        self._status("Converted to Grayscale.")

    # ── Feature 4 – Rotate ────────────────────────────────────────────────────
    def rotate(self, code):
        if not self._require_image(): return
        self._display_result(cv2.rotate(self.current_img, code))
        self._status({
            cv2.ROTATE_90_CLOCKWISE:        "Rotated 90° Clockwise",
            cv2.ROTATE_90_COUNTERCLOCKWISE: "Rotated 90° Counter-Clockwise",
            cv2.ROTATE_180:                 "Rotated 180°",
        }.get(code, "Rotated."))

    # ── Feature 4 – Zoom ──────────────────────────────────────────────────────
    def zoom(self, method):
        if not self._require_image(): return
        src = self.current_img
        h, w = src.shape[:2]
        ch, cw = h//2, w//2
        crop = src[(h-ch)//2:(h-ch)//2+ch, (w-cw)//2:(w-cw)//2+cw]
        interp = cv2.INTER_NEAREST if method == "nearest" else cv2.INTER_LINEAR
        self._display_result(cv2.resize(crop, (w, h), interpolation=interp))
        name = "Nearest Neighbor" if method == "nearest" else "Bilinear"
        self._status(f"Zoom 2× ({name}) — Nearest=blocky | Bilinear=smooth.")

    # ── Feature 5 – Flip ──────────────────────────────────────────────────────
    def flip(self, code):
        if not self._require_image(): return
        self._display_result(cv2.flip(self.current_img, code))
        self._status(f"Flipped {'Horizontal' if code == 1 else 'Vertical'}.")

    # ── Feature 6 – Histogram Equalization ───────────────────────────────────
    def hist_equalize(self):
        if not self._require_image(): return
        img = self.current_img
        if len(img.shape) == 2:
            result = cv2.equalizeHist(img)
        else:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        self._display_result(result)
        self._status("Histogram Equalisation applied — contrast redistributed across 0–255.")

    # ── Feature 6 – Gamma Correction ─────────────────────────────────────────
    def apply_gamma(self):
        if not self._require_image(): return
        gamma = max(self.gamma_var.get(), 0.01)
        lut = np.array([((i/255.0)**(1.0/gamma))*255
                        for i in range(256)], dtype="uint8").reshape(256, 1)
        if len(self.current_img.shape) == 2:
            result = cv2.LUT(self.current_img, lut)
        else:
            result = cv2.merge([cv2.LUT(ch, lut) for ch in cv2.split(self.current_img)])
        self._display_result(result)
        self._status(f"Gamma Correction applied (γ = {gamma:.2f}).")

    # ── Feature 7 – Gaussian Blur ─────────────────────────────────────────────
    def gaussian_blur(self):
        if not self._require_image(): return
        k = int(self.blur_var.get())
        if k % 2 == 0: k += 1
        self._display_result(cv2.GaussianBlur(self.current_img, (k, k), 0))
        self._status(f"Gaussian Blur applied — kernel: {k}×{k}.")

    # ── Feature 8 – Canny Edge Detection ─────────────────────────────────────
    def canny_edges(self):
        if not self._require_image(): return
        low  = int(self.canny_low_var.get())
        high = int(self.canny_high_var.get())
        if low >= high: high = min(low+10, 255)
        gray = self.current_img if len(self.current_img.shape) == 2 \
            else cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
        self._display_result(cv2.Canny(gray, low, high))
        self._status(f"Canny Edge Detection — low: {low}, high: {high}.")

    # ── Feature 9 – Show Histogram ────────────────────────────────────────────
    def show_histogram(self):
        if not self._require_image(): return
        img = self.current_img

        fig, ax = plt.subplots(figsize=(5, 3.5), tight_layout=True)
        ax.set_facecolor("#1e1e2e")
        fig.patch.set_facecolor("#1e1e2e")

        if len(img.shape) == 2:
            # Pure grayscale
            ax.plot(cv2.calcHist([img],[0],None,[256],[0,256]).flatten(),
                    color="#aaaaaa", lw=1.2, label="Gray")
        else:
            b, g, r = cv2.split(img)
            # Check which channels are NOT all zeros
            channel_map = [
                (r, "#ef9a9a", "Red"),
                (g, "#81c784", "Green"),
                (b, "#4fc3f7", "Blue"),
            ]
            active = [(ch, col, lbl) for ch, col, lbl in channel_map
                    if np.any(ch > 0)]  # ← only plot channels with actual data

            if len(active) == 1:
                # Only one channel has data → show as gray (single channel view)
                ch, col, lbl = active[0]
                ax.plot(cv2.calcHist([ch],[0],None,[256],[0,256]).flatten(),
                        color=col, lw=1.2, label=lbl)
            else:
                # Normal color image → show all 3
                for ch, col, lbl in active:
                    ax.plot(cv2.calcHist([ch],[0],None,[256],[0,256]).flatten(),
                            color=col, lw=1.2, label=lbl)

        ax.set_xlim([0, 255])
        ax.set_title("Colour Histogram", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1e1e2e", labelcolor="white")

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        arr = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        plt.close(fig)
        hist_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        hist_img = cv2.resize(hist_img, (512, 380))

        self._display_result(hist_img, update_current=False)
        self._status("Colour Histogram displayed.")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Horizontal.TScale", background="#16213e",
                    troughcolor="#0f3460", sliderthickness=14)
    style.configure("Vertical.TScrollbar", background="#0f3460",
                    troughcolor="#16213e", arrowcolor="#e94560")
    ImageProcessorApp(root)
    root.mainloop()