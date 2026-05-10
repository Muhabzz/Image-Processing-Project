"""
IMAGE PROCESSING DESKTOP APPLICATION
Team Project – Python + OpenCV + Tkinter
	Mohab Nasser Abdelkader          23013228
	Ali Mohamed Mahmoud              23015406
  Youssef Amr Mohamed					     23014943
	Mostafa Waleed Hamed 					   23110006
	Youssef Samir Abdelkader				 23016427


HOW TO RUN:
  1. pip install opencv-python pillow matplotlib
  2. python Final.py
"""

import tkinter as tk # supplies the native desktop GUI toolkit for Python (cross-platform)
from tkinter import ttk, filedialog, messagebox # ttk = themed widgets (better slider), filedialog = file open/save dialogs, messagebox = pop-up alerts
import cv2 # provides all low-level image-processing algorithms
import numpy as np # deals with array manipulation and numerical operations -> images = [255,0,0]
from PIL import Image, ImageTk # bridges OpenCV images (numpy arrays) to Tkinter's PhotoImage format for display
import matplotlib
matplotlib.use("Agg") # use a non-interactive backend for rendering histograms to off-screen buffers (no pop-up windows) - Agg = Anti-Grain Geometry, a high-quality rendering engine
import matplotlib.pyplot as plt  # renders live histograms as images displayed inside the GUI canvases
from matplotlib.backends.backend_agg import FigureCanvasAgg # convert matplotlib figures to numpy arrays for display in Tkinter


# ── Helper: OpenCV image → Tkinter PhotoImage ─────────────────────────────────
def cv2_to_tk(img, max_w=520, max_h=380): # receieve an OpenCV image (BGR or grayscale) and convert it to a Tkinter-compatible PhotoImage, resizing to fit within max_w × max_h while preserving aspect ratio
    img = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2RGB if len(img.shape) == 2 else cv2.COLOR_BGR2RGB # OpenCV uses BGR order by default, but Tkinter expects RGB, so we convert the color space accordingly. If the image is grayscale (2D), we convert it to RGB by replicating the single channel across R, G, and B.
    )
    pil = Image.fromarray(img) # convert the numpy array (OpenCV image) to a PIL Image object, which provides more flexible image manipulation capabilities and can be easily converted to a Tkinter PhotoImage
    pil.thumbnail((max_w, max_h), Image.LANCZOS) # smalling the image to fit within the specified max width and height while preserving the aspect ratio. LANCZOS is a high-quality downsampling filter that produces smoother results than simpler methods like nearest-neighbor or bilinear interpolation.
    return ImageTk.PhotoImage(pil) # convert the PIL Image to a Tkinter PhotoImage, which can be displayed in Tkinter widgets like Canvas or Label. This step is necessary because Tkinter does not natively support displaying images in formats like OpenCV's BGR or PIL's Image, so we need to bridge the gap by converting to PhotoImage format.


def make_histogram_image(img, width=510, height=130):
    """
    Render a mini histogram for *img* and return it as a BGR numpy array
    sized (height × width).  The x-axis is always the full 0-255 range so the
    plot is never zoomed in.
    """
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100) # create a matplotlib figure and axis with the specified width and height in inches (converted from pixels by dividing by 100) and a resolution of 100 dots per inch (DPI). This sets up a canvas for drawing the histogram.
    fig.patch.set_facecolor("#0d1117") # Background color of the entire figure (including margins) is set to a dark shade of gray (#0d1117). This ensures that the histogram stands out against the background and gives it a sleek, modern look.
    ax.set_facecolor("#0d1117") # Background color of the plotting area (where the histogram bars will be drawn) is also set to the same dark gray color. This creates a consistent visual theme and helps the histogram bars pop against the background.

    if len(img.shape) == 2: # If the image is grayscale (2D array), we calculate the histogram for the single channel and plot it in a light gray color (#aaaaaa). The histogram is plotted as a line graph, and the area under the curve is filled with a semi-transparent version of the same color to enhance visibility. The x-axis represents pixel intensity values from 0 to 255, and the y-axis represents the frequency of each intensity value in the image.
        # Grayscale
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        ax.plot(hist, color="#aaaaaa", lw=1.0) # lw = line width of the histogram curve
        ax.fill_between(range(256), hist, alpha=0.25, color="#aaaaaa")
    else:
        b_ch, g_ch, r_ch = cv2.split(img) # If the image is color (3D array), we split it into its Blue, Green, and Red channels using OpenCV's cv2.split function. This allows us to calculate and plot separate histograms for each color channel, which can provide insights into the distribution of pixel intensities for each color component in the image.
        channel_map = [
            (r_ch, "#ef9a9a", "R"),
            (g_ch, "#81c784", "G"),
            (b_ch, "#4fc3f7", "B"),
        ]
        active = [(ch, col, lbl) for ch, col, lbl in channel_map if np.any(ch > 0)] # We create a list of active channels by checking if any pixel in each channel has a value greater than 0. This helps us determine which color channels are present in the image and should be plotted in the histogram. If a channel is completely black (all pixel values are 0), it will be excluded from the histogram to avoid cluttering the plot with empty data.

        if len(active) == 1:
            ch, col, lbl = active[0] # If there is only one active channel, we calculate its histogram and plot it using the specified color. The histogram is plotted as a line graph, and the area under the curve is filled with a semi-transparent version of the same color to enhance visibility. This allows us to visualize the distribution of pixel intensities for that specific color channel in the image.
            hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
            ax.plot(hist, color=col, lw=1.0)
            ax.fill_between(range(256), hist, alpha=0.25, color=col)
        else:
            for ch, col, lbl in active: # draw  every channel with different color
                hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
                ax.plot(hist, color=col, lw=1.0, label=lbl)
                ax.fill_between(range(256), hist, alpha=0.15, color=col)

    # ── Zoomed-out: always show the full intensity range ───────────────────
    ax.set_xlim(-5, 260)           # slight padding either side of 0-255
    ymax = ax.get_ylim()[1] # get the current maximum y-value (frequency) from the histogram plot, which represents the highest frequency of pixel intensities in the image. This value is used to set the upper limit of the y-axis to ensure that all histogram bars fit within the plot area.
    ax.set_ylim(0, ymax * 1.10)    # 10 % headroom at the top

    ax.tick_params(colors="#606080", labelsize=6)
    ax.spines[:].set_color("#2a2a4a")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5) # set the width of the plot borders (spines) to 0.5 pixels for a thinner, more subtle appearance that complements the overall design of the histogram.
    ax.set_xlabel("Intensity  (0 – 255)", color="#606080", fontsize=6, labelpad=2)

    plt.tight_layout(pad=0.3)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    arr = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)


# ── Main App ──────────────────────────────────────────────────────────────────
class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Vision Editor")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1200x820")
        self.root.minsize(950, 680)
        self.original_img = self.current_img = None
        self.tk_img_left  = self.tk_img_right = None
        self.tk_hist_left = self.tk_hist_right = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title
        tf = tk.Frame(self.root, bg="#16213e", pady=8)
        tf.pack(fill="x")
        tk.Label(
            tf, text="Image Processing - Vision Editor",
            font=("Courier New", 18, "bold"), fg="#e94560", bg="#16213e"
        ).pack()

        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill="both", expand=True, padx=10, pady=6)
        self._build_left_panel(main)
        self._build_right_panel(main)

        # Status bar
        self.status_var = tk.StringVar(value="Load an image to begin.")
        tk.Label(
            self.root, textvariable=self.status_var,
            font=("Courier New", 10), fg="#a0a0b0", bg="#0f3460",
            anchor="w", padx=12, pady=4
        ).pack(fill="x", side="bottom")

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
        cs.bind_all("<MouseWheel>",
                    lambda e: cs.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        BTN = dict(
            font=("Courier New", 10, "bold"), bg="#0f3460", fg="#e0e0e0",
            activebackground="#e94560", activeforeground="white",
            relief="flat", bd=0, padx=8, pady=6, cursor="hand2"
        )
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
        btn("Rotate 90° CW",   lambda: self.rotate(cv2.ROTATE_90_CLOCKWISE))
        btn("Rotate 90° CCW",  lambda: self.rotate(cv2.ROTATE_90_COUNTERCLOCKWISE))
        btn("Rotate 180°",     lambda: self.rotate(cv2.ROTATE_180))
        btn("Flip Horizontal", lambda: self.flip(1))
        btn("Flip Vertical",   lambda: self.flip(0))

        # ZOOM
        section("🔍  ZOOM  (2×)")
        tk.Label(panel, text="Nearest: blocky  |  Bilinear: smooth",
                 font=("Courier New", 8), fg="#7090b0", bg="#16213e").pack(
                     anchor="w", pady=(2, 4))
        btn("Nearest Neighbor", lambda: self.zoom("nearest"))
        btn("Bilinear",         lambda: self.zoom("bilinear"))

        # ENHANCE
        section("✨  ENHANCE")
        tk.Label(panel, text="Redistributes pixel intensities\nfor better contrast.",
                 font=("Courier New", 8), fg="#7090b0", bg="#16213e").pack(
                     anchor="w", pady=(2, 2))
        btn("Histogram Equalize", self.hist_equalize)

        tk.Label(panel, text="Gamma  (γ > 1 = brighter, γ < 1 = darker)",
                 font=("Courier New", 8), fg="#7090b0", bg="#16213e",
                 wraplength=190, justify="left").pack(anchor="w", pady=(6, 0))
        self.gamma_var = tk.DoubleVar(value=1.0)
        slider(self.gamma_var, 0.1, 3.0, "1.00", 4,
               lambda lbl: lbl.config(text=f"{self.gamma_var.get():.2f}"))
        btn("Apply Gamma", self.apply_gamma)

        # FILTERS
        section("🌀  FILTERS")
        tk.Label(panel, text="Blur kernel size (odd, 3–21):",
                 font=("Courier New", 9), fg="#a0b0c0", bg="#16213e").pack(
                     anchor="w", pady=(4, 0))
        self.blur_var = tk.IntVar(value=5)
        slider(self.blur_var, 3, 21, "5", 3,
               lambda lbl: lbl.config(text=str(int(self.blur_var.get()) | 1)))
        btn("Gaussian Blur", self.gaussian_blur)

        # EDGE DETECTION
        section("🔲  EDGE DETECTION")
        tk.Label(panel, text="Low threshold:",
                 font=("Courier New", 9), fg="#a0b0c0", bg="#16213e").pack(anchor="w")
        self.canny_low_var  = tk.IntVar(value=100)
        self.canny_high_var = tk.IntVar(value=200)

        low_row = tk.Frame(panel, bg="#16213e"); low_row.pack(fill="x")
        low_s = ttk.Scale(low_row, from_=0, to=255, variable=self.canny_low_var,
                          orient="horizontal")
        low_s.pack(side="left", fill="x", expand=True)
        self.canny_low_lbl = tk.Label(low_row, text="100", font=("Courier New", 9),
                                      fg="#a0c4ff", bg="#16213e", width=4)
        self.canny_low_lbl.pack(side="left")

        tk.Label(panel, text="High threshold:",
                 font=("Courier New", 9), fg="#a0b0c0", bg="#16213e").pack(
                     anchor="w", pady=(4, 0))
        high_row = tk.Frame(panel, bg="#16213e"); high_row.pack(fill="x")
        high_s = ttk.Scale(high_row, from_=0, to=255, variable=self.canny_high_var,
                           orient="horizontal")
        high_s.pack(side="left", fill="x", expand=True)
        self.canny_high_lbl = tk.Label(high_row, text="200", font=("Courier New", 9),
                                       fg="#a0c4ff", bg="#16213e", width=4)
        self.canny_high_lbl.pack(side="left")

        def upd_low(v=None):
            v = int(self.canny_low_var.get())
            self.canny_low_lbl.config(text=str(v))
            if v >= self.canny_high_var.get():
                self.canny_high_var.set(min(v + 10, 255))
                self.canny_high_lbl.config(text=str(int(self.canny_high_var.get())))

        def upd_high(v=None):
            v = int(self.canny_high_var.get())
            self.canny_high_lbl.config(text=str(v))
            if v <= self.canny_low_var.get():
                self.canny_low_var.set(max(v - 10, 0))
                self.canny_low_lbl.config(text=str(int(self.canny_low_var.get())))

        low_s.config(command=upd_low)
        high_s.config(command=upd_high)
        btn("Canny Edge Detect", self.canny_edges)

        # COMPARE
        section("📊  COMPARE")
        btn("Show Large Histogram", self.show_histogram)

    # ── Right panel: image canvases + histogram canvases below each ──────────
    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg="#1a1a2e")
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        lbl_s = dict(font=("Courier New", 11, "bold"), bg="#1a1a2e", fg="#a0c4ff", pady=4)
        tk.Label(right, text="◀  Original", **lbl_s).grid(
            row=0, column=0, sticky="w", padx=8)
        tk.Label(right, text="Result  ▶",   **lbl_s).grid(
            row=0, column=1, sticky="e", padx=8)

        img_cfg  = dict(bg="#0d1117", bd=2, relief="ridge",
                        highlightthickness=2, highlightbackground="#e94560")
        hist_cfg = dict(bg="#0d1117", bd=1, relief="ridge",
                        highlightthickness=1, highlightbackground="#2a2a4a")

        # Image canvases
        self.canvas_orig   = tk.Canvas(right, width=510, height=360, **img_cfg)
        self.canvas_result = tk.Canvas(right, width=510, height=360, **img_cfg)
        self.canvas_orig.grid(  row=1, column=0, padx=8, pady=(4, 2), sticky="nsew")
        self.canvas_result.grid(row=1, column=1, padx=8, pady=(4, 2), sticky="nsew")

        # Histogram label row
        hlbl_s = dict(font=("Courier New", 8, "bold"), bg="#1a1a2e", fg="#606090")
        tk.Label(right, text="Histogram — Original", **hlbl_s).grid(
            row=2, column=0, sticky="w", padx=8)
        tk.Label(right, text="Histogram — Result",   **hlbl_s).grid(
            row=2, column=1, sticky="w", padx=8)

        # Histogram canvases (fixed height, no expand)
        self.canvas_hist_orig   = tk.Canvas(right, width=510, height=130, **hist_cfg)
        self.canvas_hist_result = tk.Canvas(right, width=510, height=130, **hist_cfg)
        self.canvas_hist_orig.grid(  row=3, column=0, padx=8, pady=(0, 4), sticky="ew")
        self.canvas_hist_result.grid(row=3, column=1, padx=8, pady=(0, 4), sticky="ew")

        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)

        self._placeholder(self.canvas_orig,   "Open an image to start")
        self._placeholder(self.canvas_result, "Result appears here")
        self._hist_placeholder(self.canvas_hist_orig)
        self._hist_placeholder(self.canvas_hist_result)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _placeholder(self, canvas, text):
        canvas.delete("all")
        canvas.create_text(255, 180, text=text, fill="#3a3a5c",
                           font=("Courier New", 13, "bold"))

    def _hist_placeholder(self, canvas):
        canvas.delete("all")
        canvas.create_text(255, 65, text="No data", fill="#2a2a4a",
                           font=("Courier New", 10))

    def _require_image(self):
        if self.current_img is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            return False
        return True

    def _show_orig(self):
        tk_img = cv2_to_tk(self.original_img)
        self.tk_img_left = tk_img
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(255, 180, anchor="center", image=tk_img)

    def _display_result(self, img, update_current=True):
        if update_current:
            self.current_img = img
        tk_img = cv2_to_tk(img)
        self.tk_img_right = tk_img
        self.canvas_result.delete("all")
        self.canvas_result.create_image(255, 180, anchor="center", image=tk_img)

        if update_current:
            self._update_histograms()

    def _update_histograms(self):
        """Render and display mini-histograms under both image panels."""
        # Determine canvas pixel size (may differ from widget size if resized)
        w_orig = max(self.canvas_hist_orig.winfo_width(),   510)
        w_res  = max(self.canvas_hist_result.winfo_width(), 510)
        h      = 130

        if self.original_img is not None:
            hist_img = make_histogram_image(self.original_img, width=w_orig, height=h)
            tk_h = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)))
            self.tk_hist_left = tk_h
            self.canvas_hist_orig.delete("all")
            self.canvas_hist_orig.create_image(0, 0, anchor="nw", image=tk_h)

        if self.current_img is not None:
            hist_img = make_histogram_image(self.current_img, width=w_res, height=h)
            tk_h = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)))
            self.tk_hist_right = tk_h
            self.canvas_hist_result.delete("all")
            self.canvas_hist_result.create_image(0, 0, anchor="nw", image=tk_h)

    def _status(self, msg):
        self.status_var.set(f"  {msg}")

    # ── Feature 1 – Open ──────────────────────────────────────────────────────
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read the file.")
            return
        self.original_img = self.current_img = img.copy()
        self._show_orig()
        self._placeholder(self.canvas_result, "Apply an operation →")
        self._update_histograms()
        ch = (f"Channels: {img.shape[2]}" if len(img.shape) == 3
              else "Channels: 1 (grayscale)")
        self._status(f"Loaded: {path}  |  Size: {img.shape[1]}×{img.shape[0]}  |  {ch}")

    # ── Feature 11 – Save ─────────────────────────────────────────────────────
    def save_image(self):
        if not self._require_image():
            return
        path = filedialog.asksaveasfilename(
            title="Save image as", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        cv2.imwrite(path, self.current_img)
        self._status(f"Saved to: {path}")

    # ── Feature 10 – Reset ────────────────────────────────────────────────────
    def reset_image(self):
        if not self._require_image():
            return
        self.current_img = self.original_img.copy()
        self._display_result(self.current_img)
        self._status("Reset to original image.")

    # ── Feature 2 – Channels ──────────────────────────────────────────────────
    def show_channel(self, ch):
        if not self._require_image():
            return
        if len(self.current_img.shape) == 2:
            messagebox.showinfo("Info", "Image is already grayscale.")
            return
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
        if not self._require_image():
            return
        if len(self.current_img.shape) == 2:
            self._status("Image is already grayscale.")
            return
        self._display_result(cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY))
        self._status("Converted to Grayscale.")

    # ── Feature 4 – Rotate ────────────────────────────────────────────────────
    def rotate(self, code):
        if not self._require_image():
            return
        self._display_result(cv2.rotate(self.current_img, code))
        self._status({
            cv2.ROTATE_90_CLOCKWISE:        "Rotated 90° Clockwise",
            cv2.ROTATE_90_COUNTERCLOCKWISE: "Rotated 90° Counter-Clockwise",
            cv2.ROTATE_180:                 "Rotated 180°",
        }.get(code, "Rotated."))

    # ── Feature 4 – Zoom ──────────────────────────────────────────────────────
    def zoom(self, method):
        if not self._require_image():
            return
        src = self.current_img
        h, w = src.shape[:2]
        ch, cw = h // 2, w // 2
        crop = src[(h - ch) // 2:(h - ch) // 2 + ch,
                   (w - cw) // 2:(w - cw) // 2 + cw]
        interp = cv2.INTER_NEAREST if method == "nearest" else cv2.INTER_LINEAR
        self._display_result(cv2.resize(crop, (w, h), interpolation=interp))
        name = "Nearest Neighbor" if method == "nearest" else "Bilinear"
        self._status(f"Zoom 2× ({name}) — Nearest=blocky | Bilinear=smooth.")

    # ── Feature 5 – Flip ──────────────────────────────────────────────────────
    def flip(self, code):
        if not self._require_image():
            return
        self._display_result(cv2.flip(self.current_img, code))
        self._status(f"Flipped {'Horizontal' if code == 1 else 'Vertical'}.")

    # ── Feature 6 – Histogram Equalization ───────────────────────────────────
    def hist_equalize(self):
        if not self._require_image():
            return
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
        if not self._require_image():
            return
        gamma = max(self.gamma_var.get(), 0.01)
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                        for i in range(256)], dtype="uint8").reshape(256, 1)
        if len(self.current_img.shape) == 2:
            result = cv2.LUT(self.current_img, lut)
        else:
            result = cv2.merge([cv2.LUT(ch, lut) for ch in cv2.split(self.current_img)])
        self._display_result(result)
        self._status(f"Gamma Correction applied (γ = {gamma:.2f}).")

    # ── Feature 7 – Gaussian Blur ─────────────────────────────────────────────
    def gaussian_blur(self):
        if not self._require_image():
            return
        k = int(self.blur_var.get())
        if k % 2 == 0:
            k += 1
        self._display_result(cv2.GaussianBlur(self.current_img, (k, k), 0))
        self._status(f"Gaussian Blur applied — kernel: {k}×{k}.")

    # ── Feature 8 – Canny Edge Detection ─────────────────────────────────────
    def canny_edges(self):
        if not self._require_image():
            return
        low  = int(self.canny_low_var.get())
        high = int(self.canny_high_var.get())
        if low >= high:
            high = min(low + 10, 255)
        gray = (self.current_img if len(self.current_img.shape) == 2
                else cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY))
        self._display_result(cv2.Canny(gray, low, high))
        self._status(f"Canny Edge Detection — low: {low}, high: {high}.")

    # ── Feature 9 – Show Large Histogram (detail view) ────────────────────────
    def show_histogram(self):
        if not self._require_image():
            return
        img = self.current_img

        fig, ax = plt.subplots(figsize=(5, 3.5), tight_layout=True)
        ax.set_facecolor("#1e1e2e")
        fig.patch.set_facecolor("#1e1e2e")

        if len(img.shape) == 2:
            ax.plot(cv2.calcHist([img], [0], None, [256], [0, 256]).flatten(),
                    color="#aaaaaa", lw=1.2, label="Gray")
        else:
            b, g, r = cv2.split(img)
            channel_map = [
                (r, "#ef9a9a", "Red"),
                (g, "#81c784", "Green"),
                (b, "#4fc3f7", "Blue"),
            ]
            active = [(ch, col, lbl) for ch, col, lbl in channel_map
                      if np.any(ch > 0)]

            if len(active) == 1:
                ch, col, lbl = active[0]
                ax.plot(cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten(),
                        color=col, lw=1.2, label=lbl)
            else:
                for ch, col, lbl in active:
                    ax.plot(cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten(),
                            color=col, lw=1.2, label=lbl)

        ax.set_xlim(-5, 260)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.10)
        ax.set_title("Colour Histogram (Result)", color="white")
        ax.tick_params(colors="white")
        ax.set_xlabel("Intensity  (0 – 255)", color="#a0a0c0")
        ax.legend(facecolor="#1e1e2e", labelcolor="white")

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        arr = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        plt.close(fig)
        hist_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        hist_img = cv2.resize(hist_img, (512, 380))

        self._display_result(hist_img, update_current=False)
        self._status("Large Colour Histogram displayed (result not updated).")


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