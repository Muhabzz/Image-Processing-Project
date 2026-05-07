"""
=============================================================================
  IMAGE PROCESSING DESKTOP APPLICATION
  Team Project – Python + OpenCV + Tkinter
=============================================================================
  HOW TO RUN:
    1. Install dependencies:  pip install opencv-python pillow
    2. Run this file:         python image_processor.py
=============================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2                          # OpenCV – all image operations
import numpy as np                  # NumPy – needed by OpenCV internally
from PIL import Image, ImageTk      # Pillow – converts OpenCV images → Tkinter


# =============================================================================
#  HELPER: Convert an OpenCV image to a Tkinter-displayable PhotoImage
# =============================================================================
def cv2_to_tk(cv_img, max_w=520, max_h=420):
    """
    OpenCV stores images as BGR numpy arrays.
    Tkinter needs an RGB PIL ImageTk object.
    We also resize the image to fit the canvas while keeping the aspect ratio.
    """
    # --- Handle grayscale (2-D array) by converting to RGB ---
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    else:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(cv_img)

    # --- Proportional resize so the image fits the display box ---
    pil_img.thumbnail((max_w, max_h), Image.LANCZOS)

    return ImageTk.PhotoImage(pil_img)


# =============================================================================
#  MAIN APPLICATION CLASS
# =============================================================================
class ImageProcessorApp:
    """
    The entire desktop application lives inside this class.
    • self.original_img  – the image loaded from disk (never modified)
    • self.current_img   – the image currently displayed / being worked on
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1200x780")
        self.root.minsize(900, 650)

        # ---------- State variables ----------
        self.original_img = None   # keeps the untouched original
        self.current_img  = None   # what is currently displayed
        self.tk_img_left  = None   # reference kept so GC doesn't delete it
        self.tk_img_right = None

        # ---------- Build the interface ----------
        self._build_ui()

    # =========================================================================
    #  UI CONSTRUCTION
    # =========================================================================
    def _build_ui(self):
        """Create every widget: toolbar, canvases, sliders, status bar."""

        # ── Title bar ────────────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg="#16213e", pady=8)
        title_frame.pack(fill="x")
        tk.Label(
            title_frame, text="🖼  Image Processing Studio",
            font=("Courier New", 18, "bold"),
            fg="#e94560", bg="#16213e"
        ).pack()

        # ── Main layout: left panel (buttons) + right panel (canvases) ───────
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill="both", expand=True, padx=10, pady=6)

        self._build_left_panel(main)   # control buttons
        self._build_right_panel(main)  # image display area

        # ── Status bar at the bottom ─────────────────────────────────────────
        self.status_var = tk.StringVar(value="Load an image to begin.")
        tk.Label(
            self.root, textvariable=self.status_var,
            font=("Courier New", 10), fg="#a0a0b0", bg="#0f3460",
            anchor="w", padx=12, pady=4
        ).pack(fill="x", side="bottom")

    # -------------------------------------------------------------------------
    def _build_left_panel(self, parent):
        """All the control buttons and sliders on the left side."""

        # ── Scrollable left panel ─────────────────────────────────────────────
        container = tk.Frame(parent, bg="#16213e", width=230)
        container.pack(side="left", fill="y")
        container.pack_propagate(False)

        canvas_scroll = tk.Canvas(container, bg="#16213e", width=220,
                                  highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical",
                                  command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas_scroll.pack(side="left", fill="both", expand=True)

        panel = tk.Frame(canvas_scroll, bg="#16213e", padx=10, pady=10)
        panel_window = canvas_scroll.create_window((0, 0), window=panel,
                                                   anchor="nw")

        def on_frame_configure(event):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))

        def on_canvas_configure(event):
            canvas_scroll.itemconfig(panel_window, width=event.width)

        panel.bind("<Configure>", on_frame_configure)
        canvas_scroll.bind("<Configure>", on_canvas_configure)

        # Mouse-wheel scrolling
        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)

        # Style shared by every button
        BTN = dict(
            font=("Courier New", 10, "bold"),
            bg="#0f3460", fg="#e0e0e0",
            activebackground="#e94560", activeforeground="white",
            relief="flat", bd=0, padx=8, pady=6, cursor="hand2"
        )
        HDR = dict(font=("Courier New", 9, "bold"), fg="#e94560",
                   bg="#16213e", anchor="w")

        def section(text):
            tk.Label(panel, text=text, **HDR).pack(fill="x", pady=(10, 2))
            tk.Frame(panel, bg="#e94560", height=1).pack(fill="x")

        def btn(text, cmd):
            b = tk.Button(panel, text=text, command=cmd, **BTN)
            b.pack(fill="x", pady=2)
            b.bind("<Enter>", lambda e: b.config(bg="#e94560"))
            b.bind("<Leave>", lambda e: b.config(bg="#0f3460"))
            return b

        # ── File operations ───────────────────────────────────────────────────
        section("📁  FILE")
        btn("Open Image",  self.open_image)
        btn("Save Image",  self.save_image)
        btn("⟳  Reset",   self.reset_image)

        # ── Channels & colour ─────────────────────────────────────────────────
        section("🎨  COLOUR")
        btn("Show Red Channel",   lambda: self.show_channel("R"))
        btn("Show Green Channel", lambda: self.show_channel("G"))
        btn("Show Blue Channel",  lambda: self.show_channel("B"))
        btn("Grayscale",          self.to_grayscale)

        # ── Geometric ─────────────────────────────────────────────────────────
        section("📐  GEOMETRY")
        btn("Rotate 90° CW",      lambda: self.rotate(cv2.ROTATE_90_CLOCKWISE))
        btn("Rotate 90° CCW",     lambda: self.rotate(cv2.ROTATE_90_COUNTERCLOCKWISE))
        btn("Rotate 180°",        lambda: self.rotate(cv2.ROTATE_180))
        btn("Flip Horizontal",    lambda: self.flip(1))
        btn("Flip Vertical",      lambda: self.flip(0))

        # ── Zoom ──────────────────────────────────────────────────────────────
        section("🔍  ZOOM  (2×)")

        # Info label explaining interpolation difference
        zoom_info = tk.Label(
            panel,
            text="Nearest: blocky/pixelated\nBilinear: smooth/blended",
            font=("Courier New", 8), fg="#7090b0", bg="#16213e",
            justify="left"
        )
        zoom_info.pack(anchor="w", pady=(2, 4))

        btn("Nearest Neighbor",   lambda: self.zoom("nearest"))
        btn("Bilinear",           lambda: self.zoom("bilinear"))

        # ── Enhancement ───────────────────────────────────────────────────────
        section("✨  ENHANCE")

        # --- Histogram Equalisation ---
        he_info = tk.Label(
            panel,
            text="Redistributes pixel intensities\nfor better contrast.",
            font=("Courier New", 8), fg="#7090b0", bg="#16213e",
            justify="left"
        )
        he_info.pack(anchor="w", pady=(2, 2))
        btn("Histogram Equalize", self.hist_equalize)

        # --- Gamma slider + button ---
        gamma_frame = tk.Frame(panel, bg="#16213e")
        gamma_frame.pack(fill="x", pady=(6, 0))

        tk.Label(gamma_frame, text="Gamma  (γ < 1 = brighter, γ > 1 = darker)",
                 font=("Courier New", 8), fg="#7090b0", bg="#16213e",
                 wraplength=190, justify="left").pack(anchor="w")

        slider_row = tk.Frame(gamma_frame, bg="#16213e")
        slider_row.pack(fill="x")

        self.gamma_var = tk.DoubleVar(value=1.0)
        gamma_slider = ttk.Scale(slider_row, from_=0.1, to=3.0,
                                 variable=self.gamma_var, orient="horizontal")
        gamma_slider.pack(side="left", fill="x", expand=True)

        self.gamma_label = tk.Label(slider_row, text="1.00",
                                    font=("Courier New", 9), fg="#a0c4ff",
                                    bg="#16213e", width=4)
        self.gamma_label.pack(side="left")

        def update_gamma_label(val=None):
            self.gamma_label.config(text=f"{self.gamma_var.get():.2f}")

        gamma_slider.config(command=update_gamma_label)
        btn("Apply Gamma", self.apply_gamma)

        # ── Filters ───────────────────────────────────────────────────────────
        section("🌀  FILTERS")

        # --- Gaussian Blur ---
        blur_info = tk.Label(
            panel,
            text="Smooths image by averaging\nneighbouring pixels.\nLarger kernel = stronger blur.",
            font=("Courier New", 8), fg="#7090b0", bg="#16213e",
            justify="left"
        )
        blur_info.pack(anchor="w", pady=(2, 2))

        tk.Label(panel, text="Blur kernel size (odd, 3–21):",
                 font=("Courier New", 9), fg="#a0b0c0",
                 bg="#16213e").pack(anchor="w", pady=(4, 0))

        blur_slider_row = tk.Frame(panel, bg="#16213e")
        blur_slider_row.pack(fill="x")

        self.blur_var = tk.IntVar(value=5)
        blur_slider = ttk.Scale(blur_slider_row, from_=3, to=21,
                                variable=self.blur_var, orient="horizontal")
        blur_slider.pack(side="left", fill="x", expand=True)

        self.blur_label = tk.Label(blur_slider_row, text="5",
                                   font=("Courier New", 9), fg="#a0c4ff",
                                   bg="#16213e", width=3)
        self.blur_label.pack(side="left")

        def update_blur_label(val=None):
            k = int(self.blur_var.get())
            if k % 2 == 0:
                k += 1
            self.blur_label.config(text=str(k))

        blur_slider.config(command=update_blur_label)
        btn("Gaussian Blur", self.gaussian_blur)

        # --- Canny Edge Detection ---
        section("🔲  EDGE DETECTION")

        canny_info = tk.Label(
            panel,
            text="Canny algorithm steps:\n"
                 "1. Gaussian smoothing\n"
                 "2. Gradient calculation\n"
                 "3. Non-max suppression\n"
                 "4. Hysteresis thresholding",
            font=("Courier New", 8), fg="#7090b0", bg="#16213e",
            justify="left"
        )
        canny_info.pack(anchor="w", pady=(2, 4))

        # Low threshold slider
        tk.Label(panel, text="Low threshold:",
                 font=("Courier New", 9), fg="#a0b0c0",
                 bg="#16213e").pack(anchor="w")

        low_row = tk.Frame(panel, bg="#16213e")
        low_row.pack(fill="x")
        self.canny_low_var = tk.IntVar(value=100)
        low_slider = ttk.Scale(low_row, from_=0, to=255,
                               variable=self.canny_low_var, orient="horizontal")
        low_slider.pack(side="left", fill="x", expand=True)
        self.canny_low_label = tk.Label(low_row, text="100",
                                        font=("Courier New", 9), fg="#a0c4ff",
                                        bg="#16213e", width=4)
        self.canny_low_label.pack(side="left")

        # High threshold slider
        tk.Label(panel, text="High threshold:",
                 font=("Courier New", 9), fg="#a0b0c0",
                 bg="#16213e").pack(anchor="w", pady=(4, 0))

        high_row = tk.Frame(panel, bg="#16213e")
        high_row.pack(fill="x")
        self.canny_high_var = tk.IntVar(value=200)
        high_slider = ttk.Scale(high_row, from_=0, to=255,
                                variable=self.canny_high_var, orient="horizontal")
        high_slider.pack(side="left", fill="x", expand=True)
        self.canny_high_label = tk.Label(high_row, text="200",
                                         font=("Courier New", 9), fg="#a0c4ff",
                                         bg="#16213e", width=4)
        self.canny_high_label.pack(side="left")

        def update_canny_low(val=None):
            v = int(self.canny_low_var.get())
            self.canny_low_label.config(text=str(v))
            # Keep low < high
            if v >= self.canny_high_var.get():
                self.canny_high_var.set(min(v + 10, 255))
                self.canny_high_label.config(text=str(int(self.canny_high_var.get())))

        def update_canny_high(val=None):
            v = int(self.canny_high_var.get())
            self.canny_high_label.config(text=str(v))
            if v <= self.canny_low_var.get():
                self.canny_low_var.set(max(v - 10, 0))
                self.canny_low_label.config(text=str(int(self.canny_low_var.get())))

        low_slider.config(command=update_canny_low)
        high_slider.config(command=update_canny_high)

        btn("Canny Edge Detect", self.canny_edges)

        # ── Compare ───────────────────────────────────────────────────────────
        section("📊  COMPARE")
        btn("Show Histogram", self.show_histogram)

    # -------------------------------------------------------------------------
    def _build_right_panel(self, parent):
        """Two side-by-side image canvases with labels."""

        right = tk.Frame(parent, bg="#1a1a2e")
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        lbl_style = dict(font=("Courier New", 11, "bold"),
                         bg="#1a1a2e", fg="#a0c4ff", pady=4)
        tk.Label(right, text="◀  Original", **lbl_style).grid(
            row=0, column=0, sticky="w", padx=8)
        tk.Label(right, text="Result  ▶", **lbl_style).grid(
            row=0, column=1, sticky="e", padx=8)

        canvas_cfg = dict(bg="#0d1117", bd=2, relief="ridge",
                          highlightthickness=2,
                          highlightbackground="#e94560")

        self.canvas_orig = tk.Canvas(right, width=520, height=420,
                                     **canvas_cfg)
        self.canvas_orig.grid(row=1, column=0, padx=8, pady=4, sticky="nsew")

        self.canvas_result = tk.Canvas(right, width=520, height=420,
                                       **canvas_cfg)
        self.canvas_result.grid(row=1, column=1, padx=8, pady=4, sticky="nsew")

        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)

        self._canvas_placeholder(self.canvas_orig,   "Open an image to start")
        self._canvas_placeholder(self.canvas_result, "Result appears here")

    # =========================================================================
    #  INTERNAL HELPERS
    # =========================================================================
    def _canvas_placeholder(self, canvas, text):
        canvas.delete("all")
        canvas.create_text(260, 210, text=text,
                           fill="#3a3a5c",
                           font=("Courier New", 13, "bold"))

    def _require_image(self):
        """Return False and show a warning if no image is loaded yet."""
        if self.current_img is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            return False
        return True

    def _display_original(self):
        """Render self.original_img on the left canvas."""
        tk_img = cv2_to_tk(self.original_img)
        self.tk_img_left = tk_img
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(260, 210, anchor="center", image=tk_img)

    def _display_result(self, img):
        """Render any OpenCV image on the right canvas and update current."""
        self.current_img = img
        tk_img = cv2_to_tk(img)
        self.tk_img_right = tk_img
        self.canvas_result.delete("all")
        self.canvas_result.create_image(260, 210, anchor="center", image=tk_img)

    def _set_status(self, msg):
        self.status_var.set(f"  {msg}")

    # =========================================================================
    #  FEATURE 1 – OPEN IMAGE
    # =========================================================================
    def open_image(self):
        """
        Open a file-chooser dialog filtered to JPEG and PNG.
        cv2.imread reads the file into a BGR numpy array.
        """
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("All files", "*.*")]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read the file.")
            return

        self.original_img = img.copy()
        self.current_img  = img.copy()

        self._display_original()
        self._display_result(img)
        self._canvas_placeholder(self.canvas_result, "Apply an operation →")

        # FIX: safely handle both colour (3-channel) and grayscale images
        if len(img.shape) == 3:
            ch_info = f"Channels: {img.shape[2]}"
        else:
            ch_info = "Channels: 1 (grayscale)"

        self._set_status(
            f"Loaded: {path}  |  Size: {img.shape[1]}×{img.shape[0]}  |  {ch_info}"
        )

    # =========================================================================
    #  FEATURE 11 – SAVE IMAGE
    # =========================================================================
    def save_image(self):
        if not self._require_image():
            return

        path = filedialog.asksaveasfilename(
            title="Save image as",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if not path:
            return

        cv2.imwrite(path, self.current_img)
        self._set_status(f"Saved to: {path}")

    # =========================================================================
    #  FEATURE 10 – RESET
    # =========================================================================
    def reset_image(self):
        if not self._require_image():
            return

        self.current_img = self.original_img.copy()
        self._display_result(self.current_img)
        self._set_status("Reset to original image.")

    # =========================================================================
    #  FEATURE 2 – SHOW INDIVIDUAL COLOUR CHANNELS
    # =========================================================================
    def show_channel(self, channel):
        """
        cv2.split(img) returns three 2-D arrays: (B, G, R).
        We put the chosen channel into a 3-channel image so it shows
        with the correct colour tint.
        """
        if not self._require_image():
            return

        # If already grayscale, nothing meaningful to split
        if len(self.current_img.shape) == 2:
            messagebox.showinfo("Info", "Image is already grayscale – no colour channels to split.")
            return

        b, g, r = cv2.split(self.current_img)
        zeros = np.zeros_like(b)

        if channel == "R":
            result = cv2.merge([zeros, zeros, r])
            label  = "Red Channel"
        elif channel == "G":
            result = cv2.merge([zeros, g, zeros])
            label  = "Green Channel"
        else:
            result = cv2.merge([b, zeros, zeros])
            label  = "Blue Channel"

        self._display_result(result)
        self._set_status(f"Showing {label}.")

    # =========================================================================
    #  FEATURE 3 – GRAYSCALE
    # =========================================================================
    def to_grayscale(self):
        """
        cv2.COLOR_BGR2GRAY applies the standard luminance formula:
        Y = 0.299·R + 0.587·G + 0.114·B
        """
        if not self._require_image():
            return

        if len(self.current_img.shape) == 2:
            self._set_status("Image is already grayscale.")
            return

        gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
        self._display_result(gray)
        self._set_status("Converted to Grayscale.")

    # =========================================================================
    #  FEATURE 4 – GEOMETRIC: ROTATION
    # =========================================================================
    def rotate(self, code):
        if not self._require_image():
            return

        rotated = cv2.rotate(self.current_img, code)
        self._display_result(rotated)
        labels = {
            cv2.ROTATE_90_CLOCKWISE:        "Rotated 90° Clockwise",
            cv2.ROTATE_90_COUNTERCLOCKWISE: "Rotated 90° Counter-Clockwise",
            cv2.ROTATE_180:                 "Rotated 180°",
        }
        self._set_status(labels.get(code, "Rotated."))

    # =========================================================================
    #  FEATURE 4 – GEOMETRIC: ZOOM (Nearest Neighbour vs Bilinear)   [FIXED]
    # =========================================================================
    def zoom(self, method):
        """
        cv2.resize scales the image.
        • INTER_NEAREST – picks the closest pixel  → blocky / pixelated look
        • INTER_LINEAR  – bilinear interpolation   → smoother result

        FIX: we always zoom from the CURRENT image (which may already be
        grayscale, rotated, etc.) and correctly handle both 2-D (grayscale)
        and 3-D (colour) arrays so no crash occurs.
        """
        if not self._require_image():
            return

        interp = cv2.INTER_NEAREST if method == "nearest" else cv2.INTER_LINEAR

        h, w = self.current_img.shape[:2]   # works for both 2-D and 3-D arrays
        new_w, new_h = w * 2, h * 2

        zoomed = cv2.resize(self.current_img, (new_w, new_h),
                            interpolation=interp)
        self._display_result(zoomed)

        name = "Nearest Neighbor" if method == "nearest" else "Bilinear"
        self._set_status(
            f"Zoomed 2× using {name} interpolation.  "
            f"New size: {new_w}×{new_h}  (displayed scaled to fit)"
        )

    # =========================================================================
    #  FEATURE 5 – FLIP
    # =========================================================================
    def flip(self, flip_code):
        """
        cv2.flip(img, flipCode):
          1  → horizontal (mirror left-right)
          0  → vertical   (mirror top-bottom)
        """
        if not self._require_image():
            return

        flipped = cv2.flip(self.current_img, flip_code)
        self._display_result(flipped)
        label = "Horizontal" if flip_code == 1 else "Vertical"
        self._set_status(f"Flipped {label}.")

    # =========================================================================
    #  FEATURE 6 – ENHANCEMENT: HISTOGRAM EQUALISATION
    # =========================================================================
    def hist_equalize(self):
        """
        Histogram equalisation redistributes pixel intensity values so they
        span the full 0–255 range → improves contrast in dark/washed-out images.

        For colour images we convert to YCrCb, equalise only the Y (luma)
        channel, then convert back — this avoids colour distortion.
        """
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
        self._set_status("Histogram Equalisation applied — contrast redistributed across 0–255.")

    # =========================================================================
    #  FEATURE 6 – ENHANCEMENT: GAMMA CORRECTION
    # =========================================================================
    def apply_gamma(self):
        """
        Gamma correction changes the overall brightness curve.
        • γ < 1  → brighter (lifts shadows)
        • γ > 1  → darker   (crushes highlights)
        • γ = 1  → no change

        Formula: output = (input / 255) ^ (1/γ) × 255
        A precomputed 256-entry LUT (lookup table) is used for speed.
        """
        if not self._require_image():
            return

        gamma = self.gamma_var.get()
        if gamma <= 0:
            gamma = 0.01

        inv_gamma = 1.0 / gamma
        lut = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ], dtype="uint8")

        result = cv2.LUT(self.current_img, lut)
        self._display_result(result)
        self._set_status(f"Gamma Correction applied (γ = {gamma:.2f}).")

    # =========================================================================
    #  FEATURE 7 – GAUSSIAN BLUR
    # =========================================================================
    def gaussian_blur(self):
        """
        GaussianBlur smooths the image by replacing each pixel with a
        weighted average of its neighbours. The kernel must be ODD-sized.
        Larger kernels → stronger blur.

        sigma=0 tells OpenCV to derive sigma automatically from kernel size.
        """
        if not self._require_image():
            return

        k = int(self.blur_var.get())
        if k % 2 == 0:
            k += 1

        blurred = cv2.GaussianBlur(self.current_img, (k, k), sigmaX=0)
        self._display_result(blurred)
        self._set_status(f"Gaussian Blur applied — kernel: {k}×{k}  (σ derived automatically).")

    # =========================================================================
    #  FEATURE 8 – CANNY EDGE DETECTION
    # =========================================================================
    def canny_edges(self):
        """
        Canny is a multi-stage edge detector:
        1. Gaussian smoothing         → removes noise
        2. Gradient magnitude         → finds intensity changes
        3. Non-maximum suppression    → thins the edges to 1-pixel width
        4. Double threshold + hysteresis → keeps only strong/connected edges

        Low/high thresholds are now user-controlled via sliders:
        • Pixels above HIGH  → definitely an edge
        • Pixels below LOW   → definitely not an edge
        • Pixels in between  → edge only if connected to a strong edge
        """
        if not self._require_image():
            return

        low  = int(self.canny_low_var.get())
        high = int(self.canny_high_var.get())

        # Ensure valid ordering
        if low >= high:
            high = min(low + 10, 255)

        # Canny requires a grayscale (single-channel) input
        if len(self.current_img.shape) == 2:
            gray = self.current_img
        else:
            gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, low, high)
        self._display_result(edges)
        self._set_status(
            f"Canny Edge Detection applied — low threshold: {low}, high threshold: {high}."
        )

    # =========================================================================
    #  FEATURE 9 – COMPARE: SHOW HISTOGRAM
    # =========================================================================
    def show_histogram(self):
        """
        Draws a colour histogram for the current image on the right canvas.
        For each channel (B, G, R) we compute frequencies across 256 bins
        and draw filled polygons, blended for overlap visibility.
        """
        if not self._require_image():
            return

        img = self.current_img
        hist_h, hist_w = 380, 512
        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

        if len(img.shape) == 2:
            channels_indices = [0]
            channel_data_list = [img]
            colours = [(200, 200, 200)]
            ch_labels = ["Gray"]
        else:
            channels_indices = [0, 1, 2]
            channel_data_list = [img[:, :, i] for i in range(3)]
            colours = [(255, 60, 60), (60, 255, 60), (60, 60, 255)]   # B, G, R
            ch_labels = ["Blue", "Green", "Red"]

        for ch_data, colour in zip(channel_data_list, colours):
            hist = cv2.calcHist([ch_data], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, hist_h - 40, cv2.NORM_MINMAX)

            pts = []
            for x, val in enumerate(hist):
                px = int(x * hist_w / 256)
                py = hist_h - int(val[0]) - 1
                pts.append([px, py])

            pts = [[0, hist_h - 1]] + pts + [[hist_w - 1, hist_h - 1]]
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(hist_img, [pts], colour)

        # Blend so overlapping channels remain visible
        blended = cv2.addWeighted(hist_img, 0.75,
                                  np.zeros_like(hist_img), 0.25, 0)

        # X-axis labels
        for x in range(0, 256, 64):
            px = int(x * hist_w / 256)
            cv2.putText(blended, str(x), (px, hist_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        cv2.putText(blended, "Colour Histogram", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # Legend
        legend_y = 40
        for label, colour in zip(ch_labels, colours):
            cv2.putText(blended, f"  {label}", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
            legend_y += 18

        self._display_result(blended)
        self._set_status("Colour Histogram displayed — channels: " + ", ".join(ch_labels))


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Horizontal.TScale",
                    background="#16213e",
                    troughcolor="#0f3460",
                    sliderthickness=14)
    style.configure("Vertical.TScrollbar",
                    background="#0f3460",
                    troughcolor="#16213e",
                    arrowcolor="#e94560")

    app = ImageProcessorApp(root)
    root.mainloop()