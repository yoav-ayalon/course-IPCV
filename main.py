import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import scipy.stats as stats
import cv2
# from skimage import exposure
from skimage import filters, exposure
import os
from skimage.io import imread, imsave
from skimage.color import rgb2hsv, hsv2rgb
# from skimage.util import img_as_float
from typing import List, Union
from pathlib import Path
import scipy.ndimage as ndi
#scikit-image==0.24.0
#opencv-python


script_dir = Path(__file__).resolve().parent
images_dir = script_dir/"IMG"
model_dir = script_dir/"model"
rect_start = None
rect_end = None
drawing = False


def show_image(img_or_list: Union[np.ndarray, List[np.ndarray]],
               row_plot: int = 1,
               titles: List[str] = None):
    """
    Display one or multiple images in a grid using Matplotlib.
    - img_or_list: single image or list of images (H×W or H×W×3)
    - row_plot: number of rows in the grid
    - titles: optional list of titles, same length as images
    """
    imgs = img_or_list if isinstance(img_or_list, list) else [img_or_list]
    n   = len(imgs)
    rows = row_plot
    cols = int(np.ceil(n / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    for i, im in enumerate(imgs):
        if im.ndim == 2:
            axes[i].imshow(im, cmap='gray')
        else:
            axes[i].imshow(im)
        axes[i].axis('off')
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def load_image(name: str):
    path = images_dir / name

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Read raw bytes with numpy (handles Unicode paths fine)
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise IOError(f"Failed to read any data from: {path}")

    # Decode the image from memory
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cv2.imdecode failed for: {path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, gray, rgb


def detect_corners_ShiTomasi(gray, max_corner=150):

    # maxCorners, qualityLevel, minDistance
        corners = cv2.goodFeaturesToTrack(gray,
                                        maxCorners=max_corner,
                                        qualityLevel=0.01,
                                        minDistance=10)
        corners = np.intp(corners)  # integer coordinats


        if corners is None or len(corners) == 0:
            print("No corners found in ROI.")
            return None
        else:
            corners = np.array(corners, dtype=np.float32)
            if corners.ndim == 3:   # shape (N, 1, 2)
                corners = corners.reshape(-1, 2) # shape (N, 2)

        return corners


def orb_features(img_bgr):
    """
    Compute ORB keypoints & descriptors for a BGR image.
    Returns (rgb, keypoints, descriptors).
    """
    # gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = rgb2hsv(img_bgr)[...,2]
    gray = (gray * 255).astype(np.uint8)
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    orb = cv2.ORB_create(
        nfeatures=500,        # max keypoints
        scaleFactor=1.2,
        nlevels=8,
    )
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return rgb, keypoints, descriptors


def select_roi_on_rgb(rgb, window_width=1000, window_height=600):
    """
    Allow user to select a rectangular ROI on the RGB image using mouse.
    The window size is constant, and the image is scaled to fit entirely within it.
    Returns (rect_coords, roi_rgb) where rect_coords = (x1, y1, x2, y2).
    If no ROI selected, returns (None, None).
    """
    global rect_start, rect_end, drawing
    rect_start = None
    rect_end = None
    drawing = False

    win_name = "Select ROI (drag with mouse, ENTER/ESC to finish)"

    h, w = rgb.shape[:2]
    # Always scale to fit the window, whether image is small or large
    scale = min(window_width / w, window_height / h)

    disp_w, disp_h = int(w * scale), int(h * scale)
    rgb_disp = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    def mouse_callback(event, x, y, flags, param):
        global rect_start, rect_end, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect_start = (x, y)
            rect_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect_end = (x, y)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    while True:
        # Display a copy of the image, with a rectangle if present
        frame = rgb_disp.copy() 
        if rect_start and rect_end:
            cv2.rectangle(frame, rect_start, rect_end, (0, 255, 0), 2)

        # OpenCV works in BGR, so convert only for display purposes
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(win_name, frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (13, 27): # ENTER or ESC to finish
            break

    cv2.destroyWindow(win_name)

    if not (rect_start and rect_end):
        return None, None

    x1, y1 = rect_start
    x2, y2 = rect_end
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    orig_x1 = int(x1 / scale)
    orig_y1 = int(y1 / scale)
    orig_x2 = int(x2 / scale)
    orig_y2 = int(y2 / scale)

    roi_rgb = rgb[orig_y1:orig_y2, orig_x1:orig_x2]

    return (orig_x1, orig_y1, orig_x2, orig_y2), roi_rgb


def translate_corners_to_global(corners, rect_x1, rect_y1):
    global_points = []
    for (cx, cy) in corners:
            gx = int(cx + rect_x1)
            gy = int(cy + rect_y1)
            global_points.append((gx, gy))
    return global_points


def debug_draw_corners(rgb_debug, global_points):
    for (gx, gy) in global_points:
        cv2.circle(rgb_debug, (gx, gy), 4, (0, 255, 0), -1)
    return rgb_debug


def create_blurred_mask(rgb, global_points, pixelation_strength=5, bit_depth=3, show_histogram:bool=True, verbose:bool=False):
    """
    Create irreversible anonymization by pixelation and color quantization.
    
    Args:
        rgb: Input RGB image
        global_points: Points defining the ROI
        pixelation_strength: Size in pixels for the shorter side after downsampling (lower = stronger blur)
                           Range: 5-20 (5=very strong, 20=mild)
        bit_depth: Bits per channel for color quantization (lower = stronger blur)
                  Range: 3-6 (3=very strong, 6=mild)
    
    Returns:
        mask: Binary mask of the ROI
        anonymized_full: Full image with anonymized ROI applied
    """
    h, w = rgb.shape[:2] # height, width
    mask = np.zeros((h, w), dtype=np.uint8) # binary mask
    pts = np.array(global_points, dtype=np.int32) 
    hull = cv2.convexHull(pts) # compute convex hull
    cv2.fillConvexPoly(mask, hull, 255) # fill hull area in mask

    # Extract bounding box of the ROI
    x, y, roi_w, roi_h = cv2.boundingRect(hull)
    
    # Extract the ROI region
    roi = rgb[y:y+roi_h, x:x+roi_w].copy()
    roi_mask = mask[y:y+roi_h, x:x+roi_w]
    
    # Step 1: Aggressive downsampling (pixelation)
    # Calculate new size based on pixelation_strength
    shorter_side = min(roi_h, roi_w)
    if shorter_side > 0:
        scale = pixelation_strength / shorter_side
        new_w = max(1, int(roi_w * scale))
        new_h = max(1, int(roi_h * scale))
        
        # Downsample aggressively
        downsampled = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Step 2: Color quantization (reduce bit depth)
        # Reduce from 8 bits to bit_depth bits per channel
        levels = 2 ** bit_depth
        quantized = (downsampled // (256 // levels)) * (256 // levels)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        # Step 3: Upsample back with nearest-neighbor (creates blocky effect)
        pixelated_roi = cv2.resize(quantized, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        
        if show_histogram:
            # Visualize the histogram of original vs pixelated ROI
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            # Original ROI histograms
            for i, (color, name) in enumerate(zip(['red', 'green', 'blue'], ['Red', 'Green', 'Blue'])):
                axes[0, i].hist(roi[:, :, i].ravel(), bins=256, range=(0, 256), 
                            color=color, alpha=0.7, edgecolor='black')
                axes[0, i].set_title(f'Original {name} Channel')
                axes[0, i].set_xlabel('Pixel Value')
                axes[0, i].set_ylabel('Frequency')
                axes[0, i].set_xlim(0, 255)
            
            # Pixelated ROI histograms (showing quantization effect)
            for i, (color, name) in enumerate(zip(['red', 'green', 'blue'], ['Red', 'Green', 'Blue'])):
                axes[1, i].hist(pixelated_roi[:, :, i].ravel(), bins=256, range=(0, 256), 
                            color=color, alpha=0.7, edgecolor='black')
                axes[1, i].set_title(f'Anonymized {name} Channel (quantized)')
                axes[1, i].set_xlabel('Pixel Value')
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].set_xlim(0, 255)
            
            plt.suptitle(f'ROI Histogram Comparison\nPixelation: {pixelation_strength}px, Bit depth: {bit_depth} bits', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        if verbose:
            print(f"Anonymization applied:")
            print(f"  - Original ROI size: {roi_w}x{roi_h}")
            print(f"  - Downsampled to: {new_w}x{new_h} (scale: {scale:.3f})")
            print(f"  - Color levels per channel: {levels} (from {bit_depth} bits)")
            print(f"  - Total unique colors: {len(np.unique(quantized.reshape(-1, 3), axis=0))} (theoretical max: {levels**3})")
    else:
        pixelated_roi = roi
    
    # Apply the pixelated ROI back to the full image
    anonymized_full = rgb.copy()
    # Only replace pixels within the actual mask (convex hull)
    mask_indices = roi_mask == 255
    anonymized_full[y:y+roi_h, x:x+roi_w][mask_indices] = pixelated_roi[mask_indices]
    
    return mask, anonymized_full


def mean_blur(gray, rgb, global_points):
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array(global_points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    # 31x31 mean blur (strong blur, must be odd and > 1)
    blurred_full = cv2.boxFilter(rgb, ddepth=-1, ksize=(31, 31), normalize=True)

    return mask, blurred_full


def binary_mask(roi_gray):
    t_otsu = filters.threshold_otsu(roi_gray)
    roi_binary_mask = (roi_gray > t_otsu).astype(np.uint8) * 255
    #show_image([roi_gray, roi_binary_mask], row_plot=1)
    # roi_binary_mask: 0/255 uint8 mask from above
    return cv2.bitwise_and(roi_gray, roi_gray, mask=roi_binary_mask)


def _plot_histogram(ax, image, alpha=0.3, **kwargs):
    hist, bin_centers = exposure.histogram(image)
    ax.fill_between(bin_centers, hist, alpha=alpha, **kwargs)
    ax.set_xlabel('intensity')
    ax.set_ylabel('# pixels')


def iter_channels(color_image):
    for channel in np.rollaxis(color_image, -1):
        yield channel


def plot_histogram(image, ax=None, **kwargs):
    ax = ax if ax is not None else plt.gca()
    if image.ndim == 2:
        _plot_histogram(ax, image, color='black', **kwargs)
    elif image.ndim == 3:
        for channel, channel_color in zip(iter_channels(image), 'rgb'):
            _plot_histogram(ax, channel, color=channel_color, **kwargs)


def match_axes_height(ax_src, ax_dst):
    plt.draw()
    dst = ax_dst.get_position()
    src = ax_src.get_position()
    ax_dst.set_position([dst.xmin, src.ymin, dst.width, src.height])


def imshow_with_histogram(image, **kwargs):
    """
    Show image and its histogram side-by-side.
    """
    width, height = plt.rcParams['figure.figsize']
    fig, (ax_image, ax_hist) = plt.subplots(ncols=2, figsize=(2 * width, height))
    kwargs.setdefault('cmap', plt.cm.gray)
    ax_image.imshow(image, **kwargs)
    plot_histogram(image, ax=ax_hist)
    ax_image.set_axis_off()
    match_axes_height(ax_image, ax_hist)
    return ax_image, ax_hist


def clahe(roi_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    he_clahe = clahe.apply(roi_gray)
    ax_img, ax_hist = imshow_with_histogram(he_clahe)
    show_image([roi_gray, he_clahe], row_plot=1, titles=["Original", "CLAHE (local equalization)"],)
    return he_clahe


def unsharp_masking(roi_gray):
    blur = ndi.gaussian_filter(roi_gray, sigma=1.0)
    # High-frequency component (edges/details)
    high_freq = roi_gray - blur
    # Sharpen: original + α * high_freq
    alpha = 0.5
    sharp = np.clip(roi_gray + alpha * high_freq, 0, 1)
    show_image([roi_gray, blur, high_freq, sharp], row_plot=1, titles=["Original", "Blurred", "High freq (a - blur)", "Sharpened (unsharp mask)"])
    return high_freq


def custom_kernel(roi_gray):
    kernel_sharp = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]], np.float32)
    sharp = cv2.filter2D(roi_gray, -1, kernel_sharp)
    show_image([sharp], titles=["sharp"], row_plot=1)
    return sharp


def sobel(roi_gray):
    # Sobel gradients (derivatives)
    sx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sx, sy)

    show_image([sx, sy, sobel_mag], row_plot=1,
           titles='Sobel X | Sobel Y | Magnitude')
    
    return sobel_mag


def laplacian(roi_gray):
    # Laplacian (second derivative)
    lap = cv2.Laplacian(roi_gray, cv2.CV_32F, ksize=3)

    show_image([lap], row_plot=1, titles='Laplacian')
    return lap


def canny(roi_gray):
    # Canny (works on 8-bit)
    g8 = (roi_gray*255).astype(np.uint8)
    edges = cv2.Canny(g8, threshold1=30, threshold2=90)

    show_image([edges], row_plot=1, titles='Canny')
    return edges


def diliation(roi_gray):
    k7  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(roi_gray, k7, iterations=1)
    show_image(
    [roi_gray, dilated], row_plot=1,
    titles=["Binary", "Dilate (3×3) – grows strokes, fills small gaps, makes much noise in the backround"]
)
    return dilated


def Morphological_gradient(roi_gray):
    # Choose a structuring element
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    # Dilate the binary mask
    dilated = cv2.dilate(roi_gray, k)

    # Subtract the original binary image from the dilated version
    # → the difference is only the NEW outer pixels that appeared
    outer = cv2.absdiff(dilated, roi_gray)

    # Visualize
    show_image([roi_gray, outer],
                titles=["Binary image", "Outer contour (from dilation)"])
    
    return outer


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a boolean mask on RGB (mask=True highlighted)."""
    out = rgb.copy()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    # highlight mask region in red-ish overlay (no fixed colormap assumptions)
    highlight = np.zeros_like(out)
    highlight[..., 0] = 255  # R channel
    out[mask] = (alpha * highlight[mask] + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


def seg_start(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1) Otsu binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2) Noise removal (opening)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # 3) Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4) Sure foreground via distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # 5) Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6) Markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 7) Watershed
    markers_ws = cv2.watershed(bgr.copy(), markers.copy())

    # boundaries marked as -1
    vis = img.copy()
    vis[markers_ws == -1] = [255, 0, 0]  # red boundary

    print(f"Number of segments found: {num_labels}")

    show_image(
        [img, thresh, opening, dist, sure_fg, vis],
        row_plot=2,
        titles=[
            "Input", "Otsu thresh", "Opening", "Distance transform",
            "Sure FG markers", "Watershed boundaries overlay"])


def SAM_segmentation(img):
    """
    Perform automatic segmentation using Segment Anything Model (SAM).
    
    Requirements:
    - Download SAM model checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints
    - Place it in: model/checkpoints/sam_vit_b_01ec64.pth
    """
    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install torch torchvision segment-anything")
        return

    # Path to SAM checkpoint
    SAM_CKPT = model_dir / "checkpoints" / "sam_vit_b_01ec64.pth"
    
    if not SAM_CKPT.exists():
        print(f"Checkpoint not found: {SAM_CKPT}")
        print("\nDownload from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print(f"Place it in: {SAM_CKPT.parent}")
        # Create directory if it doesn't exist
        SAM_CKPT.parent.mkdir(parents=True, exist_ok=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    sam.to(device=device)

    print("✔️ SAM loaded on:", device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=4,       # speed vs quality. Try 8, 16, etc. it is amount of prompts in a power of two. (4^2 = 16 prompts). So 4 is approx 16 times faster than 16.
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=200)

    # Convert to RGB uint8 if needed
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    masks = mask_generator.generate(img)
    print("Masks generated:", len(masks))

    # Sort masks by area (largest first)
    masks_sorted = sorted(masks, key=lambda x: x["area"], reverse=True)

    # Visualize top N masks with distinct colors
    N = 8
    # Define distinct colors for each mask
    distinct_colors = [
        [0, 0, 255],     # Blue
        [0, 255, 0],     # Green
        [255, 0, 0],     # Red
        [255, 255, 0],   # Yellow
        [255, 0, 255],   # Magenta
        [0, 255, 255],   # Cyan
        [255, 128, 0],   # Orange
        [128, 0, 255],   # Purple
    ]
    
    vis = img.copy().astype(np.float32)
    for i in range(min(N, len(masks_sorted))):
        m = masks_sorted[i]["segmentation"]
        color = np.array(distinct_colors[i % len(distinct_colors)], dtype=np.float32)
        vis[m] = vis[m] * 0.65 + color * 0.35
    
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    
    show_image([img, vis], row_plot=1, titles=["Original", f"SAM automatic masks (top {min(N,len(masks_sorted))} overlaid with distinct colors)"])


def grabcut_segmentation(roi_rgb, iterations=5, show_steps=True):
    """
    GrabCut segmentation: Estimates foreground within ROI image.
    
    Args:
        roi_rgb: ROI RGB image (extracted region)
        iterations: Number of GrabCut iterations (default=5)
        show_steps: Whether to show intermediate visualization (default=True)
    
    Returns:
        mask: Binary mask (0/255) of the segmented object within ROI
        final_vis: Visualization with mask overlay
    """
    h, w = roi_rgb.shape[:2]
    # Use entire ROI as bounding box with small margin
    margin = 5
    rect_grabcut = (margin, margin, w - 2*margin, h - 2*margin)  # (x, y, w, h) format
    
    # Initialize mask and models for GrabCut
    mask_grabcut = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    
    # Run GrabCut
    cv2.grabCut(roi_rgb, mask_grabcut, rect_grabcut, bgd_model, fgd_model, 
                iterations, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask (foreground = 1 or 3, background = 0 or 2)
    mask_fg = np.where((mask_grabcut == cv2.GC_FGD) | (mask_grabcut == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    # Keep only largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_fg, connectivity=8)
    if num_labels > 1:  # 0 is background
        # Find largest component (excluding background)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_largest = (labels == largest_component).astype(np.uint8) * 255
    else:
        mask_largest = mask_fg
    
    # Morphological cleanup (closing then opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask_largest, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Visualization
    vis_grabcut = overlay_mask(roi_rgb, mask_fg.astype(bool), alpha=0.4)
    vis_largest = overlay_mask(roi_rgb, mask_largest.astype(bool), alpha=0.4)
    vis_final = overlay_mask(roi_rgb, mask_cleaned.astype(bool), alpha=0.4)
    
    if show_steps:
        show_image(
            [roi_rgb, vis_grabcut, vis_largest, vis_final],
            row_plot=2,
            titles=[
                "Input ROI",
                "GrabCut raw output",
                "Largest component only",
                "After morphological cleanup"
            ]
        )
    
    print(f"GrabCut: Found {num_labels - 1} components, kept largest")
    return mask_cleaned, vis_final


def sam_box_prompt_segmentation(roi_rgb, show_steps=True):
    """
    SAM with box prompt: Use entire ROI as box prompt, select best mask automatically.
    
    Args:
        roi_rgb: ROI RGB image (extracted region)
        show_steps: Whether to show intermediate visualization (default=True)
    
    Returns:
        mask: Binary mask (0/255) of the selected object within ROI
        final_vis: Visualization with mask overlay
    """
    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install torch torchvision segment-anything")
        return None, None

    # Path to SAM checkpoint
    SAM_CKPT = model_dir / "checkpoints" / "sam_vit_b_01ec64.pth"
    
    if not SAM_CKPT.exists():
        print(f"Checkpoint not found: {SAM_CKPT}")
        print("\nDownload from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        SAM_CKPT.parent.mkdir(parents=True, exist_ok=True)
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    
    # Convert to RGB uint8 if needed
    if roi_rgb.dtype != np.uint8:
        roi_rgb_uint8 = (roi_rgb * 255).astype(np.uint8)
    else:
        roi_rgb_uint8 = roi_rgb
    
    predictor.set_image(roi_rgb_uint8)
    
    # Create box prompt for entire ROI with small margin
    h, w = roi_rgb.shape[:2]
    margin = 5
    box_prompt = np.array([margin, margin, w - margin, h - margin])
    
    # Generate masks with box prompt
    masks, scores, logits = predictor.predict(
        box=box_prompt,
        multimask_output=True  # Get 3 mask candidates
    )
    
    print(f"SAM box prompt: Generated {len(masks)} candidate masks")
    print(f"Scores: {scores}")
    
    # Score each mask based on:
    # 1. Area in ROI
    # 2. Centroid distance from ROI center
    # 3. Penalty for touching ROI borders
    h, w = roi_rgb.shape[:2]
    roi_center_x, roi_center_y = w / 2, h / 2
    roi_area = h * w
    
    best_idx = -1
    best_score = -np.inf
    
    mask_scores = []
    for i, mask in enumerate(masks):
        # Area score (normalized)
        area_in_roi = np.sum(mask)
        area_score = area_in_roi / roi_area
        
        # Centroid score
        if np.sum(mask) > 0:
            y_coords, x_coords = np.where(mask)
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            dist_to_center = np.sqrt((centroid_x - roi_center_x)**2 + (centroid_y - roi_center_y)**2)
            max_dist = np.sqrt(w**2 + h**2) / 2
            centroid_score = 1 - (dist_to_center / max_dist)
        else:
            centroid_score = 0
        
        # Border penalty (if mask touches ROI edges)
        border_penalty = 0
        if np.any(mask[0, :]) or np.any(mask[-1, :]) or \
           np.any(mask[:, 0]) or np.any(mask[:, -1]):
            border_penalty = 0.2
        
        # Combined score
        combined = scores[i] * 0.4 + area_score * 0.3 + centroid_score * 0.3 - border_penalty
        mask_scores.append(combined)
        
        if combined > best_score:
            best_score = combined
            best_idx = i
    
    print(f"Combined scores: {mask_scores}")
    print(f"Selected mask {best_idx} with score {best_score:.3f}")
    
    # Get best mask
    best_mask = masks[best_idx].astype(np.uint8) * 255
    
    # Keep only largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(best_mask, connectivity=8)
    if num_labels > 1:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_largest = (labels == largest_component).astype(np.uint8) * 255
    else:
        mask_largest = best_mask
    
    # Light morphological smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_final = cv2.morphologyEx(mask_largest, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Visualization
    if show_steps:
        vis_candidates = []
        for i, mask in enumerate(masks):
            vis = overlay_mask(roi_rgb_uint8, mask.astype(bool), alpha=0.4)
            # cv2.putText(vis, f"Score: {mask_scores[i]:.3f}", (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            vis_candidates.append(vis)
        
        vis_final = overlay_mask(roi_rgb_uint8, mask_final.astype(bool), alpha=0.4)
        
        show_image(
            [roi_rgb_uint8] + vis_candidates + [vis_final],
            row_plot=2,
            titles=[
                "Input ROI",
                f"Candidate 1 (SAM score: {scores[0]:.3f})",
                f"Candidate 2 (SAM score: {scores[1]:.3f})",
                f"Candidate 3 (SAM score: {scores[2]:.3f})",
                f"Selected & refined (best combined score)"
            ]
        )
    
    return mask_final, overlay_mask(roi_rgb_uint8, mask_final.astype(bool), alpha=0.4)


def watershed_roi_segmentation(roi_rgb, distance_threshold=0.3, show_steps=True):
    """
    Watershed segmentation within ROI: Use distance transform for seeds, select best component.
    
    Args:
        roi_rgb: ROI RGB image (extracted region)
        distance_threshold: Threshold multiplier for distance transform (0.1-0.5, default=0.3)
        show_steps: Whether to show intermediate visualization (default=True)
    
    Returns:
        mask: Binary mask (0/255) of the selected object within ROI
        final_vis: Visualization with mask overlay
    """
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    
    # Otsu thresholding
    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background (dilate)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Sure foreground via distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, distance_threshold * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Add 1 so background is not 0 but 1
    markers[unknown == 255] = 0  # Mark unknown region as 0
    
    # Watershed
    roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    markers_ws = cv2.watershed(roi_bgr, markers.copy())
    
    print(f"Watershed: Found {num_labels} initial components")
    
    # Score each component (exclude background=1 and boundary=-1)
    roi_h, roi_w = roi_rgb.shape[:2]
    roi_center_y, roi_center_x = roi_h / 2, roi_w / 2
    
    best_component = -1
    best_score = -np.inf
    
    component_scores = []
    for label in range(2, num_labels + 1):  # Start from 2 (1 is background)
        component_mask = (markers_ws == label)
        
        # Area score
        area = np.sum(component_mask)
        area_score = area / (roi_h * roi_w)
        
        # Centroid distance score
        if area > 0:
            y_coords, x_coords = np.where(component_mask)
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            dist_to_center = np.sqrt((centroid_x - roi_center_x)**2 + (centroid_y - roi_center_y)**2)
            max_dist = np.sqrt(roi_w**2 + roi_h**2) / 2
            centroid_score = 1 - (dist_to_center / max_dist)
        else:
            centroid_score = 0
        
        # Combined score
        combined = area_score * 0.6 + centroid_score * 0.4
        component_scores.append((label, combined, area))
        
        if combined > best_score:
            best_score = combined
            best_component = label
    
    print(f"Component scores: {[(l, f'{s:.3f}') for l, s, _ in component_scores]}")
    print(f"Selected component {best_component} with score {best_score:.3f}")
    
    # Create mask for best component
    if best_component > 0:
        roi_mask = (markers_ws == best_component).astype(np.uint8) * 255
    else:
        roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_final = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Visualization
    if show_steps:
        # Watershed boundaries visualization
        vis_watershed = roi_rgb.copy()
        vis_watershed[markers_ws == -1] = [255, 0, 0]  # Red boundaries
        
        # Distance transform normalized for display
        dist_vis = (dist_transform / dist_transform.max() * 255).astype(np.uint8)
        
        # Final mask overlay
        vis_final = overlay_mask(roi_rgb, mask_final.astype(bool), alpha=0.4)
        
        show_image(
            [roi_rgb, thresh, opening, dist_vis, sure_fg, vis_watershed, vis_final],
            row_plot=2,
            titles=[
                "ROI input",
                "Otsu threshold",
                "Opening (noise removal)",
                "Distance transform",
                f"Sure FG (thresh={distance_threshold})",
                "Watershed boundaries",
                "Final selected component"
            ]
        )
    
    return mask_final, overlay_mask(roi_rgb, mask_final.astype(bool), alpha=0.4)


if __name__ == "__main__":
    
    bgr, gray, rgb = load_image("IMG_004.jpeg")

    (rect_x1, rect_y1, rect_x2, rect_y2), roi_rgb = select_roi_on_rgb(rgb)
    print(f"Selected rectangle: ({rect_x1}, {rect_y1}) to ({rect_x2}, {rect_y2})")
    
    if roi_rgb is not None:
        show_image(roi_rgb, titles=["Selected ROI"], row_plot=1)
    
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

        # Binary than grey mask
        # roi_gray_masked = binary_mask(roi_gray)
        # corners = detect_corners_ShiTomasi(roi_gray_masked, max_corner=30)

        # Dilation 
        # dilated = diliation(roi_gray)
        # corners = detect_corners_ShiTomasi(dilated, max_corner=30)

        # Morphological gradient
        # outer = Morphological_gradient(roi_gray)
        # corners = detect_corners_ShiTomasi(outer, max_corner=30)

        # Apply CLAHE on grayscale image
        # he_clahe = clahe(roi_gray)
        # corners = detect_corners_ShiTomasi(he_clahe, max_corner=30)

        # Unsharp masking (blur → subtract → sharpen)
        sharpen = unsharp_masking(roi_gray)
        corners = detect_corners_ShiTomasi(sharpen, max_corner=30)

        # seg_start(roi_rgb)
        # SAM_segmentation(roi_rgb)

        # ========== NEW SEGMENTATION METHODS ==========
        # Uncomment to try different segmentation approaches:
        
        # 1) GrabCut segmentation (foreground/background separation)
        mask_grabcut, vis_grabcut = grabcut_segmentation(roi_rgb, iterations=5, show_steps=True)
        
        # 2) SAM with box prompt (automatic mask selection)
        mask_sam_box, vis_sam_box = sam_box_prompt_segmentation(roi_rgb, show_steps=True)
        
        # 3) Watershed segmentation (distance transform seeds)
        mask_watershed, vis_watershed = watershed_roi_segmentation(roi_rgb, distance_threshold=0.3, show_steps=True)
        # ==============================================

        # sharp = custom_kernel(roi_gray)
        # corners = detect_corners_ShiTomasi(sharp, max_corner=30)

        # Sobel gradients (derivatives)
        # sobel_mag = sobel(roi_gray)
        # corners = detect_corners_ShiTomasi(sobel_mag, max_corner=30)

        # Laplacian (second derivative)
        # lap = laplacian(roi_gray)
        # corners = detect_corners_ShiTomasi(lap, max_corner=30)

        # Canny (works on 8-bit)
        # edges = canny(roi_gray)
        # corners = detect_corners_ShiTomasi(edges, max_corner=30)

        # corners = detect_corners_ShiTomasi(roi_gray, max_corner=30)
        global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        
        rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        show_image([rgb_debug], titles=["Corners debug"], row_plot=1)

        mask, blurred_full = create_blurred_mask(rgb, global_points, pixelation_strength=13, bit_depth=5, show_histogram=True, verbose=True)
        # mask, blurred_full = mean_blur(gray, rgb, global_points)

        result = rgb.copy()
        result[mask == 255] = blurred_full[mask == 255]
        show_image([result], titles=["Object blurred"], row_plot=1)


