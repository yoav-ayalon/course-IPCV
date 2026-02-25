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
video_dir = script_dir/"VID"
model_dir = script_dir/"model"
rect_start = None
rect_end = None
drawing = False


###----------------------------------- Initialization ----------------------------------------  

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


def load_video(name: str):
    """
    Load a video file and return a VideoCapture object with metadata.
    
    Args:
        name: Video filename (e.g., "my_video.mp4")
    
    Returns:
        cap: cv2.VideoCapture object
        metadata: dict with 'fps', 'width', 'height', 'frame_count', 'duration'
    """
    path = video_dir / name
    
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    
    # Open video file
    cap = cv2.VideoCapture(str(path))
    
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {path}")
    
    # Extract metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    metadata = {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration': duration
    }
    
    print(f"Video loaded: {name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {duration:.2f} seconds")
    
    return cap, metadata


def read_video_frame(cap):
    """
    Read next frame from video.
    
    Args:
        cap: cv2.VideoCapture object
    
    Returns:
        success: Boolean indicating if frame was read successfully
        bgr: Frame in BGR format (or None if failed)
        gray: Frame in grayscale (or None if failed)
        rgb: Frame in RGB format (or None if failed)
    """
    ret, bgr = cap.read()
    
    if not ret or bgr is None:
        return False, None, None, None
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    return True, bgr, gray, rgb


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


def select_mode():
    """
    Prompt user to select between image and video processing modes.
    
    Returns:
        mode: 'image' or 'video'
    """
    print("\n" + "="*60)
    print("  IMAGE/VIDEO ANONYMIZATION TOOL")
    print("="*60)
    print("\nPlease select a processing mode:\n")
    print("  1. Image mode - Process a single image")
    print("  2. Video mode - Process a video with object tracking")
    print()
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            return 'image'
        elif choice == '2':
            return 'video'
        else:
            print("Invalid choice. Please enter 1 or 2.")


def select_tracker():
    """
    Prompt user to select tracking method for video processing.
    
    Returns:
        tracker: Instantiated MaskTracker (FlowWarpTracker or KLTTracker)
    """
    print("\n" + "="*60)
    print("  SELECT TRACKING METHOD")
    print("="*60)
    print("\nAvailable tracking methods:\n")
    print("  1. Dense Optical Flow - Pixel-wise warping (Farneback algorithm)")
    print("  2. KLT Feature Tracking - Global transform (feature points)")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            print("\nUsing FlowWarpTracker (Dense Optical Flow)")
            return FlowWarpTracker(
                roi_margin=30,
                min_mask_area=300,
                morph_kernel_size=5,
                morph_close_iterations=2,
                morph_open_iterations=1
            )
        elif choice == '2':
            print("\nUsing KLTTracker (Feature Point Tracking)")
            return KLTTracker(
                max_points=150,
                quality_level=0.01,
                min_distance=10,
                reinit_threshold=0.3,
                refine_kernel_size=5,
                refine_iterations=1
            )
        else:
            print("Invalid choice. Please enter 1 or 2.")


def select_segmentation_method():
    """
    Prompt user to select initial segmentation method for first frame.
    
    Returns:
        method: 'sam' or 'corners'
    """
    print("\n" + "="*60)
    print("  SELECT SEGMENTATION METHOD")
    print("="*60)
    print("\nChoose initial mask generation method:\n")
    print("  1. SAM (Segment Anything Model)")
    print("  2. Corner Detection (Shi-Tomasi) + Convex Hull")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            return 'sam'
        elif choice == '2':
            return 'corners'
        else:
            print("Invalid choice. Please enter 1 or 2.")


###----------------------------------- Helper funcations ----------------------------------------  

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


def _imshow_with_histogram(image, **kwargs):
    """
    Show image and its histogram side-by-side.
    """

    def _iter_channels(color_image):
        for channel in np.rollaxis(color_image, -1):
            yield channel
    def _hist(ax, image, alpha=0.3, **kwargs):
        hist, bin_centers = exposure.histogram(image)
        ax.fill_between(bin_centers, hist, alpha=alpha, **kwargs)
        ax.set_xlabel('intensity')
        ax.set_ylabel('# pixels')
    def _plot_histogram(image, ax=None, **kwargs):
        ax = ax if ax is not None else plt.gca()
        if image.ndim == 2:
            _hist(ax, image, color='black', **kwargs)
        elif image.ndim == 3:
            for channel, channel_color in zip(_iter_channels(image), 'rgb'):
                _hist(ax, channel, color=channel_color, **kwargs)
    def _match_axes_height(ax_src, ax_dst):
        plt.draw()
        dst = ax_dst.get_position()
        src = ax_src.get_position()
        ax_dst.set_position([dst.xmin, src.ymin, dst.width, src.height])


    width, height = plt.rcParams['figure.figsize']
    fig, (ax_image, ax_hist) = plt.subplots(ncols=2, figsize=(2 * width, height))
    kwargs.setdefault('cmap', plt.cm.gray)
    ax_image.imshow(image, **kwargs)
    _plot_histogram(image, ax=ax_hist)
    ax_image.set_axis_off()
    _match_axes_height(ax_image, ax_hist)
    return ax_image, ax_hist


###----------------------------------- first image processing functions -------------------------------  

def binary_mask(roi_gray):
    t_otsu = filters.threshold_otsu(roi_gray)
    roi_binary_mask = (roi_gray > t_otsu).astype(np.uint8) * 255
    #show_image([roi_gray, roi_binary_mask], row_plot=1)
    # roi_binary_mask: 0/255 uint8 mask from above
    return cv2.bitwise_and(roi_gray, roi_gray, mask=roi_binary_mask)


def diliation(roi_gray):
    k7  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(roi_gray, k7, iterations=1)
    show_image(
    [roi_gray, dilated], row_plot=1,
    titles=["Binary", "Dilate (3×3) – grows strokes, fills small gaps, makes much noise in the backround"])
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


def clahe(roi_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    he_clahe = clahe.apply(roi_gray)
    ax_img, ax_hist = _imshow_with_histogram(he_clahe)
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



###----------------------------------- second image processing functions ------------------------------

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


###----------------------------------- segmentation functions -------------------------------

def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a boolean mask on RGB (mask=True highlighted)."""
    out = rgb.copy()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    # highlight mask region in red-ish overlay (no fixed colormap assumptions)
    highlight = np.zeros_like(out)
    highlight[..., 0] = 255  # R channel
    out[mask] = (alpha * highlight[mask] + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


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
    vis_grabcut = _overlay_mask(roi_rgb, mask_fg.astype(bool), alpha=0.4)
    vis_largest = _overlay_mask(roi_rgb, mask_largest.astype(bool), alpha=0.4)
    vis_final = _overlay_mask(roi_rgb, mask_cleaned.astype(bool), alpha=0.4)
    
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
            vis = _overlay_mask(roi_rgb_uint8, mask.astype(bool), alpha=0.4)
            # cv2.putText(vis, f"Score: {mask_scores[i]:.3f}", (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            vis_candidates.append(vis)
        
        vis_final = _overlay_mask(roi_rgb_uint8, mask_final.astype(bool), alpha=0.4)
        
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
    
    return mask_final, _overlay_mask(roi_rgb_uint8, mask_final.astype(bool), alpha=0.4)


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
        vis_final = _overlay_mask(roi_rgb, mask_final.astype(bool), alpha=0.4)
        
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
    
    return mask_final, _overlay_mask(roi_rgb, mask_final.astype(bool), alpha=0.4)



###----------------------------------- Video Tracking Functions -------------------------------

class MaskTracker:
    """
    Base class for mask tracking across video frames.
    
    All trackers follow the same interface:
    - init(frame, mask): Initialize tracking from first frame
    - update(prev_frame, curr_frame, prev_mask): Track mask to current frame
    """
    
    def init(self, frame, mask):
        """
        Initialize tracker with first frame and mask.
        
        Args:
            frame: First frame (grayscale uint8)
            mask: Initial binary mask (0/255 uint8)
        """
        raise NotImplementedError
    
    def update(self, prev_frame, curr_frame, prev_mask):
        """
        Update mask from previous frame to current frame.
        
        Args:
            prev_frame: Previous frame (grayscale uint8)
            curr_frame: Current frame (grayscale uint8)
            prev_mask: Previous binary mask (0/255 uint8)
        
        Returns:
            curr_mask: Updated binary mask (0/255 uint8)
            stats: Dictionary with tracking metrics
        """
        raise NotImplementedError
    
    def get_name(self):
        """Return tracker name for logging."""
        return self.__class__.__name__


class KLTTracker(MaskTracker):
    """
    Track mask using KLT (Kanade-Lucas-Tomasi) feature point tracking.
    
    Tracks a sparse set of feature points on the object, estimates a global
    similarity transform (translation + rotation + uniform scale), and warps
    the entire mask as one unit. More stable than dense flow for rigid objects.
    """
    
    def __init__(self, 
                 max_points=150,
                 quality_level=0.01,
                 min_distance=10,
                 reinit_threshold=0.3,
                 refine_kernel_size=5,
                 refine_iterations=1):
        """
        Args:
            max_points: Maximum feature points to track
            quality_level: Quality threshold for goodFeaturesToTrack (0-1)
            min_distance: Minimum distance between feature points (pixels)
            reinit_threshold: Reinitialize when points drop below this fraction
            refine_kernel_size: Morphology kernel size for cleanup
            refine_iterations: Morphology iterations (close+open)
        """
        self.max_points = max_points
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.reinit_threshold = reinit_threshold
        self.refine_kernel_size = refine_kernel_size
        self.refine_iterations = refine_iterations
        
        self.initial_points = None
        self.prev_points = None
        self.num_initial_points = 0
        
        # KLT parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    
    def init(self, frame, mask):
        """Initialize feature points inside the mask."""
        self._detect_points(frame, mask)
        self.num_initial_points = len(self.prev_points) if self.prev_points is not None else 0
    
    def _detect_points(self, frame, mask):
        """Detect feature points inside the mask region."""
        # Detect features only inside mask
        points = cv2.goodFeaturesToTrack(
            frame,
            maxCorners=self.max_points,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask
        )
        
        if points is not None and len(points) > 0:
            self.prev_points = points
            if self.initial_points is None:
                self.initial_points = points.copy()
        else:
            self.prev_points = None
    
    def update(self, prev_frame, curr_frame, prev_mask):
        """Track points with KLT and warp mask with estimated transform."""
        stats = {
            'num_points_prev': 0,
            'num_points_tracked': 0,
            'num_points_inliers': 0,
            'transform_estimated': False,
            'reinitialized': False
        }
        
        # Check if we have points to track
        if self.prev_points is None or len(self.prev_points) < 4:
            # Need at least 4 points for affine estimation
            self._detect_points(curr_frame, prev_mask)
            stats['reinitialized'] = True
            stats['num_points_prev'] = 0
            return prev_mask, stats  # Return previous mask as fallback
        
        stats['num_points_prev'] = len(self.prev_points)
        
        # Track points with KLT
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, self.prev_points, None, **self.lk_params
        )
        
        # Filter valid points
        if status is None:
            self._detect_points(curr_frame, prev_mask)
            stats['reinitialized'] = True
            return prev_mask, stats
        
        good_prev = self.prev_points[status.ravel() == 1]
        good_curr = curr_points[status.ravel() == 1]
        
        stats['num_points_tracked'] = len(good_curr)
        
        # Check if we need to reinitialize
        min_points_threshold = max(4, int(self.num_initial_points * self.reinit_threshold))
        if len(good_curr) < min_points_threshold:
            # Reinitialize points on current frame
            self._detect_points(curr_frame, prev_mask)
            stats['reinitialized'] = True
            return prev_mask, stats  # Return previous mask as fallback
        
        # Estimate similarity transform (translation + rotation + uniform scale)
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        
        if transform_matrix is None:
            # Transform estimation failed, reinitialize
            self._detect_points(curr_frame, prev_mask)
            stats['reinitialized'] = True
            return prev_mask, stats
        
        stats['transform_estimated'] = True
        stats['num_points_inliers'] = np.sum(inliers.ravel()) if inliers is not None else len(good_curr)
        
        # Warp mask with estimated transform
        h, w = prev_mask.shape
        mask_warped = cv2.warpAffine(
            prev_mask, transform_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Threshold to binary
        mask_warped = (mask_warped > 127).astype(np.uint8) * 255
        
        # Light morphological cleanup
        if self.refine_iterations > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.refine_kernel_size, self.refine_kernel_size)
            )
            mask_warped = cv2.morphologyEx(
                mask_warped, cv2.MORPH_CLOSE, kernel, 
                iterations=self.refine_iterations
            )
            mask_warped = cv2.morphologyEx(
                mask_warped, cv2.MORPH_OPEN, kernel, 
                iterations=self.refine_iterations
            )
        
        # Update points for next frame (use inliers if available)
        if inliers is not None:
            self.prev_points = good_curr[inliers.ravel() == 1].reshape(-1, 1, 2)
        else:
            self.prev_points = good_curr.reshape(-1, 1, 2)
        
        return mask_warped, stats


class FlowWarpTracker(MaskTracker):
    """
    Track mask using dense optical flow + pixel-wise warping.
    
    Computes dense Farneback optical flow in an ROI around the mask,
    warps each pixel independently, then refines with morphology and
    component selection.
    """
    
    def __init__(self,
                 roi_margin=30,
                 min_mask_area=300,
                 morph_kernel_size=5,
                 morph_close_iterations=2,
                 morph_open_iterations=1):
        """
        Args:
            roi_margin: Pixels to expand bbox for flow computation
            min_mask_area: Minimum mask area threshold (pixels)
            morph_kernel_size: Morphology kernel size
            morph_close_iterations: Closing iterations
            morph_open_iterations: Opening iterations
        """
        self.roi_margin = roi_margin
        self.min_mask_area = min_mask_area
        self.morph_kernel_size = morph_kernel_size
        self.morph_close_iterations = morph_close_iterations
        self.morph_open_iterations = morph_open_iterations
        
        self.prev_bbox = None
        self.prev_centroid = None
    
    def init(self, frame, mask):
        """Initialize with first frame and mask."""
        self.prev_bbox = get_mask_bbox(mask)
        if np.sum(mask > 0) > 0:
            y_coords, x_coords = np.where(mask > 0)
            self.prev_centroid = (np.mean(x_coords), np.mean(y_coords))
        else:
            self.prev_centroid = None
    
    def update(self, prev_frame, curr_frame, prev_mask):
        """Track mask using optical flow."""
        stats = {
            'flow_mean': 0.0,
            'flow_max': 0.0,
            'roi_used': self.prev_bbox
        }
        
        # Get bbox for ROI
        if self.prev_bbox is None:
            self.prev_bbox = get_mask_bbox(prev_mask)
            if self.prev_bbox is None:
                return prev_mask, stats  # No mask to track
        
        # Compute optical flow in ROI
        flow, flow_roi_bbox = compute_optical_flow_roi(
            prev_frame, curr_frame, self.prev_bbox, margin=self.roi_margin
        )
        
        stats['roi_used'] = flow_roi_bbox
        
        # Compute flow statistics in ROI
        fx1, fy1, fx2, fy2 = flow_roi_bbox
        flow_roi = flow[fy1:fy2, fx1:fx2]
        if flow_roi.size > 0:
            flow_mag = np.sqrt(flow_roi[..., 0]**2 + flow_roi[..., 1]**2)
            stats['flow_mean'] = float(np.mean(flow_mag))
            stats['flow_max'] = float(np.max(flow_mag))
        
        # Warp mask with flow
        mask_warped = warp_mask_with_flow(prev_mask, flow)
        
        # Create ROI constraint to prevent background bleeding
        h, w = prev_mask.shape
        roi_constrain = (
            max(0, fx1 - self.roi_margin),
            max(0, fy1 - self.roi_margin),
            min(w, fx2 + self.roi_margin),
            min(h, fy2 + self.roi_margin)
        )
        
        # Refine mask
        mask_refined = refine_warped_mask(
            mask_warped,
            min_area=self.min_mask_area,
            kernel_size=self.morph_kernel_size,
            close_iterations=self.morph_close_iterations,
            open_iterations=self.morph_open_iterations,
            prev_centroid=self.prev_centroid,
            roi_bbox=roi_constrain
        )
        
        # Update state for next frame
        self.prev_bbox = get_mask_bbox(mask_refined)
        if self.prev_bbox is None:
            self.prev_bbox = self.prev_bbox  # Keep previous if current fails
        
        if np.sum(mask_refined > 0) > 0:
            y_coords, x_coords = np.where(mask_refined > 0)
            self.prev_centroid = (np.mean(x_coords), np.mean(y_coords))
        
        return mask_refined, stats



def get_mask_bbox(mask):
    """Get bounding box from binary mask."""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()
    return (x1, y1, x2, y2)


def compute_optical_flow_roi(gray_prev, gray_curr, bbox_prev, margin=30):
    """
    Compute dense optical flow only within an expanded ROI around the previous mask.
    
    Args:
        gray_prev: Previous frame (grayscale)
        gray_curr: Current frame (grayscale)
        bbox_prev: (x1, y1, x2, y2) bounding box of previous mask
        margin: Pixels to expand bbox (for motion tolerance)
    
    Returns:
        flow: Dense optical flow (H×W×2) with (dx, dy) per pixel
        roi_bbox: Actual ROI used for flow computation
    """
    # Expand bbox by margin
    x1, y1, x2, y2 = bbox_prev
    h, w = gray_prev.shape
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    
    # Extract ROI patches
    roi_prev = gray_prev[y1:y2, x1:x2]
    roi_curr = gray_curr[y1:y2, x1:x2]
    
    # Compute dense optical flow using Farneback
    flow_roi = cv2.calcOpticalFlowFarneback(
        roi_prev, roi_curr, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Create full-sized flow map (zero outside ROI)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[y1:y2, x1:x2] = flow_roi
    
    return flow, (x1, y1, x2, y2)


def warp_mask_with_flow(mask_prev, flow):
    """
    Warp previous mask to current frame using optical flow.
    
    Args:
        mask_prev: Binary mask from previous frame (0/255 uint8)
        flow: Dense optical flow (H×W×2)
    
    Returns:
        mask_warped: Warped mask (0/255 uint8)
    """
    h, w = mask_prev.shape
    
    # Create coordinate grid
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Apply flow to get new coordinates
    # Optical flow gives forward motion (where pixels moved TO)
    # cv2.remap does backward warping (where to sample FROM)
    # So we SUBTRACT flow to find source positions
    x_new = x_grid - flow[..., 0]
    y_new = y_grid - flow[..., 1]
    
    # Remap mask using bilinear interpolation
    mask_warped = cv2.remap(
        mask_prev.astype(np.float32),
        x_new.astype(np.float32),
        y_new.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Threshold back to binary (handle interpolation artifacts)
    mask_warped = (mask_warped > 127).astype(np.uint8) * 255
    
    return mask_warped


def refine_warped_mask(mask_warped, min_area=300, kernel_size=5, close_iterations=2, open_iterations=1, 
                       prev_centroid=None, roi_bbox=None):
    """
    Clean up warped mask: morphology + keep component closest to previous centroid.
    
    Args:
        mask_warped: Warped binary mask (0/255)
        min_area: Minimum area threshold (pixels)
        kernel_size: Size of morphological kernel
        close_iterations: Closing iterations (fewer = less aggressive, less bleeding)
        open_iterations: Opening iterations
        prev_centroid: (x, y) centroid from previous frame - selects closest component
        roi_bbox: (x1, y1, x2, y2) ROI to constrain mask (prevents background smear)
    
    Returns:
        mask_refined: Cleaned mask (0/255)
    """
    # Optional: Constrain to ROI to prevent background bleeding
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        # Create ROI mask
        roi_mask = np.zeros_like(mask_warped)
        roi_mask[y1:y2, x1:x2] = 255
        # Zero out everything outside ROI
        mask_warped = cv2.bitwise_and(mask_warped, roi_mask)
    
    # Conservative morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Moderate closing: fills small holes without excessive bleeding
    mask_clean = cv2.morphologyEx(mask_warped, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    # Opening: removes small noise
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    
    # Keep only largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_clean, connectivity=8
    )
    
    if num_labels <= 1:  # No foreground found
        return np.zeros_like(mask_warped)
    
    # Filter by minimum area first
    valid_components = []
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_components.append(i)
    
    if len(valid_components) == 0:
        return np.zeros_like(mask_warped)
    
    # Select best component: closest to previous centroid if available, else largest
    if prev_centroid is not None and len(valid_components) > 1:
        prev_cx, prev_cy = prev_centroid
        # Find component with centroid closest to previous centroid
        best_idx = -1
        min_dist = float('inf')
        for idx in valid_components:
            curr_cy, curr_cx = centroids[idx]  # centroids is (y, x) format
            dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
    else:
        # Fallback: pick largest component
        best_idx = max(valid_components, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    
    mask_refined = (labels == best_idx).astype(np.uint8) * 255
    
    return mask_refined


def check_tracking_quality(mask_prev, mask_curr, area_ratio_range=(0.5, 2.0), centroid_shift_threshold=50):
    """
    Check if tracking is stable using area and centroid metrics.
    
    Args:
        mask_prev: Previous mask (0/255)
        mask_curr: Current mask (0/255)
        area_ratio_range: (min, max) acceptable area change ratio
        centroid_shift_threshold: Maximum acceptable centroid shift in pixels
    
    Returns:
        is_good: Boolean, True if tracking is reliable
        metrics: Dict with area_ratio, centroid_shift, and issue_types
    """
    area_prev = np.sum(mask_prev > 0)
    area_curr = np.sum(mask_curr > 0)
    
    if area_prev == 0 or area_curr == 0:
        return False, {
            'area_ratio': 0 if area_prev == 0 else float('inf'),
            'centroid_shift': float('inf'),
            'issue_types': ['empty_mask']
        }
    
    # Area change ratio
    area_ratio = area_curr / area_prev
    
    # Centroid shift
    y_prev, x_prev = np.mean(np.where(mask_prev > 0), axis=1)
    y_curr, x_curr = np.mean(np.where(mask_curr > 0), axis=1)
    centroid_shift = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
    
    # Check thresholds
    area_ok = area_ratio_range[0] <= area_ratio <= area_ratio_range[1]
    centroid_ok = centroid_shift < centroid_shift_threshold
    
    # Identify specific issues
    issue_types = []
    if not area_ok:
        issue_types.append('area_anomaly')
    if not centroid_ok:
        issue_types.append('centroid_anomaly')
    
    is_good = area_ok and centroid_ok
    
    metrics = {
        'area_ratio': area_ratio,
        'centroid_shift': centroid_shift,
        'issue_types': issue_types
    }
    
    return is_good, metrics


def process_video(
    video_name,
    output_name,
    tracker,
    pixelation_strength=13,
    bit_depth=5,
    area_ratio_range=(0.5, 2.0),
    centroid_shift_threshold=50,
    progress_interval=30):
    """
    Video processing with modular tracker (Flow or KLT).
    
    Args:
        video_name: Input video filename in VID/ folder
        output_name: Output video filename
        tracker: MaskTracker instance (FlowWarpTracker or KLTTracker)
        pixelation_strength: Pixelation strength for blur (5-20)
        bit_depth: Color quantization depth (3-6)
        area_ratio_range: (min, max) acceptable area change ratio
        centroid_shift_threshold: Max centroid shift in pixels
        progress_interval: Print progress every N frames
    
    Returns:
        summary: Dict with processing statistics
    """
    import time
    from datetime import datetime
    
    print(f"\n{'='*60}")
    print(f"VIDEO PROCESSING: {video_name}")
    print(f"Tracker: {tracker.get_name()}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Load video
    cap, metadata = load_video(video_name)
    fps = metadata['fps']
    width = metadata['width']
    height = metadata['height']
    total_frames = metadata['frame_count']
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = video_dir / output_name
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # === FRAME 1: Initialize mask with SAM ===
    print("\n[INITIALIZATION] Processing first frame...\n")
    success, bgr, gray_prev, rgb = read_video_frame(cap)
    if not success:
        print("ERROR: Failed to read first frame")
        cap.release()
        writer.release()
        return None
    
    # ROI selection
    roi_coords, roi_rgb = select_roi_on_rgb(rgb)
    if roi_rgb is None:
        print("ERROR: No ROI selected")
        cap.release()
        writer.release()
        return None
    
    rect_x1, rect_y1, rect_x2, rect_y2 = roi_coords
    
    # Generate initial mask based on selected method
    segmentation_method = select_segmentation_method()
    
    if segmentation_method == 'sam':
        print("Running SAM segmentation on first frame...")
        mask_roi, _ = sam_box_prompt_segmentation(roi_rgb, show_steps=False)
        if mask_roi is None:
            print("ERROR: SAM segmentation failed")
            cap.release()
            writer.release()
            return None
        mask_prev = convert_roi_mask_to_global(mask_roi, roi_coords, rgb.shape)
    
    elif segmentation_method == 'corners':
        print("Running corner detection on first frame...")
        # Convert ROI to grayscale
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect corners in ROI
        corners = detect_corners_ShiTomasi(roi_gray, max_corner=150)
        if corners is None:
            print("ERROR: Corner detection failed - no corners found")
            cap.release()
            writer.release()
            return None
        
        print(f"Detected {len(corners)} corners in ROI")
        # Translate to global coordinates
        global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        # Create mask from convex hull of points
        mask_prev = create_mask_from_points(global_points, rgb.shape)
    
    else:
        print(f"ERROR: Unknown segmentation method: {segmentation_method}")
        cap.release()
        writer.release()
        return None
    
    if np.sum(mask_prev > 0) == 0:
        print("ERROR: Initial mask is empty")
        cap.release()
        writer.release()
        return None
    
    # Visualize initial mask on first frame
    print("\nVisualizing initial mask...")
    mask_visualization = _overlay_mask(rgb, mask_prev.astype(bool), alpha=0.4)
    show_image([rgb, mask_visualization], row_plot=1, 
               titles=["First Frame (Original)", f"Initial Mask ({segmentation_method.upper()})"])
    
    # Initialize tracker
    tracker.init(gray_prev, mask_prev)
    
    bbox_init = get_mask_bbox(mask_prev)
    print(f"Initial mask bbox: {bbox_init}")
    print(f"Initial mask area: {np.sum(mask_prev > 0)} pixels")
    
    if np.sum(mask_prev > 0) > 0:
        y_init, x_init = np.mean(np.where(mask_prev > 0), axis=1)
        print(f"Initial centroid: ({x_init:.1f}, {y_init:.1f})\n")
    
    # Blur and write first frame
    _, blurred = create_blurred_mask(rgb, mask_prev, pixelation_strength, bit_depth,
                                     show_histogram=False, verbose=False)
    writer.write(cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
    
    print(f"Frame 1/{total_frames}: Initialized with SAM ✓\n")
    print(f"{'='*60}")
    print(f"[TRACKING] Processing remaining frames...\n")
    
    # === FRAMES 2+: Track mask ===
    frame_idx = 2
    anomaly_frames = []
    all_area_ratios = []
    all_centroid_shifts = []
    frame_times = []
    tracker_stats_log = []  # For tracker-specific diagnostics
    
    while True:
        frame_start = time.time()
        
        success, bgr, gray_curr, rgb = read_video_frame(cap)
        if not success:
            break
        
        # Update mask with tracker
        mask_curr, tracker_stats = tracker.update(gray_prev, gray_curr, mask_prev)
        tracker_stats_log.append(tracker_stats)
        
        # Compute mask statistics
        mask_curr_area = np.sum(mask_curr > 0)
        if mask_curr_area > 0:
            y_coords, x_coords = np.where(mask_curr > 0)
            mask_curr_centroid = (np.mean(x_coords), np.mean(y_coords))
        else:
            mask_curr_centroid = (0, 0)
        
        # Quality check
        is_good, metrics = check_tracking_quality(
            mask_prev, mask_curr,
            area_ratio_range=area_ratio_range,
            centroid_shift_threshold=centroid_shift_threshold
        )
        
        # Log metrics
        all_area_ratios.append(metrics['area_ratio'])
        all_centroid_shifts.append(metrics['centroid_shift'])
        
        # Check for KLT-specific issues
        if isinstance(tracker, KLTTracker):
            if tracker_stats.get('num_points_tracked', float('inf')) < 4:
                metrics['issue_types'].append('too_few_points')
                is_good = False
        
        if not is_good:
            anomaly_frames.append((frame_idx, metrics))
            issues_str = ', '.join(metrics['issue_types'])
            print(f"  ⚠ Frame {frame_idx}/{total_frames}: Quality issue ({issues_str})")
            print(f"     Area ratio: {metrics['area_ratio']:.3f}, Centroid shift: {metrics['centroid_shift']:.1f}px")
        
        # Blur and write frame
        _, blurred = create_blurred_mask(rgb, mask_curr, pixelation_strength, bit_depth,
                                         show_histogram=False, verbose=False)
        writer.write(cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
        
        # Update for next iteration
        mask_prev = mask_curr
        gray_prev = gray_curr
        
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        # Progress logging
        if frame_idx % progress_interval == 0 or frame_idx <= 4:
            avg_time = np.mean(frame_times[-progress_interval:]) if len(frame_times) >= progress_interval else np.mean(frame_times)
            fps_processing = 1.0 / avg_time if avg_time > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames} (avg: {avg_time*1000:.1f}ms/frame, {fps_processing:.1f} FPS)")
            print(f"    Mask area: {mask_curr_area}px, Centroid: ({mask_curr_centroid[0]:.1f}, {mask_curr_centroid[1]:.1f})")
            
            # Tracker-specific debug info
            if isinstance(tracker, KLTTracker):
                print(f"    KLT: {tracker_stats.get('num_points_tracked', 0)} points tracked" + 
                      (f", reinit!" if tracker_stats.get('reinitialized') else ""))
            elif isinstance(tracker, FlowWarpTracker):
                print(f"    Flow: mean={tracker_stats.get('flow_mean', 0):.2f}px, max={tracker_stats.get('flow_max', 0):.2f}px")
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    frames_processed = frame_idx - 1
    
    # === FINAL SUMMARY ===
    print(f"\n{'='*60}")
    print(f"[SUMMARY] Video Processing Complete")
    print(f"{'='*60}\n")
    print(f"Output: {output_path}\n")
    print(f"Processing Stats:")
    print(f"  Frames: {frames_processed}/{total_frames}")
    print(f"  Time: {total_time:.2f}s ({total_time/frames_processed*1000:.1f}ms/frame)")
    print(f"  FPS: {frames_processed/total_time:.2f}\n")
    
    print(f"Tracking Quality:")
    print(f"  Anomaly frames: {len(anomaly_frames)} ({len(anomaly_frames)/frames_processed*100:.1f}%)")
    
    if len(anomaly_frames) > 0:
        area_issues = sum(1 for _, m in anomaly_frames if 'area_anomaly' in m['issue_types'])
        centroid_issues = sum(1 for _, m in anomaly_frames if 'centroid_anomaly' in m['issue_types'])
        point_issues = sum(1 for _, m in anomaly_frames if 'too_few_points' in m['issue_types'])
        
        print(f"  Area anomalies: {area_issues}")
        print(f"  Centroid anomalies: {centroid_issues}")
        if point_issues > 0:
            print(f"  Point tracking issues: {point_issues}")
    
    # Tracker-specific statistics
    if isinstance(tracker, KLTTracker):
        reinit_count = sum(1 for s in tracker_stats_log if s.get('reinitialized', False))
        avg_points = np.mean([s.get('num_points_tracked', 0) for s in tracker_stats_log if 'num_points_tracked' in s])
        print(f"\nKLT Stats:")
        print(f"  Reinitializations: {reinit_count}")
        print(f"  Avg points tracked: {avg_points:.1f}")
    elif isinstance(tracker, FlowWarpTracker):
        flow_means = [s.get('flow_mean', 0) for s in tracker_stats_log if 'flow_mean' in s]
        if flow_means:
            print(f"\nFlow Stats:")
            print(f"  Avg flow magnitude: {np.mean(flow_means):.2f}px")
    
    # Average quality metrics
    valid_area_ratios = [r for r in all_area_ratios if r != 0 and r != float('inf')]
    valid_centroid_shifts = [s for s in all_centroid_shifts if s != float('inf')]
    
    if valid_area_ratios:
        print(f"\nQuality Metrics:")
        print(f"  Avg area ratio: {np.mean(valid_area_ratios):.3f} ± {np.std(valid_area_ratios):.3f}")
    if valid_centroid_shifts:
        print(f"  Avg centroid shift: {np.mean(valid_centroid_shifts):.1f}px ± {np.std(valid_centroid_shifts):.1f}px")
    
    print(f"\n{'='*60}\n")
    
    summary = {
        'tracker': tracker.get_name(),
        'frames_processed': frames_processed,
        'total_frames': total_frames,
        'total_time': total_time,
        'anomaly_count': len(anomaly_frames),
        'anomaly_frames': anomaly_frames,
        'avg_area_ratio': np.mean(valid_area_ratios) if valid_area_ratios else None,
        'avg_centroid_shift': np.mean(valid_centroid_shifts) if valid_centroid_shifts else None,
        'output_path': str(output_path)
    }
    
    return summary


###----------------------------------- Image Processing Function -------------------------------

def process_image(
    image_name,
    output_name,
    pixelation_strength=13,
    bit_depth=5,
    show_steps=True):
    """
    Process a single image with ROI selection and mask-based anonymization.
    
    Args:
        image_name: Input image filename in IMG/ folder
        output_name: Output image filename (saved to IMG/ folder)
        pixelation_strength: Pixelation strength for blur (5-20, lower=stronger)
        bit_depth: Color quantization depth (3-6, lower=stronger)
        show_steps: Whether to show intermediate segmentation visualization
    
    Returns:
        summary: Dict with processing statistics
    """
    print(f"\n{'='*60}")
    print(f"  IMAGE PROCESSING: {image_name}")
    print(f"{'='*60}\n")
    
    # Load image
    print("Loading image...")
    bgr, gray, rgb = load_image(image_name)
    print(f"Image loaded: {rgb.shape[1]}x{rgb.shape[0]}\n")
    
    # ROI selection
    print("Please select ROI on the image...")
    roi_coords, roi_rgb = select_roi_on_rgb(rgb)
    if roi_rgb is None:
        print("ERROR: No ROI selected")
        return None
    
    rect_x1, rect_y1, rect_x2, rect_y2 = roi_coords
    print(f"ROI selected: ({rect_x1}, {rect_y1}) to ({rect_x2}, {rect_y2})\n")
    
    # Select segmentation method
    segmentation_method = select_segmentation_method()
    
    # Generate mask based on selected method
    if segmentation_method == 'sam':
        print("\nRunning SAM segmentation...")
        mask_roi, _ = sam_box_prompt_segmentation(roi_rgb, show_steps=show_steps)
        if mask_roi is None:
            print("ERROR: SAM segmentation failed")
            return None
        mask = convert_roi_mask_to_global(mask_roi, roi_coords, rgb.shape)
    
    elif segmentation_method == 'corners':
        print("\nRunning corner detection...")
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect corners in ROI
        corners = detect_corners_ShiTomasi(roi_gray, max_corner=150)
        if corners is None:
            print("ERROR: Corner detection failed - no corners found")
            return None
        
        print(f"Detected {len(corners)} corners in ROI")
        
        # Translate to global coordinates
        global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        
        # Create mask from convex hull
        mask = create_mask_from_points(global_points, rgb.shape)
        
        # Optional: visualize mask
        if show_steps:
            mask_visualization = _overlay_mask(rgb, mask.astype(bool), alpha=0.4)
            show_image([rgb, mask_visualization], row_plot=1,
                      titles=["Original Image", "Corner-based Mask"])
    
    else:
        print(f"ERROR: Unknown segmentation method: {segmentation_method}")
        return None
    
    # Check mask validity
    mask_area = np.sum(mask > 0)
    if mask_area == 0:
        print("ERROR: Generated mask is empty")
        return None
    
    print(f"\nMask generated successfully:")
    print(f"  Area: {mask_area} pixels")
    bbox = get_mask_bbox(mask)
    print(f"  Bbox: {bbox}\n")
    
    # Apply anonymization
    print(f"Applying anonymization (pixelation={pixelation_strength}, bit_depth={bit_depth})...")
    _, anonymized = create_blurred_mask(
        rgb, mask,
        pixelation_strength=pixelation_strength,
        bit_depth=bit_depth,
        show_histogram=False,
        verbose=False
    )
    
    # Save output
    output_path = images_dir / output_name
    imsave(str(output_path), anonymized)
    print(f"Output saved: {output_path}\n")
    
    print(f"{'='*60}")
    print(f"  IMAGE PROCESSING COMPLETE")
    print(f"{'='*60}\n")
    
    summary = {
        'image_name': image_name,
        'output_name': output_name,
        'segmentation_method': segmentation_method,
        'mask_area': int(mask_area),
        'bbox': bbox,
        'output_path': str(output_path)
    }
    
    return summary


###----------------------------------- corner detection functions -------------------------------

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


###----------------------------------- Blur functions ------------------------------------------

def create_mask_from_points(global_points, image_shape):
    """
    Create a binary mask from global corner points using convex hull.
    
    Args:
        global_points: List of (x, y) points in global image coordinates
        image_shape: Tuple (height, width) of the full image
    
    Returns:
        mask: Binary mask (0/255) in global coordinates with same size as image
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    pts = np.array(global_points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    
    return mask


def convert_roi_mask_to_global(roi_mask, roi_coords, image_shape):
    """
    Convert ROI-local mask to global image coordinates.
    
    Args:
        roi_mask: Binary mask (0/255) in ROI-local coordinates
        roi_coords: Tuple (x1, y1, x2, y2) defining ROI position in full image
        image_shape: Tuple (height, width) of the full image
    
    Returns:
        mask: Binary mask (0/255) in global coordinates with same size as full image
    """
    h, w = image_shape[:2]
    rect_x1, rect_y1, rect_x2, rect_y2 = roi_coords
    
    # Create full-sized mask
    global_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Place ROI mask at correct position
    global_mask[rect_y1:rect_y2, rect_x1:rect_x2] = roi_mask
    
    return global_mask


def create_blurred_mask(rgb, mask, pixelation_strength=5, bit_depth=3, show_histogram:bool=True, verbose:bool=False):
    """
    Create irreversible anonymization by pixelation and color quantization.
    
    Args:
        rgb: Input RGB image
        mask: Binary mask (0/255) defining the ROI in global coordinates
        pixelation_strength: Size in pixels for the shorter side after downsampling (lower = stronger blur)
                           Range: 5-20 (5=very strong, 20=mild)
        bit_depth: Bits per channel for color quantization (lower = stronger blur)
                  Range: 3-6 (3=very strong, 6=mild)
    
    Returns:
        mask: Binary mask of the ROI (same as input)
        anonymized_full: Full image with anonymized ROI applied
    """
    h, w = rgb.shape[:2]

    # Extract bounding box of the mask
    coords = np.where(mask == 255)
    if len(coords[0]) == 0:
        print("Warning: Empty mask")
        return mask, rgb.copy()
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    y = y_min
    x = x_min
    roi_h = y_max - y_min + 1
    roi_w = x_max - x_min + 1
    
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



###----------------------------------- Main - active the project ------------------------------------------

if __name__ == "__main__":
    
    # Select processing mode
    mode = select_mode()
    
    if mode == 'image':
        print("\n[IMAGE MODE] Starting image processing...\n")
        
        ## Option 1: Use the integrated process_image function (recommended)
        ## This uses select_segmentation_method() to choose SAM or Corner detection
        
        # Specify input image filename (change this to your image file)
        selected_image = "YOUR_IMAGE_FROM_IMG_FOLDER.jpeg"
        
        # Generate output filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_stem = Path(selected_image).stem
        output_image = f"{image_stem}_anonymized_{timestamp}.png"
        
        print(f"Input image: {selected_image}")
        print(f"Output will be saved as: {output_image}\n")
        
        # Process image with selected segmentation method
        summary = process_image(
            image_name=selected_image,
            output_name=output_image,
            pixelation_strength=13,
            bit_depth=5,
            show_steps=True
        )
        
        if summary is not None:
            print("Image processing completed successfully!")
        else:
            print("Image processing failed.")
        
        
        ## Option 2: Manual processing with custom function combinations
        ## Uncomment this section to use specific processing functions
        
        # ## Initial setup ##
        # bgr, gray, rgb = load_image("IMG_004.jpeg")

        # (rect_x1, rect_y1, rect_x2, rect_y2), roi_rgb = select_roi_on_rgb(rgb)
        # print(f"Selected rectangle: ({rect_x1}, {rect_y1}) to ({rect_x2}, {rect_y2})")
        
        # if roi_rgb is not None:
        #     show_image(roi_rgb, titles=["Selected ROI"], row_plot=1)
        
        #     roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
            
        #     ### ---------------------------------------------------------------------------
        #     ### first image processing functions ##

        #     # # Binary than grey mask
        #     # roi_gray_masked = binary_mask(roi_gray)
        #     # corners = detect_corners_ShiTomasi(roi_gray_masked, max_corner=30)
        #     # global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        #     # rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        #     # show_image([rgb_debug], titles=["Binary Mask debug"], row_plot=1)
        #     # mask = create_mask_from_points(global_points, rgb.shape)

        #     # # Dilation 
        #     # dilated = diliation(roi_gray)
        #     # corners = detect_corners_ShiTomasi(dilated, max_corner=30)
        #     # global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        #     # rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        #     # show_image([rgb_debug], titles=["Dilation debug"], row_plot=1)
        #     # mask = create_mask_from_points(global_points, rgb.shape)

        #     # # Morphological gradient
        #     # outer = Morphological_gradient(roi_gray)
        #     # corners = detect_corners_ShiTomasi(outer, max_corner=30)
        #     # global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        #     # rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        #     # show_image([rgb_debug], titles=["Morphological Gradient debug"], row_plot=1)
        #     # mask = create_mask_from_points(global_points, rgb.shape)

        #     # # Apply CLAHE on grayscale image
        #     # he_clahe = clahe(roi_gray)
        #     # corners = detect_corners_ShiTomasi(he_clahe, max_corner=30)
        #     # global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        #     # rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        #     # show_image([rgb_debug], titles=["CLAHE debug"], row_plot=1)
        #     # mask = create_mask_from_points(global_points, rgb.shape)

        #     # # Unsharp masking (blur → subtract → sharpen)
        #     # sharpen = unsharp_masking(roi_gray)
        #     # corners = detect_corners_ShiTomasi(sharpen, max_corner=30)
        #     # global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        #     # rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        #     # show_image([rgb_debug], titles=["Unsharp Masking debug"], row_plot=1)
        #     # mask = create_mask_from_points(global_points, rgb.shape)


        #     ### ---------------------------------------------------------------------------
        #     ## second image processing functions ##

        #     # # Custom sharpening kernel
        #     # sharp = custom_kernel(roi_gray)
        #     # corners = detect_corners_ShiTomasi(sharp, max_corner=30)
        #     # global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        #     # rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        #     # show_image([rgb_debug], titles=["Custom Kernel debug"], row_plot=1)
        #     # mask = create_mask_from_points(global_points, rgb.shape)

        #     # # Sobel gradients (derivatives)
        #     # sobel_mag = sobel(roi_gray)
        #     # corners = detect_corners_ShiTomasi(sobel_mag, max_corner=30)
        #     # global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        #     # rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        #     # show_image([rgb_debug], titles=["Sobel debug"], row_plot=1)
        #     # mask = create_mask_from_points(global_points, rgb.shape)

        #     # # Laplacian (second derivative)
        #     # lap = laplacian(roi_gray)
        #     # corners = detect_corners_ShiTomasi(lap, max_corner=30)
        #     # global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        #     # rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        #     # show_image([rgb_debug], titles=["Laplacian debug"], row_plot=1)
        #     # mask = create_mask_from_points(global_points, rgb.shape)

        #     # # Canny (works on 8-bit)
        #     # edges = canny(roi_gray)
        #     # corners = detect_corners_ShiTomasi(edges, max_corner=30)
        #     # global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        #     # rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        #     # show_image([rgb_debug], titles=["Canny debug"], row_plot=1)
        #     # mask = create_mask_from_points(global_points, rgb.shape)

        #     ### ---------------------------------------------------------------------------
        #     ## segmentation functions ##

        #     ## GrabCut segmentation (foreground/background separation)
        #     # mask_roi, vis_grabcut = grabcut_segmentation(roi_rgb, iterations=5, show_steps=True)
        #     # roi_coords = (rect_x1, rect_y1, rect_x2, rect_y2)
        #     # mask = convert_roi_mask_to_global(mask_roi, roi_coords, rgb.shape)
            
        #     # SAM with box prompt (automatic mask selection)
        #     mask_roi, vis_sam_box = sam_box_prompt_segmentation(roi_rgb, show_steps=True)
        #     roi_coords = (rect_x1, rect_y1, rect_x2, rect_y2)
        #     mask = convert_roi_mask_to_global(mask_roi, roi_coords, rgb.shape)
            
        #     ## Watershed segmentation (distance transform seeds)
        #     # mask_roi, vis_watershed = watershed_roi_segmentation(roi_rgb, distance_threshold=0.3, show_steps=True)
        #     # roi_coords = (rect_x1, rect_y1, rect_x2, rect_y2)
        #     # mask = convert_roi_mask_to_global(mask_roi, roi_coords, rgb.shape)


        #     ### ---------------------------------------------------------------------------
        #     ## Blurring the object ##
            
        #     if mask is not None:
        #         mask, blurred_full = create_blurred_mask(rgb, mask, pixelation_strength=13, bit_depth=5, show_histogram=True, verbose=True)

        #         result = rgb.copy()
        #         result[mask == 255] = blurred_full[mask == 255]
        #         show_image([result], titles=["Object blurred"], row_plot=1)

        #     else: 
        #         print("No mask generated, skipping blurring step.")
        #         print("check and activate one of the image processing / segmentation functions.")


        # else:
        #     print("No ROI selected - active again and select ROI from the image.")
    
    elif mode == 'video':
        print("\n[VIDEO MODE] Starting video processing...\n")
        
        # Select tracking method
        tracker = select_tracker()
        
        # Specify video filename (change this to your video file)
        selected_video = "YOUR_VIDEO_FROM_VID_FOLDER.mp4" 
        
        # Generate output filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_stem = Path(selected_video).stem
        tracker_suffix = "flow" if isinstance(tracker, FlowWarpTracker) else "klt"
        output_video = f"{video_stem}_blur_{tracker_suffix}_{timestamp}.mp4"
        
        print(f"\nInput video: {selected_video}")
        print(f"Output will be saved as: {output_video}\n")
        
        # Process video with selected tracker
        summary = process_video(
            video_name=selected_video,
            output_name=output_video,
            tracker=tracker,
            pixelation_strength=13,
            bit_depth=5,
            area_ratio_range=(0.5, 2.0),
            centroid_shift_threshold=50,
            progress_interval=10
        )
        
        if summary is not None:
            print("Video processing completed successfully!")
        else:
            print("Video processing failed.")





