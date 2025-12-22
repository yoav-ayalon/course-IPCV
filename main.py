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
        corners = np.intp(corners)  # integer coordinates

        # rgb_shi = rgb.copy()
        # for c in corners:
        #     x, y = c.ravel()
        #     cv2.circle(rgb_shi, (x, y), 4, (0, 255, 0), -1)


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


if __name__ == "__main__":
    
    bgr, gray, rgb = load_image("IMG_006.jpeg")

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

        mask, blurred_full = create_blurred_mask(rgb, global_points, pixelation_strength=5, bit_depth=3, show_histogram=False)
        # mask, blurred_full = mean_blur(gray, rgb, global_points)

        result = rgb.copy()
        result[mask == 255] = blurred_full[mask == 255]
        show_image([result], titles=["Object blurred"], row_plot=1)


