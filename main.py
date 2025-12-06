import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import scipy.stats as stats
import cv2
# from skimage import exposure
import os
from skimage.io import imread, imsave
from skimage.color import rgb2hsv, hsv2rgb
# from skimage.util import img_as_float
from typing import List, Union
from pathlib import Path


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


def select_roi_on_rgb(rgb, max_width=1200, max_height=1000):
    """
    Allow user to select a rectangular ROI on the RGB image using mouse.
    Returns (rect_coords, roi_rgb) where rect_coords = (x1, y1, x2, y2).
    If no ROI selected, returns (None, None).
    """


    global rect_start, rect_end, drawing
    rect_start = None
    rect_end = None
    drawing = False

    win_name = "Select ROI (drag with mouse, ENTER/ESC to finish)"

    h, w = rgb.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0) 

    disp_w, disp_h = int(w * scale), int(h * scale)
    rgb_disp = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

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


def create_blurred_mask(rgb, global_points):

    h, w = rgb.shape[:2] # height, width
    mask = np.zeros((h, w), dtype=np.uint8) # binary mask
    pts = np.array(global_points, dtype=np.int32) 
    hull = cv2.convexHull(pts) # compute convex hull
    cv2.fillConvexPoly(mask, hull, 255) # fill hull area in mask

    blurred_full = cv2.GaussianBlur(rgb, ksize=(0, 0), sigmaX=25, sigmaY=25)
    return mask, blurred_full










if __name__ == "__main__":
    
    bgr, gray, rgb = load_image("IMG_002.jpeg")

    (rect_x1, rect_y1, rect_x2, rect_y2), roi_rgb = select_roi_on_rgb(rgb)
    print(f"Selected rectangle: ({rect_x1}, {rect_y1}) to ({rect_x2}, {rect_y2})")
    
    if roi_rgb is not None:
        show_image(roi_rgb, titles=["Selected ROI"], row_plot=1)
    
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        corners = detect_corners_ShiTomasi(roi_gray, max_corner=20)
        global_points = translate_corners_to_global(corners, rect_x1, rect_y1)
        
        rgb_debug = debug_draw_corners(rgb.copy(), global_points)
        show_image([rgb_debug], titles=["Corners debug"], row_plot=1)

        mask, blurred_full = create_blurred_mask(rgb, global_points)
        result = rgb.copy()
        result[mask == 255] = blurred_full[mask == 255]
        show_image([result], titles=["Object blurred"], row_plot=1)




    

