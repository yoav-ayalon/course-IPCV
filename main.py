
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




rect_start = None
rect_end = None
drawing = False

def select_roi_on_rgb(rgb):
    """
    פותח חלון, נותן לבחור מלבן עם העכבר על התמונה rgb.
    מחזיר: (x1, y1, x2, y2), roi_rgb
    אם המשתמש יצא בלי לבחור – מחזיר (None, None)
    """
    global rect_start, rect_end, drawing
    rect_start = None
    rect_end = None
    drawing = False

    win_name = "Select ROI (drag with mouse, ENTER/ESC to finish)"

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
        # מציגים עותק של התמונה, עם מלבן אם יש
        frame = rgb.copy()
        if rect_start and rect_end:
            cv2.rectangle(frame, rect_start, rect_end, (0, 255, 0), 2)

        # OpenCV עובד ב-BGR, אז נהפוך רק לצורך הצגה
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(win_name, frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (13, 27):  # ENTER או ESC
            break

    cv2.destroyWindow(win_name)

    if not (rect_start and rect_end):
        return None, None

    x1, y1 = rect_start
    x2, y2 = rect_end
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    roi_rgb = rgb[y1:y2, x1:x2]

    return (x1, y1, x2, y2), roi_rgb







if __name__ == "__main__":
    
    

    # img_hsv = rgb2hsv(rgb)           # returns floats in [0,1]
    # H, S, V = img_hsv[...,0], img_hsv[...,1], img_hsv[...,2]
    # # show_image([rgb, H, S, V], row_plot=1)
    
    # gray_S = (S * 255).astype(np.uint8)
    # rgb_shi = detect_corners_ShiTomasi(gray_S, rgb, max_corner=100)
    # show_image([rgb,S, rgb_shi], titles=["Original RGB", "S - for saturation","Shi-Tomasi Corners"], row_plot=1)

    # rgb1_orb, kps1_orb, desc1_orb = orb_features(bgr)

    # # Visualize keypoints
    # vis1 = rgb1_orb.copy()
    # cv2.drawKeypoints(rgb1_orb, kps1_orb, vis1,
    #                 color=(0, 255, 0),
    #                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # show_image([vis1], titles=["ORB Keypoints"], row_plot=1)

    bgr, gray, rgb = load_image("IMG_001.PNG")
    # show_image([rgb], titles=["RGB"], row_plot=1)

    (rect_x1, rect_y1, rect_x2, rect_y2), roi_rgb = select_roi_on_rgb(rgb)
    print(f"Selected rectangle: ({rect_x1}, {rect_y1}) to ({rect_x2}, {rect_y2})")
    if roi_rgb is not None:
        show_image(roi_rgb, titles=["Selected ROI"], row_plot=1)
    
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

        roi_shi_corners = detect_corners_ShiTomasi(roi_gray, max_corner=20)

        if roi_shi_corners is None or len(roi_shi_corners) == 0:
            print("No corners found in ROI.")
        else:
            # לוודא צורה (N, 2)
            corners = np.array(roi_shi_corners, dtype=np.float32)
            if corners.ndim == 3:   # במקרה של (N,1,2) כמו goodFeaturesToTrack
                corners = corners.reshape(-1, 2)


            # 4. לתרגם את הפינות ל"קואורדינטות גלובליות" ולהציג על התמונה
            global_points = []
            for (cx, cy) in corners:
                gx = int(cx + rect_x1)
                gy = int(cy + rect_y1)
                global_points.append((gx, gy))
            
            rgb_debug = rgb.copy()
            for (gx, gy) in global_points:
                cv2.circle(rgb_debug, (gx, gy), 4, (0, 255, 0), -1)
            show_image([rgb_debug], titles=["Corners debug"], row_plot=1)

            h, w = rgb.shape[:2]
            # מסיכה ריקה בגודל התמונה
            mask = np.zeros((h, w), dtype=np.uint8)
            # נקודות האובייקט כ־np.array
            pts = np.array(global_points, dtype=np.int32)
            # מעטפת קמורה של הפינות
            hull = cv2.convexHull(pts)
            # מילוי הפוליגון במסיכה
            cv2.fillConvexPoly(mask, hull, 255)   # אזור האובייקט = 255

            # טשטוש חזק של כל התמונה
            # אפשר לשחק עם sigmaX / sigmaY כדי להעצים
            blurred_full = cv2.GaussianBlur(rgb, ksize=(0, 0), sigmaX=25, sigmaY=25)

            # שילוב – רק איפה שהמסיכה 255 ניקח מהתמונה המטושטשת
            result = rgb.copy()
            result[mask == 255] = blurred_full[mask == 255]

            show_image([result], titles=["Object blurred (no green points)"], row_plot=1)




    

