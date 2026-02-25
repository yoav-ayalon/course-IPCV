# Image & Video Anonymization Tool

Interactive object segmentation and anonymization using SAM, KLT tracking, and optical flow.

---

## Quick Start

### 1. Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Ensure the SAM checkpoint exists:
```
model/checkpoints/sam_vit_b_01ec64.pth
```

### 2. Prepare Input

- **Images**: Place files in the `IMG/` folder
- **Videos**: Place files in the `VID/` folder

**Sample files are already included** in both folders for immediate testing.

### 3. Configure Input File

Open [main.py](main.py) and specify your input:

- **For Image mode** (line ~1927):  
  ```python
  selected_image = "YOUR_IMAGE_FROM_IMG_FOLDER.jpeg"
  ```

- **For Video mode** (line ~2072):  
  ```python
  selected_video = "YOUR_VIDEO_FROM_VID_FOLDER.mp4"
  ```

### 4. Run

```bash
python main.py
```

---

## Interactive Workflow

### Step 1: Select Processing Mode

Choose between:
1. **Image** – Process a single image
2. **Video** – Process video with frame-by-frame tracking

### Step 2: Select ROI (Region of Interest)

- A window displays your image (or video's first frame)
- **Drag** a rectangle around the object to anonymize
- Press **ENTER** or **ESC** to confirm

### Step 3: Choose Mask Initialization Method

Select how the mask is generated:

1. **SAM (Segment Anything Model)**  
   AI-powered segmentation—automatic and accurate

2. **Shi–Tomasi Corner Detection + Convex Hull**  
   Classical feature-based approach—lightweight

### Step 4 (Video Only): Choose Tracking Method

Select how the mask is propagated across frames:

1. **Dense Optical Flow (Farnebäck)**  
   Pixel-wise motion warping—handles deformations

2. **KLT (Kanade–Lucas–Tomasi)**  
   Feature point tracking with global transform—fast and stable

---

## Outputs

- **Images**: Saved to `IMG/` with filename pattern:  
  `{original_name}_anonymized_{timestamp}.png`

- **Videos**: Saved to `VID/` with filename pattern:  
  `{original_name}_blur_{tracker}_{timestamp}.mp4`

