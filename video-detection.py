import cv2
import numpy as np
import os

# ---------- Config (tweak here if needed) ----------
# Video list
INPUT_VIDEOS = ["ppe-1.mp4", "ppe-2.mp4", "ppe-3.mp4"]
OUTPUT_SUFFIX = "-det.mp4"
OUTPUT_FPS = 20.0

# Min areas (in pixels) to ignore tiny blobs
MIN_AREA_HELMET = 800
MIN_AREA_JACKET = 2000

# IoU threshold for NMS (higher = keep more overlaps)
NMS_IOU = 0.45

# Kernels for morphology
K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
K7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# ---------- Helpers ----------
def iou(a, b):
    # boxes: x1,y1,x2,y2
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (a[2]-a[0]+1) * (a[3]-a[1]+1)
    areaB = (b[2]-b[0]+1) * (b[3]-b[1]+1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def nms(boxes, scores, iou_thresh=NMS_IOU):
    if not boxes:
        return []
    idxs = np.argsort(scores)[::-1]
    picked = []
    while len(idxs) > 0:
        i = idxs[0]
        picked.append(i)
        suppress = [0]
        for pos in range(1, len(idxs)):
            j = idxs[pos]
            if iou(boxes[i], boxes[j]) > iou_thresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return [boxes[i] for i in picked]

def color_mask(hsv, lower, upper):
    return cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))

def build_masks(hsv):
    # --- Helmets: white + yellow ---
    helmet_white = color_mask(hsv, (0, 0, 200), (179, 45, 255))
    helmet_yellow = color_mask(hsv, (20, 90, 140), (35, 255, 255))
    helmet_mask = cv2.bitwise_or(helmet_white, helmet_yellow)
    helmet_mask = cv2.morphologyEx(helmet_mask, cv2.MORPH_OPEN, K3, iterations=1)
    helmet_mask = cv2.morphologyEx(helmet_mask, cv2.MORPH_CLOSE, K5, iterations=2)

    # --- Jackets: neon yellow/green + orange/red (hi-vis) ---
    jacket_yellow = color_mask(hsv, (22, 100, 130), (40, 255, 255))
    jacket_orange = color_mask(hsv, (5, 150, 120), (18, 255, 255))
    jacket_red_1  = color_mask(hsv, (0, 150, 120), (5, 255, 255))
    jacket_red_2  = color_mask(hsv, (170, 120, 120), (179, 255, 255))
    jacket_mask = jacket_yellow | jacket_orange | jacket_red_1 | jacket_red_2

    # Don’t let helmets be counted as jackets:
    helmet_dilated = cv2.dilate(helmet_mask, K9, iterations=1)
    jacket_mask = cv2.bitwise_and(jacket_mask, cv2.bitwise_not(helmet_dilated))

    jacket_mask = cv2.morphologyEx(jacket_mask, cv2.MORPH_OPEN, K5, iterations=2)
    jacket_mask = cv2.morphologyEx(jacket_mask, cv2.MORPH_CLOSE, K7, iterations=2)
    return helmet_mask, jacket_mask

def contour_boxes(mask, cls_name):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if cls_name == "Helmet" and area < MIN_AREA_HELMET:
            continue
        if cls_name == "Jacket" and area < MIN_AREA_JACKET:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue

        # Shape features
        peri = cv2.arcLength(c, True)
        circularity = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
        aspect = w / float(h)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) if hull is not None else 0.0
        solidity = (area / hull_area) if hull_area > 0 else 0.0
        extent = area / (w * h)

        # Class-specific filters (tight to reduce false positives)
        if cls_name == "Helmet":
            # round-ish, not too skinny, fairly solid
            if not (0.6 <= aspect <= 1.6 and circularity >= 0.45 and extent >= 0.30 and solidity >= 0.75):
                continue
        else:  # Jacket
            # torso patch: bigger, rectangular-ish, not too thin
            if not (0.5 <= aspect <= 2.6 and extent >= 0.25 and solidity >= 0.60):
                continue

        # Slightly expand box for nicer visualization
        pad = 4
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = x + w + pad, y + h + pad
        boxes.append([x1, y1, x2, y2])
        scores.append(float(area))
    # NMS to drop duplicates
    kept = nms(boxes, scores, iou_thresh=NMS_IOU)
    return kept

def draw_boxes(frame, boxes, label, color):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def process_video(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"Could not open {in_path}")
        return

    # Prepare writer once we know frame size
    out = None
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        helmet_mask, jacket_mask = build_masks(hsv)

        helmet_boxes = contour_boxes(helmet_mask, "Helmet")
        jacket_boxes = contour_boxes(jacket_mask, "Jacket")

        # Draw (Helmet = cyan, Jacket = green)
        draw_boxes(frame, helmet_boxes, "Helmet", (255, 255, 0))
        draw_boxes(frame, jacket_boxes, "Jacket", (0, 255, 0))

        if out is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, OUTPUT_FPS, (w, h))

        out.write(frame)
        frame_id += 1

    cap.release()
    if out is not None:
        out.release()
    print(f"✅ Saved: {out_path}")

def main():
    for vid in INPUT_VIDEOS:
        base, ext = os.path.splitext(vid)
        process_video(vid, base + OUTPUT_SUFFIX)

if __name__ == "__main__":
    main()
