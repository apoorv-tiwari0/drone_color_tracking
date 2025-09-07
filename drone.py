import cv2
import numpy as np
import time
import csv
from collections import OrderedDict, deque


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=60):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.trails = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.trails[self.nextObjectID] = deque(maxlen=30)
        self.trails[self.nextObjectID].append(centroid)
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.trails[objectID]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        D = np.linalg.norm(np.array(objectCentroids)[:, None] - np.array(input_centroids)[None, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows, usedCols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.max_distance:
                continue
            objectID = objectIDs[row]
            self.objects[objectID] = tuple(input_centroids[col])
            self.disappeared[objectID] = 0
            self.trails[objectID].append(tuple(input_centroids[col]))
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(D.shape[0])) - usedRows
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.max_disappeared:
                self.deregister(objectID)

        unusedCols = set(range(D.shape[1])) - usedCols
        for col in unusedCols:
            self.register(tuple(input_centroids[col]))

        return self.objects

VIDEO_PATH = r"C:\drone\drone_video.mp4"   
OUTPUT_VIDEO = "output_tracked.mp4"
LOG_FILE = "detections.csv"
COLOR_TO_TRACK = "orange"      
DISPLAY = True                 
SAVE_TRAIL = True             
MIN_AREA = 600                 

COLOR_RANGES = {
    "orange": ((5, 120, 120), (20, 255, 255)),
    "yellow": ((20, 100, 100), (35, 255, 255)),
    "red1": ((0, 100, 100), (8, 255, 255)),
    "red2": ((160, 100, 100), (179, 255, 255)),
    "green": ((35, 60, 60), (85, 255, 255)),
    "blue": ((90, 60, 60), (135, 255, 255)),
}


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))

    csvfile = open(LOG_FILE, mode="w", newline="")
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["timestamp", "frame", "object_id", "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])

    tracker = CentroidTracker(max_disappeared=40, max_distance=80)
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        masks = []
        if COLOR_TO_TRACK == "red":
            lower1, upper1 = COLOR_RANGES["red1"]
            lower2, upper2 = COLOR_RANGES["red2"]
            mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            masks.append(cv2.bitwise_or(mask1, mask2))
        else:
            lower, upper = COLOR_RANGES[COLOR_TO_TRACK]
            masks.append(cv2.inRange(hsv, np.array(lower), np.array(upper)))

        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections, bboxes = [], []

        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx = int(x + w/2)
            cy = int(y + h/2)
            detections.append((cx, cy))
            bboxes.append((x, y, w, h))

        objects = tracker.update(detections)

        for oid, centroid in objects.items():
            cx, cy = centroid
            chosen, best_dist = None, float("inf")
            for bb in bboxes:
                bx, by, bw, bh = bb
                bcx, bcy = int(bx + bw/2), int(by + bh/2)
                d = np.hypot(cx - bcx, cy - bcy)
                if d < best_dist:
                    best_dist = d
                    chosen = bb
            if chosen:
                bx, by, bw, bh = chosen
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {oid}", (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                timestamp = time.time() - start_time
                csvwriter.writerow([f"{timestamp:.2f}", frame_idx, oid, cx, cy, bx, by, bw, bh])

            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            if SAVE_TRAIL:
                pts = tracker.trails.get(oid, [])
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    thickness = int(np.sqrt(30 / float(i + 1)) * 2)
                    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        cv2.putText(frame, f"Frame: {frame_idx}  Detected: {len(objects)}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if DISPLAY:
            cv2.imshow("Mask", mask)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        writer.write(frame)

    csvfile.close()
    cap.release()
    writer.release()
    if DISPLAY:
        cv2.destroyAllWindows()

    print("Finished. Log saved to:", LOG_FILE)
    print("Output video saved to:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
