import cv2
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
VIDEO_SOURCE      = "video.mp4"
DETECTION_OUTPUT  = "detection_tracked.mp4"   # input to Stage 2
MODEL_SIZE        = (640, 640)
FRAME_SKIP        = 3


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
yolo_model = YOLO("yolov8s.pt")


# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: cannot open video.")
    exit()

src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(DETECTION_OUTPUT, fourcc, fps, (src_w, src_h))

frame_count  = 0
last_display = None

print("Stage 1 — Detection & Tracking …  (press Q to quit)")

while True:
    ret, raw_frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ── Skip frames: write last annotated frame to keep video smooth ──
    if frame_count % FRAME_SKIP != 0:
        if last_display is not None:
            out.write(last_display)
        else:
            out.write(raw_frame)        # first few frames before first detect
        continue

    # ── Resize for YOLO (letterbox not needed here — YOLO handles it) ──
    results = yolo_model.track(
        raw_frame,
        persist=True,
        classes=[0],        # class 0 = person
        conf=0.5,
        imgsz=MODEL_SIZE[0]
    )

    annotated = raw_frame.copy()

    if results[0].boxes.id is not None:
        boxes      = results[0].boxes.xyxy.cpu().numpy()
        track_ids  = results[0].boxes.id.cpu().numpy().astype(int)

        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"Track {tid}",
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2
            )

    last_display = annotated
    out.write(annotated)

    cv2.imshow("Stage 1 — Detection & Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Stage 1 done. Tracked video saved to: {DETECTION_OUTPUT}")