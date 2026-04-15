import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import torchreid
from PIL import Image

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
VIDEO_SOURCE    = "detection_tracked.mp4"   # output of Stage 1
OUTPUT_FRAMES   = "output_frames"           # folder for saved frames
OUTPUT_VIDEO    = "reid_output.mp4"         # optional final video
MODEL_SIZE      = (640, 640)
FRAME_SKIP      = 3
REID_THRESHOLD  = 0.6

os.makedirs(OUTPUT_FRAMES, exist_ok=True)


# ─────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────
yolo_model = YOLO("yolov8s.pt")

reid_model = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    pretrained=True
)
reid_model.eval()

reid_transform = torchreid.data.transforms.build_transforms(
    height=256, width=128, is_train=False
)[1]


# ─────────────────────────────────────────
# PERSON DATABASE
# ─────────────────────────────────────────
person_database = {}
next_person_id  = 1


# ═════════════════════════════════════════
# PREPROCESSING HELPERS
# ═════════════════════════════════════════

def mask_watermark(frame):
    """Black-out a fixed bottom-right watermark region."""
    h, w = frame.shape[:2]
    frame[int(h * 0.9):h, int(w * 0.75):w] = 0
    return frame


def crop_roi(frame):
    """Keep only the lower 70 % of the frame (remove sky / ceiling)."""
    h = frame.shape[0]
    return frame[int(h * 0.3):h, :]


def apply_clahe(frame):
    """Adaptive histogram equalisation on the L-channel (LAB space)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def letterbox(frame, target=(640, 640), color=(114, 114, 114)):
    th, tw = target
    h,  w  = frame.shape[:2]

    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dw = (tw - nw) / 2
    dh = (th - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, (dw, dh)


def unscale_box(box, scale, pad):
    dw, dh = pad
    x1, y1, x2, y2 = box
    x1 = (x1 - dw) / scale
    y1 = (y1 - dh) / scale
    x2 = (x2 - dw) / scale
    y2 = (y2 - dh) / scale
    return int(x1), int(y1), int(x2), int(y2)


def preprocess_frame(frame):
    """Full preprocessing pipeline: watermark → ROI crop → enhance → letterbox."""
    frame = mask_watermark(frame)
    frame = crop_roi(frame)

    enhanced = apply_clahe(frame)
    enhanced = gamma_correction(enhanced)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    model_input, scale, pad = letterbox(enhanced, target=MODEL_SIZE)

    # display_frame is the enhanced+cropped frame (original coords for drawing)
    return model_input, scale, pad, enhanced


# ═════════════════════════════════════════
# RE-ID HELPERS
# ═════════════════════════════════════════

def extract_feature(crop):
    """Extract OSNet embedding from a BGR person crop."""
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = reid_transform(img).unsqueeze(0)
    with torch.no_grad():
        feature = reid_model(img)
    return feature.cpu().numpy()


def match_person(feature):
    """Return an existing person ID or register a new one."""
    global next_person_id

    if not person_database:
        person_database[next_person_id] = feature
        pid = next_person_id
        next_person_id += 1
        return pid

    best_id, best_score = None, 0.0
    for pid, stored in person_database.items():
        score = cosine_similarity(feature, stored)[0][0]
        if score > best_score:
            best_score, best_id = score, pid

    if best_score > REID_THRESHOLD:
        # Running average of the stored embedding
        person_database[best_id] = (person_database[best_id] + feature) / 2
        return best_id

    person_database[next_person_id] = feature
    pid = next_person_id
    next_person_id += 1
    return pid


# ═════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: cannot open video '{VIDEO_SOURCE}'.")
    print("Make sure Stage 1 has run and produced detection_tracked.mp4")
    exit()

src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30

# Output video dimensions follow the ROI crop (lower 70 % of source)
roi_h    = src_h - int(src_h * 0.3)
out_w, out_h = src_w, roi_h

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (out_w, out_h))

frame_count  = 0
saved_count  = 0
last_display = None

print("Stage 2 — Preprocessing + ReID …  (press Q to quit)")

while True:
    ret, raw_frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        if last_display is not None:
            out.write(last_display)
        continue

    # ── Preprocess ────────────────────────────────────────────────────
    model_input, scale, pad, display_frame = preprocess_frame(raw_frame)

    # ── Detect & track on preprocessed (letterboxed) frame ───────────
    results = yolo_model.track(
        model_input, persist=True, classes=[0], conf=0.5
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = unscale_box(box, scale, pad)

            # Clamp to display_frame bounds
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(out_w - 1, x2);  y2 = min(out_h - 1, y2)

            person_crop = display_frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            feature   = extract_feature(person_crop)
            person_id = match_person(feature)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"ID {person_id}",
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2
            )

    # ── Save frame to OUTPUT_FRAMES folder ───────────────────────────
    frame_filename = os.path.join(OUTPUT_FRAMES, f"frame_{frame_count:06d}.jpg")
    cv2.imwrite(frame_filename, display_frame)
    saved_count += 1

    last_display = display_frame
    out.write(display_frame)

    cv2.imshow("Stage 2 — ReID Tracking", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Stage 2 done.")
print(f"  Frames saved : {saved_count}  →  ./{OUTPUT_FRAMES}/")
print(f"  Video saved  : {OUTPUT_VIDEO}")