import os
import sys
import cv2
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from train.mobilenet_b0 import create_mobilenet_b0
from framework.distraction_level.low_distraction import show_indicator
# from framework.distraction_level.medium_disctriction import audio_warning
# from framework.distraction_level.high_distraction import autonomous_takeover

MODEL_PATH = os.path.join(project_root, "models", "mobilenet_b0.pt")
VIDEO_PATH = os.path.join(project_root, "input_video.mp4")

# CLASS NAMES
CLASS_NAMES = [
    "c0: safe driving",
    "c1: texting right",
    "c2: talking phone right",
    "c3: texting left",
    "c4: talking phone left",
    "c5: operating radio",
    "c6: drinking",
    "c7: reaching behind",
    "c8: hair/makeup",
    "c9: talking passenger",
]

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def preprocess_frame(frame, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    frame = (frame - MEAN) / STD
    frame = np.transpose(frame, (2, 0, 1))
    tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)

# ---------------------------------------------------------
# RISK SCORING STATE
# ---------------------------------------------------------
history = []
fps = 30
batch_seconds = 5
risk_thresholds = {
    "SAFE": 0.05,
    "LOW": 0.1,
    "MEDIUM": 0.33,
    "HIGH": 0.67,
}

severity_raw = torch.tensor(
    [0.2, 1.0, 1.0, 1.0, 1.0, 0.6, 0.7, 1.0, 1.0, 0.8], dtype=torch.float32
)
severity = torch.softmax(severity_raw, dim=0)


def longest_run(batch, cls):
    max_len = 0
    cur_len = 0
    prev = None
    for v in batch:
        if v == cls and v == prev:
            cur_len += 1
        elif v == cls:
            cur_len = 1
        else:
            cur_len = 0
        max_len = max(max_len, cur_len)
        prev = v
    return max_len


def compute_risk(batch):
    n = len(batch)
    if n == 0:
        return 0.0

    scores = []
    for cls in range(10):
        if cls not in batch:
            continue
        dur = longest_run(batch, cls) / n
        freq = batch.count(cls) / n
        combined = 0.6 * dur + 0.4 * freq
        scores.append(float(severity[cls]) * combined)

    return max(scores) if scores else 0.0


def apply_risk_action(risk):
    print(f"[RISK] Score: {risk:.3f}")

    if risk < risk_thresholds["SAFE"]:
        show_indicator(False)
        return

    if risk < risk_thresholds["LOW"]:
        print("[ACTION] LOW: Dashboard indicator")
        show_indicator(True)
        return

    if risk < risk_thresholds["MEDIUM"]:
        print("[ACTION] MEDIUM: Audible warning")
        show_indicator(True)
        #audio_warning()
        return

    print("[ACTION] HIGH: Autonomous takeover")
    show_indicator(True)
    #autonomous_takeover()


# ---------------------------------------------------------
# MAIN REAL-TIME LOOP
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = create_mobilenet_b0(num_classes=10, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    global fps
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    batch_size_frames = fps * batch_seconds

    print("🎥 Running real-time system... Press 'q' to stop.")

    show_indicator(False)
    frame_idx = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess + predict
            tensor = preprocess_frame(frame, device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, pred])

            # Update history for risk scoring
            history.append(pred)
            if len(history) > batch_size_frames:
                history.pop(0)

            # Every batch window, compute risk
            if frame_idx % batch_size_frames == 0 and frame_idx > 0:
                risk = compute_risk(history)
                apply_risk_action(risk)

            # Draw label on video
            label = f"{CLASS_NAMES[pred]} ({conf*100:.1f}%)"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

            cv2.imshow("Driver Monitoring", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
