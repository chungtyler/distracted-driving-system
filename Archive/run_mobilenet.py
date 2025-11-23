## Can delete this code since the real time video is implemented in run_full_system.pv but idk how LOL

import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from train.mobilenet_b0 import create_mobilenet_b0
from Archive.real_time_feed import run_real_time_feed

MODEL_PATH = os.path.join(project_root, "models", "mobilenet_b0.pt")
VIDEO_PATH = os.path.join(project_root, "input_video.mp4")

# Same order as training (c0..c9)
class_names = [
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build architecture
    model = create_mobilenet_b0(num_classes=10, pretrained=False)

    # Load checkpoint (dict with 'model_state', 'classes', etc.)
    print(f"Loading checkpoint from: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    # Run the real-time video classification
    run_real_time_feed(
        video_path=VIDEO_PATH,
        model=model,
        class_names=class_names,
        device=device,
    )

if __name__ == "__main__":
    main()
