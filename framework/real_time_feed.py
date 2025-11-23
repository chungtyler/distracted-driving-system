import cv2
import torch
import numpy as np

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def preprocess_frame(frame, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype("float32") / 255.0
    frame = (frame - MEAN) / STD
    frame = np.transpose(frame, (2, 0, 1))

    # Convert to Float32 tensor
    tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)




def run_real_time_feed(video_path, model, class_names, device):
    cap = cv2.VideoCapture(video_path)
    model.to(device)
    model.eval()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = preprocess_frame(frame, device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_idx])

        label = f"{class_names[pred_idx]} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

        cv2.imshow("Driver Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
