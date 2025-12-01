from framework.distraction_level.low_distraction import show_indicator
from framework.distraction_level.medium_distrction import audio_warning
from framework.distraction_level.high_distraction import start_simulator, activate_autonomous_takeover
from train.efficientnet_b0 import EfficientNet

import torch
import torch.nn.functional as F
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import time

global root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize File Paths
MODEL_PATH = os.path.join(project_root, "models", "efficientnet_b0.pth")
VIDEO_PATH = os.path.join(project_root, "input_video.mp4")

# Classification weights [1 to 10] (higher = more distracted)
distraction_severity = {
    0: 0, # c0: safe driving
    1: 9, # c1: texting - right
    2: 5, # c2: talking on the phone - right
    3: 9, # c3: texting - left
    4: 5, # c4: talking on the phone - left
    5: 3, # c5: operating the radio
    6: 4, # c6: drinking
    7: 10, # c7: reaching behind
    8: 8, # c8: hair and makeup
    9: 5 # c9: talking to passenger
}
distraction_severity = torch.tensor([0, 9, 5, 9, 5, 3, 4, 10, 8, 5], dtype=torch.float32) / 10.0
#distraction_severity = torch.softmax(distraction_severity, dim=0)

# Initialize classification model
efficientnet_b0 = EfficientNet(len(distraction_severity))
efficientnet_b0.load_weights(MODEL_PATH)
model = efficientnet_b0.model
model.eval()

is_autonomous_takeover_active = False

# Driver state history to video FPS and states
driver_state_history = {'fps': 30, 'states': []}

# Risk score calculation parameters
batch_size = 5 # Number of seconds to process to calculate risk score
current_frame_position = 0 # Most recent frame processed in driver_state_history
global previous_risk_score
previous_risk_score = 0
decay_factor = 0.8

# Risk score thresholds
risk_score_threshold = {
    'LOW': 0.25,
    'MEDIUM': 0.5,
    'HIGH': 0.75
}

# Process video feed
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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

def show_feed(confidence, distraction, frame):
    #print(confidence)
    label = f"{CLASS_NAMES[distraction].upper()} ({confidence*100:.1f}%)"
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

    cv2.imshow("Driver Monitoring", frame)
    cv2.moveWindow("Driver Monitoring", 175, 0)
import matplotlib
matplotlib.use("TkAgg")
_, ax = plt.subplots(figsize=(9.6, 5.6), dpi=100)
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+250+875")
ax.grid(True)
ax.set_ylabel('Risk Score [0, 1]')
ax.set_xlabel('Time (s)')
ax.set_title('Risk Score vs Time')
ax.set_ylim((0, 1))
for label, threshold in risk_score_threshold.items():
    #print(label)
    plt.axhline(y=threshold, color='r', linestyle='--', label=label)

total_time = []
total_risk_score = []

def plot_risk_score(time, risk_score):
    total_time.append(time)
    total_risk_score.append(risk_score)
    ax.plot(total_time, total_risk_score, color='blue')
    plt.draw()

def preprocess_frame(frame, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    frame = (frame - MEAN) / STD
    frame = np.transpose(frame, (2, 0, 1))
    tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)

def get_distraction_class(image):
    '''
    Get the distracted driving classification type (c0 - c9)
    '''
    with torch.no_grad():
        output = model(image) # Generate distribution
        # _, predicted_class = torch.max(output, dim=1) # Get highest probable class
        # return predicted_class.item()
        probability = F.softmax(output, dim=1)
        confidence, classification = torch.max(probability, dim=1)
        return confidence.item(), classification.item()
    
def longest_distraction_duration(batch, distraction):
    '''
    Calculate the longest length of a specified distraction in the frame batch in seconds
    '''
    max_duration = 0
    current_duration = 0
    previous_distraction = None
    
    for current_distraction in batch:
        if current_distraction == distraction and current_distraction == previous_distraction:
            current_duration += 1
        elif current_distraction == distraction:
            current_duration = 1
        else:
            current_duration = 0

        max_duration = max(max_duration, current_duration)
        previous_distraction = current_distraction
    return max_duration / driver_state_history['fps']

def calculate_risk_scores():
    # TODO calculate overall driver distraction score (SIGMOID)
    # TODO some score based on duration of classification or how frequent it appears?
    # TODO reset or reduce distraction driver attention level decay rate?
    # TODO Calculate TOTAL score

    # in state history get last X seconds of data
    # Severity defined in severity based on real-world statistics [0, 1] normalize it
    # Duration over time horizon can be calculated on highest duration or total sum [0, 1] (sum or max duration/time_horizon)
    # Frequency over time horizon can be calculated based on how frequent (stable the model is) count as one for coulpe count, counts / total number of distractions
    # Multiply together and crunch into a normalization function (e.g. sigmoid with certain scaling)

    current_batch = driver_state_history['states'][current_frame_position:]
    risk_scores = dict.fromkeys(range(len(distraction_severity)), 0)
    for distraction in risk_scores:
        # Skip score calculations if distraction is not in batch
        if not distraction in current_batch:
            continue

        severity = distraction_severity[distraction]
        duration = longest_distraction_duration(current_batch, distraction) / batch_size
        frequency = current_batch.count(distraction) / len(current_batch)
        risk_scores[distraction] = severity * duration * frequency * 10
        #print(f"Severity: {severity} || Duration: {duration} || Frequency: {frequency}")
    return risk_scores

def normalize_risk_scores(risk_scores, decay_factor=0.9, c1=1, c2=2.5):
    risk_score = sum([looped_risk_score for looped_risk_score in risk_scores.values()])
    decayed_risk_score = decay_factor * previous_risk_score + risk_score # Exponential decay
    normalized = 1 / (1 + np.exp(-c1 * (np.asarray(decayed_risk_score) - c2))) # Sigmoid normalization
   #print(f"Risk Scores: {risk_scores}")
    #print(f"Risk Score: {risk_score:.3f} || Decay Risk Score: {decayed_risk_score:.3f} || Norm: {normalized:.3f}")
    return normalized


def generate_safety_action():
    '''
    Using distraction score threshold value and find proper plan of action
    '''
    risk_scores = calculate_risk_scores() # TODO normalize score or softmax?
    risk_score = normalize_risk_scores(risk_scores)
    global previous_risk_score
    previous_risk_score = risk_score
    time = len(driver_state_history['states']) / driver_state_history['fps']
    #print(risk_score)
    plot_risk_score(time, risk_score)
    #plt.pause(0.001)
    root.after(0, lambda: plt.draw())

    if risk_score >= risk_score_threshold['HIGH']:
        show_indicator(True)
        global is_autonomous_takeover_active
        print(f"[ACTION] ☢️  HIGH: Autonomous takeover! || Risk Score: [{risk_score:.2f} / 1.00]")
        if not is_autonomous_takeover_active:
            is_autonomous_takeover_active = True
            #print("[ACTION] HIGH: Autonomous takeover!")
            activate_autonomous_takeover(risk_score)
    elif risk_score >= risk_score_threshold['MEDIUM']:
        print(f"[ACTION] 🛑 MEDIUM: Audible warning! || Risk Score: [{risk_score:.2f} / 1.00]")
        clamped_score = (risk_score - risk_score_threshold['MEDIUM']) / (risk_score_threshold['HIGH'] - risk_score_threshold['MEDIUM'])
        
        audio_warning(clamped_score, max(risk_scores, key=risk_scores.get))
        show_indicator(True)
    elif risk_score >= risk_score_threshold['LOW']:
        print(f"[ACTION] ⚠️  LOW: Dashboard indicator! || Risk Score: [{risk_score:.2f} / 1.00]")
        show_indicator(True)
    else:
        # If it's not high, medium, or low risk, it's safe driving
        #print("Driving safely")
        show_indicator(False) # Turn off the indicator if it was on

def video_loop():
        global capture, all_frames, index, last_frame, counter
        
        # Loop for stream of images (video)
        #index = 0
        done = False
        X = 10
        #while not done:
        # TODO Show current video processed and classification
        ret, frame = capture.read()
        #print(len(all_frames))
        #print(batch_size)

        if not ret:
            final_frames = all_frames[-X:]
            frame = final_frames[index%X]
            index += 1
        else:
            all_frames.append(frame)

        if keyboard.is_pressed('q'):
            done = True
        processed_frame = preprocess_frame(frame, 'cpu')
        confidence, distraction_class = get_distraction_class(processed_frame) # Get distraction class
        driver_state_history['states'].append(distraction_class) # Update driver state history
        if (counter % batch_size) == 0: # Only calculate score once enough frames collected
            generate_safety_action()
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.flush_events()

        show_feed(confidence, distraction_class, frame)
        counter += 1

        #time.sleep(1/driver_state_history['fps'])
        # else:
        #     if index == len(final_frames):
        #         final_frames = all_frames[-X:]
        #         frame = final_frames[index]
        #         #for frame in final_frames:
        #         if keyboard.is_pressed('q'):
        #             done = True
        #         processed_frame = preprocess_frame(frame, 'cpu')
        #         distraction_class = get_distraction_class(processed_frame) # Get distraction class
        #         driver_state_history['states'].append(distraction_class) # Update driver state history
        #         if (len(all_frames) % batch_size) == 0: # Only calculate score once enough frames collected
        #             generate_safety_action()
        #             plt.gcf().canvas.draw_idle()
        #             plt.gcf().canvas.flush_events()

        #         show_feed(distraction_class, frame)

        #         time.sleep(1/driver_state_history['fps'])
        #         #break
        #         index += 1
        root.after(int(1000/driver_state_history['fps']), video_loop)

import threading

def run_video():
    video_loop()               # process one loop iteration
    root.after(1, run_video)   # schedule next

def main():
    global root
    plt.ion()
    show_indicator(False) # Display normal indicator dashboard
    root = start_simulator()
    #threading.Thread(target=video_loop, daemon=True).start()
    #run_video()
    # Check for video capture
    global capture, all_frames, ret, last_frame, index, counter
    ret = True
    index = 0
    counter = 0
    all_frames = []
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return
    os.system('cls')
    root.after(0, video_loop)

    def keep_plot_alive():
        plt.pause(0.001)
        root.after(100, keep_plot_alive)

    keep_plot_alive()

    root.mainloop()
    print("YES")
    plt.ioff()
    plt.show()

if __name__=='__main__':
    main()
