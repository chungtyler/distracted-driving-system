from distraction_level.low_distraction import show_indicator
from distraction_level.medium_distrction import audio_warning
from distraction_level.high_distraction import autonomous_takeover
from train.efficientnet_b0 import EfficientNet

import torch

# Initialize File Paths
path_to_video = 'example.mp4'

# Classification weights (higher = more distracted)
distraction_severity = {
    0: 1, # c0: safe driving
    1: 1, # c1: texting - right
    2: 1, # c2: talking on the phone - right
    3: 1, # c3: texting - left
    4: 1, # c4: talking on the phone - left
    5: 1, # c5: operating the radio
    6: 1, # c6: drinking
    7: 1, # c7: reaching behind
    8: 1, # c8: hair and makeup
    9: 1 # c9: talking to passenger
}
distraction_severity = torch.softmax(distraction_severity)

# Initialize classification model
efficientnet_b0 = EfficientNet(len(distraction_severity))
efficientnet_b0.load_weights('efficientnet_b0.pth')
model = efficientnet_b0.model
model.eval()

# Driver state history to video FPS and states
driver_state_history = {'fps': 30, 'states': []}

# Risk score calculation parameters
batch_size = 5 # Number of seconds to process to calculate risk score
current_frame_position = 0 # Most recent frame processed in driver_state_history

# Risk score thresholds
risk_score_threshold = {
    'SAFE': 0.05,
    'LOW': 0.1,
    'MEDIUM': 0.33,
    'HIGH': 0.67
}

def get_distraction_class(image):
    '''
    Get the distracted driving classification type (c0 - c9)
    '''
    with torch.no_grad():
        output = model(image) # Generate distribution
        _, predicted_class = torch.max(output, dim=1) # Get highest probable class
        return predicted_class.item()
    
def longest_distraction_duration(batch, distraction):
    '''
    Calculate the longest length of a specified distraction in the frame batch
    '''
    max_duration = 0
    current_duration = 0
    previous_distraction = -1
    
    for current_distraction in batch:
        if current_distraction == distraction and current_distraction == previous_distraction:
            current_duration += 1
        else:
            current_duration = 0

        max_duration = max(max_duration, current_duration)
        previous_distraction = current_distraction
    return max_duration

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

    current_batch = driver_state_history.states[current_frame_position:]
    risk_scores = dict.fromkeys(range(len(distraction_severity)), 0)

    for distraction in risk_scores:
        # Skip score calculations if distraction is not in batch
        if not distraction in current_batch:
            continue

        severity = distraction_severity[distraction]
        duration = longest_distraction_duration(current_batch, distraction)
        frequency = current_batch.count(distraction)
        risk_scores[distraction] = severity * duration * frequency

    return risk_scores

def generate_safety_action():
    '''
    Using distraction score threshold value and find proper plan of action
    '''
    risk_scores = calculate_risk_scores() # TODO normalize score or softmax?
    risk_score = risk_scores

    if risk_score < risk_score_threshold['SAFE']:
        print("Driving safely")
    elif risk_score < risk_score_threshold['LOW']:
        print("Showing indicator warning on dashboard!")
        show_indicator(True)
    elif risk_score < risk_score_threshold['MEDIUM']:
        print("Sending audible warning!")
        audio_warning()
    elif risk_score < risk_score_threshold['HIGH']:
        print("Performing safety autonomous takeover!")
        autonomous_takeover()

def main():
    show_indicator(False) # Display normal indicator dashboard
    autonomous_takeover() # Setup CARLO simulator driving

    # Loop for stream of images (video)
    for image, index in video:
        # TODO Show current video processed and classification
        distraction_class = get_distraction_class(image) # Get distraction class
        driver_state_history.states.append(distraction_class) # Update driver state history
        if (index % batch_size) == 0: # Only calculate score once enough frames collected
            generate_safety_action()

if __name__=='__main__':
    main()
