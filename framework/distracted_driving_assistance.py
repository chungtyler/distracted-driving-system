from distraction_level.low_distraction import show_indicator
from distraction_level.medium_distrction import audio_warning
from distraction_level.high_distraction import autonomous_takeover

import torch

# Initialize File Paths
path_to_video = 'example.mp4'
path_to_model = 'efficientnet_b0.pth'

# Initialize classification model
model = torch.load(path_to_model)
model.eval()

# Classification weights (higher = more distracted)
distraction_severity = [
    1,  # c0: safe driving
    2,  # c1: texting - right
    3,  # c2: talking on the phone - right
    4,  # c3: texting - left
    5,  # c4: talking on the phone - left
    6,  # c5: operating the radio
    7,  # c6: drinking
    8,  # c7: reaching behind
    9,  # c8: hair and makeup
    10  # c9: talking to passenger
]

# Driver state history to store weighted score and frequency
driver_state_history = {}
for i in range(len(distraction_severity)):
    driver_state_history[str(i)] = {'score': 0, 'count': 0}

def get_distraction_class(image):
    '''
    Get the distracted driving classification type (c0 - c9)
    '''
    with torch.no_grad():
        output = model(image) # Generate distribution
        _, predicted_class = torch.max(output, dim=1) # Get highest probable class
        return predicted_class.item()
    
def update_driver_state_history(distraction_class):
    '''
    Update the driver state history for the class (weighted score and frequency count)
    '''
    distraction_weight = distraction_severity[distraction_class] # Get distraction weight

    # Store current image to history
    class_state_history = driver_state_history[str(distraction_class)]
    class_state_history['score'] += distraction_weight
    class_state_history['count'] += 1

def calculate_risk_score():
    # TODO calculate overall driver distraction score (SIGMOID)
    # TODO some score based on duration of classification or how frequent it appears?
    # TODO reset or reduce distraction driver attention level decay rate?
    # TODO Calculate TOTAL score

    # in state history get last X seconds of data
    # Severity defined in severity based on real-world statistics [0, 1] normalize it
    # Duration over time horizon can be calculated on highest duration or total sum [0, 1] (sum or max duration/time_horizon)
    # Frequency over time horizon can be calculated based on how frequent (stable the model is) count as one for coulpe count, counts / total number of distractions
    # Multiply together and crunch into a normalization function (e.g. sigmoid with certain scaling)
    risk_score = severity * duration * frequency
    pass

def generate_safety_action(risk_score):
    '''
    Using distraction score threshold value and find proper plan of action
    '''
    if risk_score < 0.1:
        print("Showing indicator warning on dashboard!")
        show_indicator(True)
    elif risk_score < 0.33:
        print("Sending audible warning!")
        audio_warning()
    elif risk_score < 0.67:
        print("Performing safety autonomous takeover!")
        autonomous_takeover()

def main():
    show_indicator(False) # Display normal indicator dashboard
    autonomous_takeover() # Setup CARLO simmulator driving

    # Loop for stream of images (video)
    for image in video:
        # TODO Show current video processed and classification
        distraction_class = get_distraction_class(image) # Get distraction class
        update_driver_state_history(distraction_class) # Update driver state history
        risk_score = calculate_risk_score()
        generate_safety_action(risk_score)

if __name__=='__main__':
    main()
