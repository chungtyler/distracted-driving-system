import pyttsx3
import threading
import time
import random

global last_run
last_run = None
low_phone = ["Just a reminder: keep your eyes on the road.", 
             "Hey, watch out for phone use while driving.", 
             "Friendly reminder: focus on driving, not your phone."] 

low = ["Keep your eyes on the road, please.", 
       "Make sure to face forward while driving.", 
       "Heads up, try to pay attention to the road."] 

high_phone = ["High Risk! Stop using your phone immediately!", 
              "Danger! Phone use detected, pull over if needed!", 
              "Critical alert, hands on the wheel, eyes on the road!"] 

high = ["High Risk! You’re not facing the road!", 
        "Danger! Eyes off the road detected, focus immediately!",
        "Critical alert: pay full attention to driving now!"]

def audio_warning(risk_score, distraction):
    def run():
        global last_run
        if last_run is None or ((time.time() - last_run) > 7):
            last_run = time.time()
            if risk_score < 0.5:  
                if distraction < 5:
                    text = random.choice(low_phone)
                else:
                    text = random.choice(low)
            else:
                if distraction < 5:
                    text = random.choice(high_phone)
                else:
                    text = random.choice(high)
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id)
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()





