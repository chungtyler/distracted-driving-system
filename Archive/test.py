import pyttsx3
 

text = "Phone Use Detected, Please keep your hands on the wheel"

engine = pyttsx3.init()
voices = engine.getProperty("voices")
for i, v in enumerate(voices):
    print(i, v.name, v.id)
engine.setProperty('voice', voices[1].id)
engine.say(text)
engine.runAndWait()
engine.stop()