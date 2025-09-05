
#User Customisation Variables

#Imports
import os
import io
import json
import base64
import random
import time
from datetime import datetime, timedelta
from timeit import repeat
from dotenv import load_dotenv
import threading
import pyautogui
from PIL import Image
import cv2 as cv
import numpy as np
from openai import OpenAI
from elevenlabs import ElevenLabs
from elevenlabs import play, save, stream, Voice, VoiceSettings
import pydub, ffmpeg
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play as pydub_play

# File Paths
ffmpegPath = r"libs/ffmpeg/bin/"

#ENV Variables
load_dotenv()

elevenApiKey = os.getenv("ElevenApiKey")

os.environ['PATH'] = ffmpegPath

# INIT Code
client = OpenAI()

elevenlabs = ElevenLabs(
  api_key=elevenApiKey,
)

pydub.AudioSegment.converter = ffmpegPath + "ffmpeg.exe"

recognizer = sr.Recognizer()

#Function Defs

def userSetup():
    return

def keywordTrigger(text):
    keywords = char["trigger_words"]
    keywords = keywords.split(",")

    text = text.lower()

    for keyword in keywords:
        keyword = keyword.lower().strip()

        if "&" in keyword:
            state = True
            reqKeys = keyword.split("&") #used to ignore punctuation EG: zero&two , Zero, two = same
            for reqKey in reqKeys:
                if reqKey not in text:
                    state = False
            if state == True:
                return True

        if keyword in text:
            return True

    return False

def screenshot():
    #Take Screenshot
    img = pyautogui.screenshot()

    # Lower Image Quality
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=15, optimize=True)
    buffer.seek(0)

    #Reopen lower quality image
    img = Image.open(buffer)

    return img

def grayscale(img):

    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)

    return img

# For Viewing the hists for testing purposes
def drawHist(hist, height=150, width=256, color=(255,255,255)):
    """Render a histogram as an image."""
    hist_img = np.zeros((height, width, 3), dtype=np.uint8)
    cv.normalize(hist, hist)
    bin_w = int(round(width / hist.shape[0]))
    for i in range(1, len(hist)):
        cv.line(hist_img,
                (bin_w*(i-1), height - int(hist[i-1]*height)),
                (bin_w*(i), height - int(hist[i]*height)),
                color, 2)
    return hist_img

def compareFrame(frame, prevFrame, binNum=32, histRange=[0,256], threshold=0.40):

    #Bin Num : ?
    #Hist Range : Intensity Range
    #Threshold : Detects changes : Higher is stricter

    # Convert Frames into Histograms
    histCurr = cv.calcHist([frame], [0], None, [binNum], histRange)
    histPrev = cv.calcHist([prevFrame], [0], None, [binNum], histRange)
    
    # Normalize for comparison
    cv.normalize(histCurr, histCurr)
    cv.normalize(histPrev, histPrev)

    # Compare using Bhattacharyya distance
    dist = cv.compareHist(histPrev, histCurr, cv.HISTCMP_BHATTACHARYYA)

    #Show Both Histograms Stacked for Testing Purposes
    """
    histCurrImg = drawHist(histCurr, color=(0,255,0))
    histPrevImg = drawHist(histPrev, color=(255,0,0))
    stacked = np.vstack((histPrevImg, histCurrImg))
    cv.imshow("win",stacked)
    cv.waitKey(1)
    """

    # Return True if difference is big enough
    return dist > threshold, float(dist)

def sendToAI(img, text=None):

    sysMessage = f"""

You are an AI Companion that receives screenshots of a pc screen and based on those screenshots, you should talk to the player using the details from the screenshot or just make general small talk where suitable

Your answers should be short (30 words maximum), you should provide unique, personal conversation using the screenshot attached as context 

You should speak based on the personality info below:

{char["ai_settings"]["ai_description"]}

"""

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=15, optimize=True)
    buffer.seek(0)

    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    b64Url = f"data:image/jpeg;base64,{b64}"

    msgs = [
        {"role": "system", "content": [{"type": "input_text", "text": sysMessage}]},
        {"role": "user", "content": [{ "type": "input_image", "image_url": b64Url} ]},
    ]

    if text is not None:
        msgs.append({"role": "user", "content": [{"type": "input_text", "text": text}] } )

    response = client.responses.create(
        model="gpt-4o-mini",
        input=msgs,

        temperature= int(char["ai_settings"]["temperature"]),
        top_p= int(char["ai_settings"]["top_p"])
    )

    text = response.output_text

    return text

def loadCharacterProfile(charName="zero-two"):
    charName = charName.lower().replace(" ","-")
    charPath = f"characters/{charName}.json"

    global char
    with open(charPath, "r", encoding="utf-8") as file:
        char = json.load(file)

def transcribe(audio):

    wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
    wav = io.BytesIO(wav_bytes)
    wav.name = "speech.wav"
    wav.seek(0)

    #play(wav)

    """
    #OpenAI Whispers Transcription
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=wav,
        prompt=f"Each transcription should start with FALSE:text or TRUE:text if any of these {char['trigger_words']} are detected. The audio is in English, and should be translated in its raw form as accurately as possible."
    )
    """

    # ElevenLabs Transcription
    transcription = elevenlabs.speech_to_text.convert(
        file=wav,
        model_id="scribe_v1", # Transcription Model
        tag_audio_events=True, # Tag audio events like laughter, applause, etc.
        language_code="en", # Language of the audio file. If set to None, the model will detect the language automatically.
        diarize=False, # Whether to annotate who is speaking
    )

    return transcription.text

def speak(text):

    audio = elevenlabs.text_to_speech.convert(
        voice_id = char['voice_id'],
        text = text,
        model_id = char['model_id'],
        voice_settings={
            "stability": char['voice_settings']['stability'],
            "similarity_boost": char['voice_settings']['similarity_boost'],
            "style": char['voice_settings']['style'],
            "use_speaker_boost": char['voice_settings']['use_speaker_boost'],
            "speed": char['voice_settings']['voice_speed']
        },
    )

    #Splits generator object into bytes
    audio_bytes = b"".join(audio)

    # Load bytes into AudioSegment
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

    boosted_audio = audio_segment + int(char["voice_settings"]["db_volume_addition"]) #In dBs, Adds to original volume

    # Play boosted audio
    pydub_play(boosted_audio)

    return boosted_audio

# Different Implementations for triggering the AI
def cvMain():
    x=0
    preState = 0
    frameLimit = 3 #How Many Frames to Check before allowing a change

    prevImg = None
    #cv.namedWindow("win", cv.WINDOW_NORMAL)

    heartbeat = datetime.now() + timedelta(0,60) #Sends screenshot after 60 sec idle time
    loadCharacterProfile()
    while True:
        img = screenshot()
        analImg = grayscale(img)

        if datetime.now() > heartbeat:
            text = sendToAI(img)

            print(f"HEART {datetime.now()}: {text}\n")
            speak(text)

            heartbeat = datetime.now() + timedelta(0,60)

        #cv.imshow("win",analImg)
        #cv.waitKey(1)

        if prevImg is None:
            prevImg = analImg
            continue
    
        if preState != 0:
            state, diff = compareFrame(analImg, savedImg)
        else:
            state, diff = compareFrame(analImg, prevImg)

        #Occurs when enough frames with changes have been detected
        if preState == frameLimit and state == False:

            text = sendToAI(img)

            print(f"EVENT {datetime.now()}: {text}\n")
            speak(text)

            heartbeat = datetime.now() + timedelta(0,65)
            preState = 0
            time.sleep(5)

        #Occurs when Frame Verification has occurred and a candidate passes the check
        if preState != 0 and state == False:
            preState += 1

        #Occurs when Frame Verification has occurred and a candidate fails the check
        if preState != 0 and state == True:
            preState = 0

        #Occurs when first change is detected
        if preState == 0 and state == True:
            savedImg = analImg
            preState += 1

        prevImg = analImg
        time.sleep(0.2)

def audioMain():

    loadCharacterProfile()
    while True:
        with sr.Microphone() as source:
            #Adjust for background noise
            recognizer.dynamic_energy_threshold = True
            recognizer.energy_threshold = 400  # start point; tune 200–1200
            recognizer.pause_threshold = 0.65  # How long to wait before ending phrase
            recognizer.non_speaking_duration = 0.5 # How much sielnce to ignore before recording
            recognizer.phrase_threshold = 0.15  # require ~150ms of speech to start
            recognizer.adjust_for_ambient_noise(source, duration=0.8)

            #Start listening
            print("Recording...")

            recorded_audio = recognizer.listen(source)

            img = screenshot()

            speech = transcribe(recorded_audio)

            if keywordTrigger(speech) == True:
                print(f"KEY: {speech}")
                
                text = sendToAI(img, text=speech)

                speak(text)

            else:
                print(f"NO KEY: {speech}")
    return

def timedMain():

    loadCharacterProfile()
    while True:
        sleepTime = random.randint(30,60)
        time.sleep(sleepTime)
        print("Main Screenshot Thread")
        img = screenshot()

        text = sendToAI(img)
        print(f"{text} \n")

        speak(text)

# Parallel Threads
speechThread = threading.Thread(target=audioMain)
speechThread.start()

#cvMain()
#audioMain()
timedMain()