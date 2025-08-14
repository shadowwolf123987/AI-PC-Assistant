
#User Customisation Variables

#Imports
import os
import io
import json
import base64
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pyautogui
from PIL import Image
import cv2 as cv
import numpy as np
from openai import OpenAI
from elevenlabs import ElevenLabs
from elevenlabs import play, save, stream, Voice, VoiceSettings
import pydub, ffmpeg

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

#Function Defs

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

def sendToAI(img):

    sysMessage = f"""

You are an AI Companion that receives screenshots of a pc screen and based on those screenshots, you should talk to the player using the details from the screenshot or just make general small talk where suitable

Your answers should be short (20 words maximum), you should ignore background details and focus on the features that stand out. Dont repeat yourself 

You should speak based on the personality info below:

{char["ai_description"]}

"""

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=15, optimize=True)
    buffer.seek(0)

    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    b64Url = f"data:image/jpeg;base64,{b64}"

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": sysMessage}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": b64Url
                    }
                ]
            }
        ]
    )

    text = response.output_text

    return text

def loadCharacterProfile(charName="zero-two"):
    charName = charName.lower().replace(" ","-")
    charPath = f"characters/{charName}.json"

    global char
    with open(charPath, "r", encoding="utf-8") as file:
        char = json.load(file)

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

    play(audio)

    return audio

def main():
    x=0
    preState = 0
    frameLimit = 3 #How Many Frames to Check before allowing a change

    prevImg = None
    #cv.namedWindow("win", cv.WINDOW_NORMAL)

    heartbeat = datetime.now() + timedelta(0,60) #Sends screenshot after 60 sec idle time
    loadCharacterProfile("zero-two")
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

main()