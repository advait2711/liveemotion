import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import tkinter as tk
from tkinter import ttk
import speech_recognition as sr
from PIL import Image, ImageTk
import threading
from textblob import TextBlob

class EmotionAnalysisApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Multimodal Emotion Detection")

        self.face_model = model_from_json(open("livemodel.json", "r").read())
        self.face_model.load_weights('livemodel.h5')
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

       
        self.recognizer = sr.Recognizer()

       
        self.setup_ui()

        
        self.cap = cv2.VideoCapture(0)

        self.speech_running = False
        self.speech_thread = None

        
        self.thread_facial = threading.Thread(target=self.perform_emotion_analysis)  
        self.thread_facial.start()

    def setup_ui(self):
        custom_font = ("Arial", 14)

        
        self.facial_frame = ttk.Frame(self.master)
        self.facial_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.facial_label = ttk.Label(self.facial_frame, text="Facial Emotion:")
        self.facial_label.pack(pady=10)

        self.facial_canvas = tk.Canvas(self.facial_frame, width=640, height=480)
        self.facial_canvas.pack()
        self.error_label = ttk.Label(self.facial_frame, text="", foreground="red")
        self.error_label.pack(pady=10)

        
        self.speech_frame = ttk.Frame(self.master)
        self.speech_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.start_speech_button = ttk.Button(self.speech_frame, text="Start Speech Recognition", command=self.toggle_speech_recognition, style="Custom.TButton")
        self.start_speech_button.pack(pady=10)

        self.speech_label = ttk.Label(self.speech_frame, text="Speech Sentiment:")
        self.speech_label.pack(pady=10)

        self.result_label = ttk.Label(self.speech_frame, text="", font=custom_font)
        self.result_label.pack(pady=10)

    def perform_emotion_analysis(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                error_message = "Error: Unable to grab frame. Exiting..."
                print(error_message)
                self.error_label.config(text=error_message)
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_with_emotion = self.detect_and_draw_emotion(frame)

            
            img = Image.fromarray(frame_with_emotion)
            img = ImageTk.PhotoImage(img)
            self.facial_canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.facial_canvas.img = img

    def detect_and_draw_emotion(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_cascade.detectMultiScale(frame_gray, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            roi_gray = frame_gray[y:y+w, x:x+h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = self.face_model.predict(img_pixels)
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
            cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def perform_sentiment_analysis(self, text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def listen_and_analyze_sentiment(self):
        while self.speech_running:
            with sr.Microphone() as source:
                try:
                    self.speech_label.config(text="Listening...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    audio = self.recognizer.listen(source, timeout=5)
                    text = self.recognizer.recognize_google(audio)
                    sentiment = self.perform_sentiment_analysis(text)
                    self.result_label.config(text=f"You said: {text}\nSpeech Sentiment: {sentiment}")

                except sr.UnknownValueError:
                    self.result_label.config(text="Sorry, could not understand audio.")
                except sr.RequestError as e:
                    self.result_label.config(text=f"Could not request results from Google Speech Recognition service; {e}")

    def toggle_speech_recognition(self):
        if not self.speech_running:
            self.speech_running = True
            self.speech_thread = threading.Thread(target=self.listen_and_analyze_sentiment)
            self.speech_thread.start()
        else:
            self.speech_running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionAnalysisApp(root)
    root.mainloop()
