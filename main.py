import cv2
import mediapipe as mp
import pyttsx3
import threading
import time

# Voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak_async(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Webcam setup
cap = cv2.VideoCapture(0)
cv2.namedWindow("Concentration Tracker", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Concentration Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Head tilt
LEFT_EYE = [33]
RIGHT_EYE = [263]

last_alert_time = 0
head_movement_count = 0
was_tilted_last_frame = False  # to avoid double-counting

def get_head_tilt(landmarks):
    left_y = landmarks[LEFT_EYE[0]].y
    right_y = landmarks[RIGHT_EYE[0]].y
    return "Head Tilted" if abs(left_y - right_y) > 0.03 else "Head Center"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            head_pos = get_head_tilt(landmarks)

            if head_pos == "Head Tilted":
                status = "Distracted"
                color = (0, 0, 255)

                if not was_tilted_last_frame:
                    head_movement_count += 1
                    was_tilted_last_frame = True

                if time.time() - last_alert_time > 5:
                    speak_async("Please concentrate")
                    last_alert_time = time.time()

            else:
                status = "Focused"
                color = (0, 255, 0)
                was_tilted_last_frame = False

            # Display info
            cv2.putText(frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            cv2.putText(frame, f"Head: {head_pos}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Head Moves: {head_movement_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    else:
        cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Concentration Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
