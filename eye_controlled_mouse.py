import cv2
import mediapipe as mp
import autopy
import numpy as  np

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize Autopy
wscr, hscr = autopy.screen.size()
screen_size = autopy.screen.size()
smoothness = 5  # adjust this to change the smoothness of the cursor movement
framer = 100  # frame reduction
wcam = 640
hcam = 480
measurements = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# Define a function to move the mouse cursor
def move_cursor(wscr,x3, y3):
    #x = int(x * screen_size[0])
    #y = int(y * screen_size[1])
    autopy.mouse.move(wscr-x3, y3)


# Start webcam feed
cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)

# Start Mediapipe Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        # Get webcam feed
        _, frame = cap.read()
        cv2.rectangle(frame, (framer, framer), (wcam - framer, hcam - framer), (255, 0, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make detections
        results = face_mesh.process(frame)

        #eye closing opening part

        left_eye_landmarks = [249, 263, 466, 388, 387, 386]
        right_eye_landmarks = [7, 33, 246, 161, 160, 159]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


        def eye_aspect_ratio(eye):
            A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
            B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
            C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))

            ear = (A + B) / (2.0 * C)
            return ear




        # Check for eye landmarks
        left_eye = []
        right_eye = []
        if results.multi_face_landmarks is not None and len(results.multi_face_landmarks) > 0:
            left_eye_pts = [results.multi_face_landmarks[0].landmark[i] for i in left_eye_landmarks]
            right_eye_pts = [results.multi_face_landmarks[0].landmark[i] for i in right_eye_landmarks]

            left_ear = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)

            threshold = 3
            print(left_ear, right_ear)


            ##################################
            mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1))
            ####################################
            if left_ear < threshold and right_ear < threshold:
                b = 0
                cv2.putText(frame, "open", (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            else:
                b += 1
                cv2.putText(frame, "closed", (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                autopy.mouse.click()
            # Extract left and right eye landmarks
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    h,w,c = frame.shape
                    if idx == 159:  # left eye pupil landmark
                        left_eye.append((idx, lm.x * w, lm.y * h))
                    elif idx == 386:  # right eye pupil landmark
                        right_eye.append((idx, lm.x * w, lm.y * h))

            # Calculate average eye position
            if len(left_eye) > 0 and len(right_eye) > 0:
                eye_x = (left_eye[0][1] + right_eye[0][1]) / 2
                eye_y = (left_eye[0][2] + right_eye[0][2]) / 2

                x3 = np.interp(eye_x, (framer, wcam-framer), (0, wscr))
                y3 = np.interp(eye_y, (framer, hcam-framer), (0, hscr))

                # Move mouse cursor
                move_cursor(wscr,x3, y3)


        # Show webcam feed with Mediapipe annotations

        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('harvis ui', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

