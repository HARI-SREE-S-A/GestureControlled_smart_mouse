import cv2
import mediapipe as mp
import autopy


# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize Autopy
wscr,hscr = autopy.screen.size()
screen_size = autopy.screen.size()
smoothness = 5 # adjust this to change the smoothness of the cursor movement
framer = 100 #frame reduction

# Define a function to move the mouse cursor
def move_cursor(x, y):
    x = int(x * screen_size[0])
    y = int(y * screen_size[1])
    autopy.mouse.move(x, y)

# Start webcam feed
cap = cv2.VideoCapture(0)

# Start Mediapipe Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        # Get webcam feed
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make detections
        results = face_mesh.process(frame)

        # Check for eye landmarks
        left_eye = []
        right_eye = []
        if results.multi_face_landmarks:
            # Extract left and right eye landmarks
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 159: # left eye pupil landmark
                        left_eye.append((lm.x, lm.y))
                    elif idx == 386: # right eye pupil landmark
                        right_eye.append((lm.x, lm.y))

            # Calculate average eye position
            if len(left_eye) > 0 and len(right_eye) > 0:
                eye_x = (left_eye[0][0] + right_eye[0][0]) / 2
                eye_y = (left_eye[0][1] + right_eye[0][1]) / 2

                # Move mouse cursor
                move_cursor(eye_x, eye_y)

        # Show webcam feed with Mediapipe annotations
        mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('harvis ui', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

