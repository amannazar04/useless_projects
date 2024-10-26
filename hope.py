import dlib
import cv2
import numpy as np

# Load the face detector and landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def calculate_eye_aspect_ratio(eye_points):
    # Vertical distances
    A = np.linalg.norm(np.array((eye_points[1].x, eye_points[1].y)) - np.array((eye_points[5].x, eye_points[5].y)))
    B = np.linalg.norm(np.array((eye_points[2].x, eye_points[2].y)) - np.array((eye_points[4].x, eye_points[4].y)))
    # Horizontal distance
    C = np.linalg.norm(np.array((eye_points[0].x, eye_points[0].y)) - np.array((eye_points[3].x, eye_points[3].y)))
    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold and frame count parameters
EAR_THRESHOLD = 0.15  # If EAR is below this, eyes are considered closed
CLOSED_FRAMES = 15  # Number of frames the eyes must be closed before taking a picture
frame_count = 0

# Open the webcam
cap = cv2.VideoCapture(0)#if using built-in webcam
looper=True
picno=1

while looper:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray)
    
    # Loop over detected faces
    for face in faces:
        # Get the landmarks
        landmarks = predictor(gray, face)
        
        # Indices for the left and right eye landmarks
        left_eye_indices = [36, 37, 38, 39, 40, 41]
        right_eye_indices = [42, 43, 44, 45, 46, 47]
        
        # Extract points for both eyes
        left_eye_points = [landmarks.part(i) for i in left_eye_indices]
        right_eye_points = [landmarks.part(i) for i in right_eye_indices]
        
        # Calculate EAR for both eyes
        left_ear = calculate_eye_aspect_ratio(left_eye_points)
        right_ear = calculate_eye_aspect_ratio(right_eye_points)
        
        # Average EAR of both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Check if the eyes are closed
        if ear < EAR_THRESHOLD:
            frame_count += 1
        else:
            frame_count = 0
        strname="captured"
        
        # If eyes have been closed for sufficient frames, take a picture
        if frame_count >= CLOSED_FRAMES:
            strnew=strname+str(picno)+".jpg"
            cv2.imwrite(strnew, frame)
            print("Picture taken! Eyes detected as closed.")
            frame_count = 0  # Reset the count after taking a picture
            print(picno)
            picno=int(picno)
            picno+=1
            
        
        # Draw rectangles around the eyes (optional visualization)
        #for point in left_eye_points + right_eye_points:
            #cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)
    
    # Display the frame with visualization
    cv2.imshow("Eye Detection", frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        looper=False
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
