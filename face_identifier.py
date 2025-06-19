
import face_recognition
import cv2
import numpy as np
import pickle
import os
import time

def load_known_faces(encodings_file='known_faces.pkl'):
    """Loads known face encodings and names from a pickle file."""
    print(f"Loading known faces from {encodings_file}...")
    try:
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data['names'])} known faces.")
        return data['encodings'], data['names']
    except FileNotFoundError:
        print(f"Error: Encodings file 	{encodings_file}	 not found. ")
        print("Please run the face_encoder.py script first to generate it.")
        return [], []
    except Exception as e:
        print(f"Error loading {encodings_file}: {e}")
        return [], []

def identify_faces(input_source='webcam', video_path=None, encodings_file='known_faces.pkl', tolerance=0.6):
    """
    Performs real-time face identification from webcam or video file.

    Args:
        input_source (str): \'webcam\' or \'video\'.
        video_path (str, optional): Path to the video file if input_source is \'video\'.
        encodings_file (str): Path to the pickle file containing known face encodings.
        tolerance (float): How much distance between faces to consider it a match. Lower is stricter.
    """
    known_face_encodings, known_face_names = load_known_faces(encodings_file)

    if not known_face_encodings:
        print("Cannot proceed without known face encodings.")
        return

    # Initialize video capture
    if input_source == 'webcam':
        video_capture = cv2.VideoCapture(0) # 0 is usually the default webcam
        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            return
        print("Starting webcam feed...")
    elif input_source == 'video' and video_path:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        print(f"Processing video file: {video_path}...")
    else:
        print("Error: Invalid input source or missing video path.")
        return

    # Initialize variables for processing
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    frame_count = 0
    start_time = time.time()

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if not ret:
            print("End of video file reached or error reading frame.")
            break

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # Use cv2.COLOR_BGR2RGB directly
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
                name = "Intruder" # Default to Intruder

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                
                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit \'q\' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
        
        frame_count += 1

    # Release handle to the webcam/video file and close windows
    video_capture.release()
    cv2.destroyAllWindows()
    end_time = time.time()
    print(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # --- Configuration --- 
    # Set input_source to \'webcam\' or \'video\'
    INPUT_TYPE = 'webcam' # or \'video\'
    
    # If using \'video\', provide the path to your video file
    VIDEO_FILE_PATH = '/path/to/your/video.mp4' # Replace with actual path if INPUT_TYPE is \'video\'
    
    # Path to the generated encodings file
    ENCODINGS_PATH = 'known_faces.pkl'
    
    # Recognition tolerance (lower is stricter, 0.6 is a good default)
    RECOGNITION_TOLERANCE = 0.6
    # --- Configuration --- 

    if INPUT_TYPE == 'webcam':
        identify_faces(input_source='webcam', encodings_file=ENCODINGS_PATH, tolerance=RECOGNITION_TOLERANCE)
    elif INPUT_TYPE == 'video':
        identify_faces(input_source='video', video_path=VIDEO_FILE_PATH, encodings_file=ENCODINGS_PATH, tolerance=RECOGNITION_TOLERANCE)
    else:
        print(f"Error: Invalid INPUT_TYPE specified: {INPUT_TYPE}. Choose \'webcam\' or \'video\'.")

