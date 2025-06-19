'''
This script loads images from a dataset directory, extracts face encodings,
and stores them along with the corresponding names.

The dataset directory should be structured as follows:

dataset/
    friend_name_1/
        image1.jpg
        image2.png
        ...
    friend_name_2/
        image1.jpeg
        image2.jpg
        ...
    ...

Each subdirectory name under 'dataset/' is treated as the name of the person.
'''

import face_recognition
import os
import cv2 # OpenCV might be needed for frame extraction later, imported here for consistency
import pickle

def load_known_faces(dataset_path):
    """
    Loads known faces from the dataset directory.

    Args:
        dataset_path (str): The path to the main dataset directory.

    Returns:
        tuple: A tuple containing two lists: known_face_encodings and known_face_names.
    """
    known_face_encodings = []
    known_face_names = []

    print(f"Loading known faces from {dataset_path}...")

    # Check if the dataset path exists
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' not found.")
        return known_face_encodings, known_face_names

    # Loop through each person in the dataset directory
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)

        # Check if it's a directory
        if not os.path.isdir(person_dir):
            continue

        print(f"Processing images for: {person_name}")
        image_count = 0
        # Loop through each image file for the current person
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)

            # Check if it's a file and a common image type
            if os.path.isfile(image_path) and filename.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    # Load the image file
                    image = face_recognition.load_image_file(image_path)

                    # Find face encodings in the image
                    # We assume one face per image for the dataset for simplicity.
                    # If multiple faces are present, face_encodings will return a list.
                    # We'll take the first encoding found.
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        # Add the first found encoding and the name to our lists
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
                        image_count += 1
                    else:
                        print(f"  Warning: No face found in {filename}")
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
            else:
                print(f"  Skipping non-image file: {filename}")
        print(f"  Processed {image_count} images for {person_name}.")

    print(f"Finished loading known faces. Found {len(known_face_encodings)} encodings.")
    return known_face_encodings, known_face_names

if __name__ == "__main__":
    # --- IMPORTANT --- 
    # Replace './dataset' with the actual path to your dataset directory
    dataset_directory = './dataset' 
    # --- IMPORTANT --- 

    known_encodings, known_names = load_known_faces(dataset_directory)

    # Save the encodings and names for later use in the real-time identification script
    # Using pickle to save the data
    if known_encodings:
        encodings_file = 'known_faces.pkl'
        print(f"Saving encodings to {encodings_file}...")
        with open(encodings_file, 'wb') as f:
            pickle.dump({'encodings': known_encodings, 'names': known_names}, f)
        print("Encodings saved successfully.")
    else:
        print("No encodings were generated. Please check your dataset directory and images.")

