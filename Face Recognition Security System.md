# Realtime facial recognition ai model system

A Python-based face recognition system that identifies known individuals ("friends") and marks unknown individuals as "intruders" in real-time using webcam or video input.

## Features

- Real-time face detection and recognition
- Identifies known individuals by name
- Marks unknown individuals as "intruders"
- Supports both webcam and video file input
- Easy-to-use command line interface

## Requirements

- Python 3.6+
- OpenCV
- face_recognition library (which uses dlib)
- NumPy
- pickle

## Installation

1. Clone this repository:
```bash
git clone https://github.com/abanikanndatolu/Realtime-facial-recognition-ai-model-system.git
cd Realtime-facial-recognition-ai-model-system
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Note: The `face_recognition` library requires `dlib` which can be challenging to install on some systems. If you encounter issues, please refer to the [dlib installation guide](https://github.com/davisking/dlib).

## Usage

### Step 1: Prepare Your Dataset

Create a directory structure for your known individuals (friends):

```
dataset/
    person_name_1/
        image1.jpg
        image2.jpg
        ...
    person_name_2/
        image1.jpg
        image2.jpg
        ...
    ...
```

Each subdirectory should be named after the person and contain multiple clear images of their face from different angles and lighting conditions.

### Step 2: Generate Face Encodings

Run the face encoder script to process your dataset and generate encodings:

```bash
python face_encoder.py
```

This will create a `known_faces.pkl` file containing the facial encodings of your friends.

### Step 3: Run the Face Identifier

To use webcam input:

```bash
python face_identifier.py
```

To use a video file:

1. Open `face_identifier.py` in a text editor
2. Change the `INPUT_TYPE` variable to `'video'`
3. Update the `VIDEO_FILE_PATH` variable with the path to your video file
4. Run the script:
```bash
python face_identifier.py
```

## Configuration

You can adjust the following parameters in `face_identifier.py`:

- `INPUT_TYPE`: Set to `'webcam'` or `'video'`
- `VIDEO_FILE_PATH`: Path to your video file (if using video input)
- `ENCODINGS_PATH`: Path to the generated encodings file
- `RECOGNITION_TOLERANCE`: Recognition sensitivity (lower is stricter, 0.6 is default)

## How It Works

1. The `face_encoder.py` script:
   - Loads images from the dataset directory
   - Detects faces in each image
   - Extracts facial features (encodings)
   - Saves these encodings along with person names to a pickle file

2. The `face_identifier.py` script:
   - Loads the saved encodings
   - Captures frames from webcam or video file
   - Detects faces in each frame
   - Compares detected faces against known encodings
   - Labels each face as a known person or "Intruder"
   - Displays the result in real-time

## Controls

- Press 'q' to quit the application

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses the [face_recognition](https://github.com/ageitgey/face_recognition) library
- OpenCV for image processing and display

## Future Improvements

- Add recording capability for detected intruders
- Implement notification system
- Add a graphical user interface
- Improve performance on lower-end hardware
