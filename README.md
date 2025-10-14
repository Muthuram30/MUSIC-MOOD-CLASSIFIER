# MUSIC-MOOD-CLASSIFIER
Music Mood Classifier – Classifies songs by mood using audio features and a Random Forest model, with a simple GUI for playback and predictions.
# AuraBeat - Music Mood Classifier

AuraBeat is a Python GUI application that classifies the mood of a song into **Happy**, **Sad**, or **Tender** using machine learning. The app also allows you to play music with an interactive progress bar and animations.

## Features
- **Mood Classification:** Predicts song mood with a Random Forest model.
- **Audio Playback:** Play, pause, resume, and seek within songs.
- **Interactive Progress Bar:** Click to jump to any part of the song.
- **Visual Feedback:** GIF animation runs during playback and pauses when paused.
- **Probability Display:** Shows confidence for each mood category.

## How to Use
1. Launch the app:  
   ```bash
   python app.py
##Folder Structure
/project_folder/
├── app.py
├── model.py
├── model.pkl
├── music_dataset/
│   ├── happy/
│   ├── sad/
│   └── tender/
└── images/
    ├── icon1.png
    ├── icon2.png
    └── animation.gif
