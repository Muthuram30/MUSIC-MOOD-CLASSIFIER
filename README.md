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
First look the folder structure and create a music_dataset forlder with subfolders namely happy,sad,tender.<br>
1. Launch the app:  
   ```bash
   python app.py
##Folder Structure
/project_folder/<br>
├── app.py<br>
├── model.py<br>
├── model.pkl<br>
├── music_dataset/<br>
│   ├── happy/<br>
│   ├── sad/<br>
│   └── tender/<br>
└── images/<br>
    ├── icon1.png<br>
    ├── icon2.png<br>
    └── animation.gif<br>
