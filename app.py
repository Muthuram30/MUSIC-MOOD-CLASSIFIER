import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import threading

import itertools
import pygame
import time

# ------------------------------
# Configuration
# ------------------------------
dataset_path = r"D:\My Files\Proj\Python\music_dataset"
classes = ['happy', 'sad', 'tender']
model_file = "model.pkl"

# ------------------------------
# Feature extraction
# ------------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features = np.concatenate((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(tonnetz.T, axis=0)
        ))
        return features.reshape(1, -1)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros((1, 38))

# ------------------------------
# Model training (optional)
# ------------------------------
def train_model():
    X, y_labels = [], []
    for label in classes:
        folder = os.path.join(dataset_path, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith((".mp3", ".wav", ".ogg")):
                try:
                    features = extract_features(os.path.join(folder, file))
                    X.append(features.flatten())
                    y_labels.append(label)
                except Exception as e:
                    print(f"Error: {e}")
    if not X:
        raise ValueError("No audio files found.")
    X = np.array(X)
    y_labels = np.array(y_labels)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_labels)
    joblib.dump(clf, model_file)
    return clf

# ------------------------------
# Main Application
# ------------------------------
class MusicClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AuraBeat - Music Mood Classifier")
        self.root.geometry("620x780")
        self.root.configure(bg="#121212")
        self.root.resizable(False, False)

        self.model = None
        self.current_file = None
        self.is_playing = False
        self.is_paused = False
        self.song_length = 0
        self.animating = True

        pygame.mixer.init()
        self.load_images()
        self.setup_ui()
        self.load_or_train_model()

    def load_images(self):
        try:
            self.window_icon = ImageTk.PhotoImage(Image.open("images/icon1.png"))
            gif_path = "images/animation.gif"
            gif = Image.open(gif_path)
            self.gif_frames = []
            for frame in range(gif.n_frames):
                gif.seek(frame)
                resized_frame = gif.resize((200, 200), Image.Resampling.LANCZOS)
                self.gif_frames.append(ImageTk.PhotoImage(resized_frame.convert("RGBA")))
            self.gif_cycle = itertools.cycle(self.gif_frames)
            self.current_frame = next(self.gif_cycle)
            self.animation_speed_ms = 50
        except Exception:
            pass

    def setup_ui(self):
        self.root.iconphoto(False, self.window_icon)
        frame = tk.Frame(self.root, bg="#121212")
        frame.pack(pady=20)

        self.animation_label = tk.Label(frame, bg="#121212")
        self.animation_label.pack(pady=10)
        self.animation_label.config(image=self.current_frame)
        self.animate()

        self.file_path_var = tk.StringVar(value="Select a song to begin")
        tk.Label(frame, textvariable=self.file_path_var, font=("Segoe UI", 10),
                 bg="#121212", fg="#b3b3b3", wraplength=500).pack(pady=8)

        self.lbl_result = tk.Label(frame, text="Predicted Mood:", font=("Segoe UI Semibold", 18),
                                   bg="#121212", fg="#1DB954")
        self.lbl_result.pack(pady=10)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("green.Horizontal.TProgressbar", background='#1DB954', troughcolor='#282828')

        self.progress_bars = {}
        self.percent_labels = {}

        for cls in classes:
            f = tk.Frame(frame, bg="#121212")
            f.pack(pady=4)
            tk.Label(f, text=cls.capitalize(), width=8, anchor='w', font=("Segoe UI", 12),
                     bg="#121212", fg="white").pack(side='left')
            pb = ttk.Progressbar(f, length=280, style="green.Horizontal.TProgressbar", maximum=100)
            pb.pack(side='left', padx=10)
            self.progress_bars[cls] = pb
            lbl = tk.Label(f, text="0%", font=("Segoe UI", 12), bg="#121212", fg="white")
            lbl.pack(side='left')
            self.percent_labels[cls] = lbl

        self.btn_select = tk.Button(
            frame, text="🎵 Choose Song", command=self.select_file,
            font=("Segoe UI", 12, "bold"), bg="#1DB954", fg="black",
            relief="flat", bd=0, padx=20, pady=10, activebackground="#17a64a"
        )
        self.btn_select.pack(pady=15)
        self.btn_select.config(state="disabled")

        self.btn_play = tk.Button(
            frame, text="▶ Play", command=self.toggle_play,
            font=("Segoe UI", 12, "bold"), bg="#1DB954", fg="black",
            relief="flat", bd=0, padx=25, pady=10, activebackground="#14863b"
        )
        self.btn_play.pack(pady=5)
        self.btn_play.config(state="disabled")

        # --- Seekable Progress Bar ---
        self.song_progress = ttk.Progressbar(frame, length=400, maximum=100, style="green.Horizontal.TProgressbar")
        self.song_progress.pack(pady=10)

        # Make the progress bar clickable
        self.song_progress.bind("<Button-1>", self.seek_song)

        # Time label
        self.lbl_time = tk.Label(frame, text="00:00 / 00:00", font=("Segoe UI", 10),
                                 bg="#121212", fg="#b3b3b3")
        self.lbl_time.pack(pady=5)

    def animate(self):
        if self.is_playing and not self.is_paused:
            self.current_frame = next(self.gif_cycle)
            self.animation_label.config(image=self.current_frame)
        self.root.after(self.animation_speed_ms, self.animate)

    def load_or_train_model(self):
        if os.path.exists(model_file):
            self.model = joblib.load(model_file)
            self.btn_select.config(state="normal")
        else:
            messagebox.showinfo("Model Training", "No model found. Training will start.")
            threading.Thread(target=self.train_thread, daemon=True).start()

    def train_thread(self):
        try:
            self.model = train_model()
            messagebox.showinfo("Training Complete", "Model trained successfully!")
            self.btn_select.config(state="normal")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.ogg")])
        if not file_path:
            return

        pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False
        self.song_progress['value'] = 0

        self.current_file = file_path
        self.file_path_var.set(os.path.basename(file_path))
        self.btn_play.config(state="normal")
        self.classify_music(file_path)

    def classify_music(self, file_path):
        try:
            features = extract_features(file_path)
            probs = self.model.predict_proba(features)[0]
            prob_dict = {cls: prob for cls, prob in zip(self.model.classes_, probs)}
            for cls in classes:
                val = prob_dict.get(cls, 0)
                self.progress_bars[cls]['value'] = val * 100
                self.percent_labels[cls]['text'] = f"{int(val * 100)}%"
            predicted = max(prob_dict, key=prob_dict.get)
            self.lbl_result.config(text=f"Predicted Mood: {predicted.capitalize()}")
        except Exception as e:
            messagebox.showerror("Error", f"Error classifying file:\n{e}")

    def toggle_play(self):
        if not self.current_file:
            messagebox.showwarning("No Song", "Please select a song first.")
            return

        if not self.is_playing:
            try:
                pygame.mixer.music.load(self.current_file)
                pygame.mixer.music.play()
                self.song_length = librosa.get_duration(filename=self.current_file)
                self.is_playing = True
                self.is_paused = False
                self.btn_play.config(bg="#14863b", text="⏸ Pause")
                threading.Thread(target=self.update_progress, daemon=True).start()
            except Exception as e:
                messagebox.showerror("Playback Error", str(e))
        else:
            if self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
                self.btn_play.config(bg="#14863b", text="⏸ Pause")
            else:
                pygame.mixer.music.pause()
                self.is_paused = True
                self.btn_play.config(bg="#17a64a", text="▶ Resume")

    def seek_song(self, event):
        """Jump to a position in the song when clicking the progress bar."""
        if not self.is_playing or not self.current_file or self.song_length == 0:
            return

        progress_bar_width = self.song_progress.winfo_width()
        click_x = event.x
        click_percentage = click_x / progress_bar_width
        seek_time = self.song_length * click_percentage

        pygame.mixer.music.stop()
        pygame.mixer.music.load(self.current_file)
        pygame.mixer.music.play(start=seek_time)
        self.btn_play.config(bg="#14863b", text="⏸ Pause")
        self.is_paused = False

    def update_progress(self):
        while self.is_playing:
            if not self.is_paused:
                pos = pygame.mixer.music.get_pos() / 1000
                if self.song_length > 0:
                    progress = min((pos / self.song_length) * 100, 100)
                    self.song_progress['value'] = progress
                    self.lbl_time.config(text=f"{int(pos//60):02d}:{int(pos%60):02d} / {int(self.song_length//60):02d}:{int(self.song_length%60):02d}")
                if not pygame.mixer.music.get_busy():
                    self.is_playing = False
                    self.btn_play.config(bg="#1DB954", text="▶ Play")
                    self.song_progress['value'] = 0
                    self.lbl_time.config(text="00:00 / 00:00")
                    break
            time.sleep(0.5)

# ------------------------------
# Run app
# ------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MusicClassifierApp(root)
    root.mainloop()
