"""
The McGurk Effect Demonstrator

This Python script demonstrates the McGurk Effect, an auditory illusion where visual and auditory signals combine to create a perceived sound that differs from the actual auditory input. The script enables users to record themselves pronouncing syllables "ba" and "ga," then processes these recordings to produce a video that demonstrates the McGurk effect using Dynamic Time Warping (DTW) for audio alignment and synchronization.

Key functionalities include:
- Video capture via webcam for real-time recording.
- Audio recording through the system's microphone.
- Audio processing with DTW to align 'ba' and 'ga' sounds to demonstrate the effect.
- Integration of audio with video to create the final demonstration output.

This script is designed to be both a practical tool for experiencing the McGurk effect firsthand and an educational resource for studying audio-visual perception in cognitive science.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import PIL.Image, PIL.ImageTk
import sounddevice as sd
import numpy as np
import wave
import librosa
import soundfile as sf
from tslearn.metrics import dtw_path
from moviepy.editor import VideoFileClip, AudioFileClip
import subprocess
import matplotlib.pyplot as plt

# Global flags and variables:
# 'recording' indicates whether recording is active.
# 'audio_data' stores the audio samples recorded from the microphone.
recording = False
audio_data = []

def start_recording_window(syllable, video_filename, audio_filename):
    def update_video_feed():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return

        def show_frame():
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to grab frame.")
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            video_label.after(10, show_frame)

        show_frame()
        return cap  # Initialize the video capture and return the object to manage the video stream

    def start_audio_recording():
        global audio_data
        fs = 48000  # Increased sample rate for better quality
        audio_data = []  # Clear previous recordings
        recording_event = threading.Event()

        def callback(indata, frames, time, status):
            audio_data.append(indata.copy())

        # Start capturing audio in a non-blocking mode using a callback to handle incoming audio frames
        stream = sd.InputStream(samplerate=fs, channels=2, callback=callback, dtype='int16')
        stream.start()

        return stream, recording_event

    def stop_audio_recording(stream, audio_filename):
        stream.stop()
        # Save recorded audio to a WAV file
        wf = wave.open(audio_filename, 'wb')
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(b''.join(np.concatenate(audio_data)))
        wf.close()

    def start_recording(cap, audio_filename):
        global recording
        recording = True

        # Start audio recording
        audio_stream, recording_event = start_audio_recording()

        threading.Thread(target=record, args=(cap, audio_stream, recording_event, video_filename, audio_filename)).start()

    def record(cap, audio_stream, recording_event, video_filename, audio_filename):
        global recording
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480), isColor=True)
        while recording:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        # Release resources: stop video capture, close file stream, and stop audio recording
        stop_audio_recording(audio_stream, audio_filename)

    def stop_recording():
        global recording
        recording = False
        recording_window.destroy()
        messagebox.showinfo("Info", f"{syllable.capitalize()} recording saved.")

    # Create a new window for recording
    recording_window = tk.Toplevel()
    recording_window.title(f"Record '{syllable}'")
    recording_window.geometry("800x800")

    # Instruction label
    instruction_label = ttk.Label(recording_window, text=f"Please say '{syllable}' 5 times, 2 seconds apart.", font=("Arial", 14))
    instruction_label.pack(pady=20)

    # Video feed label
    video_label = ttk.Label(recording_window)
    video_label.pack()

    # Start and Stop buttons
    button_frame = ttk.Frame(recording_window)
    button_frame.pack(pady=20)

    start_button = ttk.Button(button_frame, text="Start Recording", command=lambda: start_recording(cap, f"{syllable}.wav"))
    start_button.pack(side=tk.LEFT, padx=10)

    stop_button = ttk.Button(button_frame, text="Stop Recording", command=stop_recording)
    stop_button.pack(side=tk.LEFT, padx=10)

    cap = update_video_feed()

    recording_window.protocol("WM_DELETE_WINDOW", stop_recording)
    recording_window.mainloop()

def align_audio_files_segmented(ba_file, ga_file, output_file, segment_size=5000, visualize=False):
    try:
        # Load audio files and ensure they are not empty or filled with NaNs
        ba_audio, sr_ba = librosa.load(ba_file, sr=None)
        if ba_audio.size == 0 or np.isnan(ba_audio).all():
            raise ValueError("Loaded 'ba' audio is empty or NaN.")

        ga_audio, sr_ga = librosa.load(ga_file, sr=None)
        if ga_audio.size == 0 or np.isnan(ga_audio).all():
            raise ValueError("Loaded 'ga' audio is empty or NaN.")

        # Resample if necessary
        if sr_ba != sr_ga:
            ba_audio = librosa.resample(ba_audio, orig_sr=sr_ba, target_sr=sr_ga)

        # Ensure mono audio
        if ba_audio.ndim > 1:
            ba_audio = ba_audio[:, 0]
        if ga_audio.ndim > 1:
            ga_audio = ga_audio[:, 0]

        aligned_audio = []
        total_segments = (len(ba_audio) // segment_size) + (1 if len(ba_audio) % segment_size != 0 else 0)

        for i in range(total_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            if end_idx > len(ba_audio):
                end_idx = len(ba_audio)

            ba_segment = ba_audio[start_idx:end_idx]
            ga_segment = ga_audio[start_idx:end_idx]

            if ba_segment.size == 0 or ga_segment.size == 0:
                continue  # Skip empty segments

            # Calculate DTW path
            path, sim_cost = dtw_path(ba_segment.reshape(-1, 1), ga_segment.reshape(-1, 1))

            aligned_segment = np.zeros_like(ga_segment)
            for (ba_idx, ga_idx) in path:
                aligned_segment[ga_idx] = ba_segment[ba_idx] if ba_idx < len(ba_segment) else 0

            aligned_audio.extend(aligned_segment)

            if visualize:
                plt.figure()
                plt.plot(ba_segment, label='Original Ba', color='blue')
                plt.plot(ga_segment, label='Ga', color='red')
                plt.plot(aligned_segment, label='Aligned Ba', linestyle='--', color='green')
                plt.legend()
                plt.title(f'Segment {i + 1} Alignment')
                plt.show()

        # Save the aligned audio
        sf.write(output_file, np.array(aligned_audio), sr_ga)

    except Exception as e:
        print(f"Failed to align audio: {e}")
        raise

def combine_video_and_audio(video_file, audio_file, output_file):
    try:
        video_clip = VideoFileClip(video_file)
        audio_clip = AudioFileClip(audio_file)

        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac', audio_bitrate='320k')

        messagebox.showinfo("Success", "McGurk Effect video created and saved as " + output_file)
        subprocess.Popen(['start', output_file], shell=True)
    except Exception as e:
        messagebox.showerror("Error", "Failed to combine video and audio: " + str(e))

def create_mcgurk_effect():
    align_audio_files_segmented('ba.wav', 'ga.wav', 'aligned_ba.wav', visualize=True)
    combine_video_and_audio('ga.avi', 'aligned_ba.wav', 'mcgurk_effect_final.mp4')

# Main window setup
root = tk.Tk()
root.title("McGurk Effect Demonstrator")
root.geometry("1300x600")
root.configure(bg="#F0F0F0")

intro_text = """
The McGurk Effect is an auditory illusion where what you see influences what you hear.
For example, seeing someone mouth "ga" while hearing "ba" can lead you to perceive "da".
This demonstrator allows you to explore this effect by recording yourself saying the syllables "ba" and "ga",
then combining the videos to create the McGurk effect.
"""
intro_label = ttk.Label(root, text=intro_text, font=("Arial", 16, "bold"), foreground="#333333", background="#F0F0F0")
intro_label.place(relwidth=0.8, relheight=0.3, relx=0.1, rely=0.05)

record_ba_button = ttk.Button(root, text="Record 'Ba'", command=lambda: start_recording_window("ba", "ba.avi", "ba.wav"))
record_ba_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

record_ga_button = ttk.Button(root, text="Record 'Ga'", command=lambda: start_recording_window("ga", "ga.avi", "ga.wav"))
record_ga_button.place(relx=0.1, rely=0.55, relwidth=0.3, relheight=0.1)

create_mcgurk_button = ttk.Button(root, text="Create McGurk Effect", command=create_mcgurk_effect)
create_mcgurk_button.place(relx=0.1, rely=0.7, relwidth=0.3, relheight=0.1)

root.mainloop()