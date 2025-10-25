#!/usr/bin/env python3
"""
EXPRESSIVE GESTURE PIANO 

Features:
- Keyboard selects notes (2 octaves)
- Hand Y-position controls duration (0.1s - 3.0s)
- Hand speed controls volume (dynamic expression)
- Hand X-position controls stereo pan
- Hand openness toggles sustain
- Smooth gesture processing with moving averages
- Proper note stopping when duration changes
- Modular architecture for easy extension

Architecture:
- KeyboardInput: Handles key detection
- GestureProcessor: Tracks and interprets hand gestures
- AudioEngine: Manages sound generation and playback
- Visualizer: Real-time visual feedback
- Main Controller: Coordinates all modules

Controls:
- A-Z, etc. = Play notes
- Hand UP = Longer notes
- Hand DOWN = Shorter notes
- Move hand FAST = Louder
- Move hand X = Pan left/right
- OPEN hand = Sustain ON
- CLOSE hand = Sustain OFF
- SPACE = Toggle tracking
- R = Reset
- ESC = Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum


# ==================== DATA STRUCTURES ====================

@dataclass
class GestureState:
    """Current state of hand gestures."""
    hand_y: Optional[float] = None  # Vertical position (0-1)
    hand_x: Optional[float] = None  # Horizontal position (0-1)
    hand_speed: float = 0.0  # Movement speed
    hand_openness: float = 0.0  # 0=closed, 1=open
    is_detected: bool = False

    # Computed values
    duration: float = 0.5  # Note duration in seconds
    volume: float = 0.7  # Volume 0-1
    pan: float = 0.5  # Stereo pan 0=left, 0.5=center, 1=right
    sustain: bool = False  # Sustain pedal effect


@dataclass
class NoteEvent:
    """Represents a note to be played."""
    note_name: str
    frequency: float
    duration: float
    volume: float
    pan: float
    sustain: bool
    timestamp: float


class NoteState(Enum):
    """State of a playing note."""
    PLAYING = 1
    STOPPING = 2
    STOPPED = 3


# ==================== KEYBOARD INPUT MODULE ====================

class KeyboardInput:
    """Handles keyboard input for note selection."""

    # Complete 2-octave mapping
    NOTE_MAP = {
        # First octave (C4)
        'a': ('C4', 261.63, False),
        's': ('D4', 293.66, False),
        'd': ('E4', 329.63, False),
        'f': ('F4', 349.23, False),
        'g': ('G4', 392.00, False),
        'h': ('A4', 440.00, False),
        'j': ('B4', 493.88, False),

        # Black keys (first octave)
        'q': ('C#4', 277.18, True),
        'w': ('D#4', 311.13, True),
        't': ('F#4', 369.99, True),
        'y': ('G#4', 415.30, True),
        'u': ('A#4', 466.16, True),

        # Second octave (C5)
        'k': ('C5', 523.25, False),
        'l': ('D5', 587.33, False),
        ';': ('E5', 659.25, False),
        "'": ('F5', 698.46, False),
        'z': ('G5', 783.99, False),
        'x': ('A5', 880.00, False),
        'c': ('B5', 987.77, False),

        # Black keys (second octave)
        ']': ('C#5', 554.37, True),
        '[': ('D#5', 622.25, True),
        '-': ('F#5', 739.99, True),
        '=': ('G#5', 830.61, True),
    }

    @staticmethod
    def get_note_info(key: str) -> Optional[Tuple[str, float, bool]]:
        """Get note information for a keyboard key."""
        return KeyboardInput.NOTE_MAP.get(key)


# ==================== GESTURE PROCESSOR MODULE ====================

class GestureProcessor:
    """Processes hand gestures and computes control parameters."""

    def __init__(self, smoothing_window: int = 10):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Smoothing buffers
        self.y_history = deque(maxlen=smoothing_window)
        self.x_history = deque(maxlen=smoothing_window)
        self.speed_history = deque(maxlen=5)

        # Previous position for speed calculation
        self.prev_position = None
        self.prev_time = None

        # Duration mapping
        self.min_duration = 0.1
        self.max_duration = 3.0

        # Volume mapping
        self.min_volume = 0.3
        self.max_volume = 1.0

    def process_frame(self, frame: np.ndarray) -> GestureState:
        """Process a camera frame and return gesture state."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        state = GestureState()

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            state.is_detected = True

            # Get palm center (landmark 0)
            palm = hand_landmarks.landmark[0]
            raw_y = palm.y
            raw_x = palm.x

            # Smooth Y position
            self.y_history.append(raw_y)
            state.hand_y = np.mean(list(self.y_history))

            # Smooth X position
            self.x_history.append(raw_x)
            state.hand_x = np.mean(list(self.x_history))

            # Calculate speed
            current_time = time.time()
            if self.prev_position is not None and self.prev_time is not None:
                dt = current_time - self.prev_time
                if dt > 0:
                    dx = raw_x - self.prev_position[0]
                    dy = raw_y - self.prev_position[1]
                    speed = np.sqrt(dx ** 2 + dy ** 2) / dt
                    self.speed_history.append(speed)
                    state.hand_speed = np.mean(list(self.speed_history))

            self.prev_position = (raw_x, raw_y)
            self.prev_time = current_time

            # Calculate hand openness (distance between thumb and pinky)
            thumb_tip = hand_landmarks.landmark[4]
            pinky_tip = hand_landmarks.landmark[20]
            distance = np.sqrt(
                (thumb_tip.x - pinky_tip.x) ** 2 +
                (thumb_tip.y - pinky_tip.y) ** 2
            )
            state.hand_openness = np.clip(distance * 5, 0, 1)  # Scale and clip

            # Compute control parameters
            state.duration = self._compute_duration(state.hand_y)
            state.volume = self._compute_volume(state.hand_speed)
            state.pan = state.hand_x  # Direct mapping
            state.sustain = state.hand_openness > 0.6  # Threshold for sustain

            return state, hand_landmarks

        return state, None

    def _compute_duration(self, hand_y: float) -> float:
        """Compute note duration from hand Y position."""
        if hand_y is None:
            return 0.5
        # Invert: top (0) = max duration, bottom (1) = min duration
        normalized = 1.0 - hand_y
        return self.min_duration + normalized * (self.max_duration - self.min_duration)

    def _compute_volume(self, speed: float) -> float:
        """Compute volume from hand speed."""
        # Map speed to volume (faster = louder)
        # Speed is typically 0-5 for normal hand movements
        normalized = np.clip(speed / 3.0, 0, 1)
        return self.min_volume + normalized * (self.max_volume - self.min_volume)

    def reset(self):
        """Reset all buffers."""
        self.y_history.clear()
        self.x_history.clear()
        self.speed_history.clear()
        self.prev_position = None
        self.prev_time = None

    def close(self):
        """Clean up resources."""
        self.hands.close()


# ==================== AUDIO ENGINE MODULE ====================

class AudioEngine:
    """Manages sound generation and playback with proper duration control."""

    def __init__(self, sample_rate: int = 22050):
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=512)
        self.sample_rate = sample_rate

        # Active notes: key -> (channel, stop_time, note_event)
        self.active_notes: Dict[str, Tuple[pygame.mixer.Channel, float, NoteEvent]] = {}
        self.note_lock = threading.Lock()

        # Start background thread to stop notes at the right time
        self.running = True
        self.stop_thread = threading.Thread(target=self._note_stopper_thread, daemon=True)
        self.stop_thread.start()

    def generate_note_sound(self, frequency: float, duration: float,
                            volume: float = 0.7, pan: float = 0.5) -> pygame.mixer.Sound:
        """Generate a piano-like sound with proper duration."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # Generate wave with harmonics
        wave = np.sin(2 * np.pi * frequency * t)
        wave += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
        wave += 0.15 * np.sin(2 * np.pi * frequency * 3 * t)
        wave += 0.08 * np.sin(2 * np.pi * frequency * 4 * t)

        # ADSR envelope
        attack_samples = int(0.01 * self.sample_rate)
        decay_samples = int(0.1 * self.sample_rate)
        sustain_level = 0.7
        release_samples = int(0.15 * self.sample_rate)

        total_samples = len(t)
        envelope = np.ones(total_samples)

        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay
        decay_end = min(attack_samples + decay_samples, total_samples)
        if decay_end > attack_samples:
            envelope[attack_samples:decay_end] = np.linspace(1, sustain_level, decay_end - attack_samples)

        # Sustain
        sustain_start = decay_end
        sustain_end = max(sustain_start, total_samples - release_samples)
        if sustain_end > sustain_start:
            envelope[sustain_start:sustain_end] = sustain_level

        # Release
        if release_samples > 0 and sustain_end < total_samples:
            envelope[sustain_end:] = np.linspace(sustain_level, 0, total_samples - sustain_end)

        wave *= envelope

        # Normalize and apply volume
        wave = wave / np.max(np.abs(wave)) * volume
        wave = (wave * 32767).astype(np.int16)

        # Apply stereo panning
        left_volume = 1.0 - pan
        right_volume = pan

        left_channel = (wave * left_volume).astype(np.int16)
        right_channel = (wave * right_volume).astype(np.int16)

        stereo_wave = np.column_stack((left_channel, right_channel))

        return pygame.sndarray.make_sound(stereo_wave)

    def play_note(self, note_event: NoteEvent, keyboard_key: str):
        """Play a note with the given parameters."""
        with self.note_lock:
            # Stop any existing note on this key
            if keyboard_key in self.active_notes:
                channel, _, _ = self.active_notes[keyboard_key]
                channel.stop()
                del self.active_notes[keyboard_key]

            # Generate and play new sound
            sound = self.generate_note_sound(
                note_event.frequency,
                note_event.duration,
                note_event.volume,
                note_event.pan
            )

            channel = sound.play()
            if channel:
                stop_time = time.time() + note_event.duration
                self.active_notes[keyboard_key] = (channel, stop_time, note_event)

    def _note_stopper_thread(self):
        """Background thread that stops notes at the right time."""
        while self.running:
            current_time = time.time()
            with self.note_lock:
                keys_to_remove = []
                for key, (channel, stop_time, _) in list(self.active_notes.items()):
                    if current_time >= stop_time:
                        channel.fadeout(50)  # 50ms fadeout
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.active_notes[key]

            time.sleep(0.01)  # Check every 10ms

    def stop_all(self):
        """Stop all playing notes."""
        with self.note_lock:
            for channel, _, _ in self.active_notes.values():
                channel.stop()
            self.active_notes.clear()

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        self.stop_thread.join(timeout=1)
        self.stop_all()


# ==================== VISUALIZER MODULE ====================

class Visualizer:
    """Handles visual feedback and UI rendering."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # Active keys for visual feedback
        self.active_keys: Dict[str, Tuple[str, float]] = {}
        self.key_display_duration = 0.3

    def draw_frame(self, frame: np.ndarray, gesture_state: GestureState,
                   hand_landmarks, fps: int, last_note: Optional[str] = None):
        """Draw complete UI on frame."""
        # Info panel
        self._draw_info_panel(frame, gesture_state, fps)

        # Gesture meters
        self._draw_gesture_meters(frame, gesture_state)

        # Keyboard layout
        self._draw_keyboard_layout(frame)

        # Hand tracking
        if hand_landmarks:
            self._draw_hand_tracking(frame, hand_landmarks, gesture_state)

        # Last note indicator
        if last_note:
            self._draw_last_note(frame, last_note)

        return frame

    def _draw_info_panel(self, frame, gesture_state, fps):
        """Draw information panel."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        y = 35
        cv2.putText(frame, "EXPRESSIVE GESTURE PIANO", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y += 30
        status = "ACTIVE" if gesture_state.is_detected else "NO HAND"
        color = (0, 255, 0) if gesture_state.is_detected else (0, 0, 255)
        cv2.putText(frame, f"Status: {status} | FPS: {fps}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if gesture_state.is_detected:
            y += 25
            cv2.putText(frame, f"Duration: {gesture_state.duration:.2f}s | "
                               f"Volume: {gesture_state.volume:.2f} | "
                               f"Pan: {gesture_state.pan:.2f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            y += 25
            sustain_text = "ON" if gesture_state.sustain else "OFF"
            sustain_color = (0, 255, 0) if gesture_state.sustain else (100, 100, 100)
            cv2.putText(frame, f"Sustain: {sustain_text} | "
                               f"Speed: {gesture_state.hand_speed:.2f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, sustain_color, 1)

        y += 30
        cv2.putText(frame, "SPACE=Toggle | R=Reset | Q/ESC=Quit", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _draw_gesture_meters(self, frame, gesture_state):
        """Draw gesture control meters."""
        meter_x = self.width - 250
        meter_y = 50

        # Duration meter
        self._draw_meter(frame, meter_x, meter_y, 60, 300,
                         "DURATION", gesture_state.duration, 0.1, 3.0,
                         [(255, 100, 0), (0, 255, 255), (0, 255, 100)])

        # Volume meter
        self._draw_meter(frame, meter_x + 80, meter_y, 60, 300,
                         "VOLUME", gesture_state.volume, 0.3, 1.0,
                         [(100, 100, 255), (0, 255, 255), (100, 255, 100)])

        # Pan meter (horizontal)
        self._draw_pan_meter(frame, meter_x, meter_y + 320, 180, 40,
                             gesture_state.pan)

    def _draw_meter(self, frame, x, y, width, height, label, value,
                    min_val, max_val, colors):
        """Draw a vertical meter."""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)

        # Label
        cv2.putText(frame, label, (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Fill
        if value is not None:
            fill_ratio = (value - min_val) / (max_val - min_val)
            fill_ratio = np.clip(fill_ratio, 0, 1)
            fill_height = int(height * fill_ratio)
            fill_y = y + height - fill_height

            # Color gradient
            if fill_ratio < 0.33:
                color = colors[0]
            elif fill_ratio < 0.66:
                color = colors[1]
            else:
                color = colors[2]

            cv2.rectangle(frame, (x + 3, fill_y),
                          (x + width - 3, y + height - 3), color, -1)

            # Value text
            cv2.putText(frame, f"{value:.2f}", (x + 5, fill_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_pan_meter(self, frame, x, y, width, height, pan):
        """Draw horizontal pan meter."""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)

        # Labels
        cv2.putText(frame, "L", (x - 15, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "PAN", (x + width // 2 - 15, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "R", (x + width + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Center line
        center_x = x + width // 2
        cv2.line(frame, (center_x, y), (center_x, y + height),
                 (100, 100, 100), 1)

        # Pan indicator
        if pan is not None:
            indicator_x = int(x + pan * width)
            cv2.rectangle(frame, (indicator_x - 3, y + 3),
                          (indicator_x + 3, y + height - 3), (0, 255, 255), -1)

    def _draw_keyboard_layout(self, frame):
        """Draw keyboard layout at bottom."""
        layout_y = self.height - 180
        white_key_width = 45
        white_key_height = 130
        black_key_width = 28
        black_key_height = 80
        start_x = 30

        # White keys
        white_keys = [
            ('A', 'C4'), ('S', 'D4'), ('D', 'E4'), ('F', 'F4'),
            ('G', 'G4'), ('H', 'A4'), ('J', 'B4'), ('K', 'C5'),
            ('L', 'D5'), (';', 'E5'), ("'", 'F5'), ('Z', 'G5'),
            ('X', 'A5'), ('C', 'B5')
        ]

        for i, (key, note) in enumerate(white_keys):
            x = start_x + i * white_key_width

            # Check if active
            is_active = key.lower() in self.active_keys and \
                        time.time() - self.active_keys[key.lower()][1] < self.key_display_duration

            color = (100, 255, 100) if is_active else (240, 240, 240)

            cv2.rectangle(frame, (x, layout_y),
                          (x + white_key_width - 2, layout_y + white_key_height),
                          color, -1)
            cv2.rectangle(frame, (x, layout_y),
                          (x + white_key_width - 2, layout_y + white_key_height),
                          (0, 0, 0), 2)

            # Labels
            cv2.putText(frame, key, (x + 12, layout_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, note, (x + 8, layout_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        # Black keys
        black_keys = [
            ('Q', 'C#4', 0), ('W', 'D#4', 1), ('T', 'F#4', 3),
            ('Y', 'G#4', 4), ('U', 'A#4', 5), (']', 'C#5', 7),
            ('[', 'D#5', 8), ('-', 'F#5', 10), ('=', 'G#5', 11)
        ]

        for key, note, pos in black_keys:
            x = start_x + pos * white_key_width + white_key_width - black_key_width // 2

            is_active = key.lower() in self.active_keys and \
                        time.time() - self.active_keys[key.lower()][1] < self.key_display_duration

            color = (100, 255, 255) if is_active else (40, 40, 40)

            cv2.rectangle(frame, (x, layout_y),
                          (x + black_key_width, layout_y + black_key_height),
                          color, -1)
            cv2.rectangle(frame, (x, layout_y),
                          (x + black_key_width, layout_y + black_key_height),
                          (0, 0, 0), 2)

            text_color = (0, 0, 0) if is_active else (200, 200, 200)
            cv2.putText(frame, key, (x + 6, layout_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            cv2.putText(frame, note, (x + 3, layout_y + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)

    def _draw_hand_tracking(self, frame, hand_landmarks, gesture_state):
        """Draw hand tracking visualization."""
        # Draw skeleton
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

        # Palm center
        palm = hand_landmarks.landmark[0]
        palm_x = int(palm.x * self.width)
        palm_y = int(palm.y * self.height)

        cv2.circle(frame, (palm_x, palm_y), 12, (0, 255, 255), -1)
        cv2.circle(frame, (palm_x, palm_y), 12, (255, 255, 255), 2)

        # Direction indicators
        cv2.arrowedLine(frame, (palm_x, palm_y), (palm_x, palm_y - 35),
                        (255, 255, 0), 2, tipLength=0.3)
        cv2.putText(frame, "UP=Long", (palm_x + 15, palm_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.arrowedLine(frame, (palm_x, palm_y), (palm_x, palm_y + 35),
                        (255, 0, 255), 2, tipLength=0.3)
        cv2.putText(frame, "DN=Short", (palm_x + 15, palm_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Speed indicator
        if gesture_state.hand_speed > 1.0:
            cv2.circle(frame, (palm_x, palm_y), 20, (255, 0, 0), 2)
            cv2.putText(frame, "FAST!", (palm_x - 20, palm_y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def _draw_last_note(self, frame, note_name):
        """Draw last played note."""
        cv2.putText(frame, f"Last: {note_name}", (self.width - 450, self.height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def mark_key_active(self, key: str, note_name: str):
        """Mark a key as recently pressed."""
        self.active_keys[key] = (note_name, time.time())


# ==================== MAIN CONTROLLER ====================

class ExpressivePiano:
    """Main controller that coordinates all modules."""

    def __init__(self):
        # Initialize modules
        self.gesture_processor = GestureProcessor(smoothing_window=10)
        self.audio_engine = AudioEngine()
        self.keyboard_input = KeyboardInput()

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.visualizer = Visualizer(width, height)

        # State
        self.tracking_enabled = True
        self.last_note_played = None

        self.print_welcome()

    def print_welcome(self):
        """Print welcome message."""
        print("\n" + "=" * 70)
        print("EXPRESSIVE GESTURE PIANO - Complete System")
        print("=" * 70)
        print("\nGesture Controls:")
        print("  • Hand Y-position  → Duration (up=long, down=short)")
        print("  • Hand Speed       → Volume (fast=loud, slow=soft)")
        print("  • Hand X-position  → Stereo Pan (left/center/right)")
        print("  • Hand Openness    → Sustain (open=ON, closed=OFF)")
        print("\nKeyboard Controls:")
        print("  • A-Z, ;, ', -, =, [, ]  → Play notes (2 octaves)")
        print("  • SPACE  → Toggle hand tracking")
        print("  • R      → Reset tracking")
        print("  • ESC    → Quit")
        print("=" * 70 + "\n")

    def run(self):
        """Main application loop."""
        fps_time = time.time()
        fps_counter = 0
        fps_display = 0

        gesture_state = GestureState()

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            # Process gestures
            if self.tracking_enabled:
                gesture_state, hand_landmarks = self.gesture_processor.process_frame(frame)
            else:
                hand_landmarks = None
                gesture_state.is_detected = False

            # Render visualization
            frame = self.visualizer.draw_frame(
                frame, gesture_state, hand_landmarks,
                fps_display, self.last_note_played
            )

            # FPS calculation
            fps_counter += 1
            if time.time() - fps_time > 1:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()

            # Display
            cv2.imshow('Expressive Gesture Piano', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC only (q is a musical note!)
                break
            elif key == ord(' '):  # Space - toggle tracking
                self.tracking_enabled = not self.tracking_enabled
                print(f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}")
            elif key == ord('r'):  # Reset
                self.gesture_processor.reset()
                print("Tracking reset")
            elif key != 255:  # Some key pressed
                char_key = chr(key)
                note_info = self.keyboard_input.get_note_info(char_key)

                if note_info:
                    note_name, frequency, is_black = note_info

                    # Create note event with current gesture parameters
                    note_event = NoteEvent(
                        note_name=note_name,
                        frequency=frequency,
                        duration=gesture_state.duration,
                        volume=gesture_state.volume,
                        pan=gesture_state.pan,
                        sustain=gesture_state.sustain,
                        timestamp=time.time()
                    )

                    # Play note
                    self.audio_engine.play_note(note_event, char_key)

                    # Update visuals
                    self.visualizer.mark_key_active(char_key, note_name)
                    self.last_note_played = note_name

        self.cleanup()

    def cleanup(self):
        """Clean up all resources."""
        print("\nShutting down...")
        self.audio_engine.cleanup()
        self.gesture_processor.close()
        self.cap.release()
        cv2.destroyAllWindows()


# ==================== ENTRY POINT ====================

def main():
    """Entry point."""
    try:
        piano = ExpressivePiano()
        piano.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
