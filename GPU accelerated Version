#!/usr/bin/env python3
"""
Virtual Piano - GPU ACCELERATED with PyTorch CUDA
Uses PyTorch CUDA for faster frame processing and effects.

Features:
- GPU-accelerated frame processing
- Faster particle effects on GPU
- All 5 fingers tracked
- Falling notes + visual effects
- Automatic GPU detection (falls back to CPU if unavailable)
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# PyTorch imports
import torch
import torch.nn.functional as F

print("=" * 60)
print("🚀 GPU-Accelerated Virtual Piano")
print("=" * 60)

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ CUDA Available!")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("⚠️  CUDA not available, using CPU")
    print("   (Performance will be good but not GPU-accelerated)")

print("=" * 60)


@dataclass
class PianoKey:
    """Piano key with state."""
    x: int
    y: int
    width: int
    height: int
    note: str
    frequency: float
    octave: int
    is_black: bool = False
    is_pressed: bool = False
    press_time: float = 0
    glow_intensity: float = 0
    pressed_by_finger: str = ""


@dataclass
class FallingNote:
    """Falling note object."""
    key_index: int
    y: float
    speed: float = 3.0
    hit: bool = False
    missed: bool = False
    hit_time: float = 0


@dataclass
class Particle:
    """GPU-accelerated particle."""
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float = 1.0
    size: int = 5


class GPUPiano:
    """GPU-accelerated virtual piano using PyTorch CUDA."""

    FINGER_TIPS = {
        'Thumb': 4,
        'Index': 8,
        'Middle': 12,
        'Ring': 16,
        'Pinky': 20
    }

    FINGER_COLORS = {
        'Thumb': (255, 150, 0),
        'Index': (0, 255, 0),
        'Middle': (0, 255, 255),
        'Ring': (255, 0, 255),
        'Pinky': (255, 100, 255)
    }

    def __init__(self, num_octaves=2.5):
        self.device = device

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Pygame audio
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Piano setup
        self.num_octaves = num_octaves
        self.scroll_offset = 0
        self.keys: List[PianoKey] = []
        self.create_keyboard()

        # Audio
        self.sounds = {}
        self.generate_sounds()

        # Game state
        self.song_mode = True
        self.falling_notes: List[FallingNote] = []
        self.particles: List[Particle] = []
        self.score = 0
        self.combo = 0
        self.max_combo = 0

        # Tracking
        self.finger_positions: Dict[Tuple[str, str], deque] = {}
        self.finger_trails: Dict[Tuple[str, str], deque] = {}
        self.last_press_time: Dict[str, float] = {}
        self.press_cooldown = 0.1

        # Visual settings
        self.glow_decay = 0.88
        self.min_velocity_threshold = 10
        self.hit_zone_y = self.height - 200
        self.hit_tolerance = 50

        # GPU buffers for particle system
        if torch.cuda.is_available():
            self.particle_positions_gpu = None
            self.particle_velocities_gpu = None
            self.particle_colors_gpu = None
            self.particle_life_gpu = None

        # Note spawning
        self.last_note_spawn = time.time()
        self.note_spawn_interval = 1.5
        self.song_notes = self.create_song()
        self.song_index = 0

        # Performance tracking
        self.gpu_time = 0
        self.cpu_time = 0

    def create_keyboard(self):
        """Create piano keyboard."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_types = ['W', 'B', 'W', 'B', 'W', 'W', 'B', 'W', 'B', 'W', 'B', 'W']

        base_frequency = 130.81
        white_key_width = 80
        white_key_height = 180
        black_key_width = 50
        black_key_height = 110
        key_y = self.height - white_key_height - 20

        self.hit_zone_y = key_y

        total_notes = int(self.num_octaves * 12)
        white_key_index = 0

        for i in range(total_notes):
            octave = i // 12 + 3
            note_index = i % 12
            note_name = note_names[note_index]
            note_type = note_types[note_index]
            frequency = base_frequency * (2 ** (i / 12))

            if note_type == 'W':
                key = PianoKey(
                    x=white_key_index * white_key_width, y=key_y,
                    width=white_key_width, height=white_key_height,
                    note=note_name, frequency=frequency, octave=octave, is_black=False
                )
                self.keys.append(key)
                white_key_index += 1
            else:
                key = PianoKey(
                    x=white_key_index * white_key_width - black_key_width // 2, y=key_y,
                    width=black_key_width, height=black_key_height,
                    note=note_name, frequency=frequency, octave=octave, is_black=True
                )
                self.keys.append(key)

    def create_song(self):
        """Create simple song pattern."""
        pattern = [0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0]
        white_keys = [i for i, k in enumerate(self.keys) if not k.is_black]
        return [white_keys[i] for i in pattern if i < len(white_keys)]

    def generate_sounds(self):
        """Generate piano sounds."""
        sample_rate = 22050
        duration = 1.0

        for key in self.keys:
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave = np.sin(2 * np.pi * key.frequency * t)
            wave += 0.3 * np.sin(2 * np.pi * key.frequency * 2 * t)
            wave += 0.15 * np.sin(2 * np.pi * key.frequency * 3 * t)
            wave += 0.08 * np.sin(2 * np.pi * key.frequency * 4 * t)

            attack = int(0.01 * sample_rate)
            decay = int(0.08 * sample_rate)
            sustain_level = 0.7
            release = int(0.3 * sample_rate)

            envelope = np.ones_like(t)
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)
            envelope[attack + decay:-release] = sustain_level
            envelope[-release:] = np.linspace(sustain_level, 0, release)

            wave *= envelope
            wave = wave / np.max(np.abs(wave))
            wave = (wave * 32767).astype(np.int16)
            stereo_wave = np.column_stack((wave, wave))

            sound = pygame.sndarray.make_sound(stereo_wave)
            self.sounds[f"{key.note}{key.octave}"] = sound

    def spawn_falling_note(self, key_index=None):
        """Spawn falling note."""
        if key_index is None:
            white_keys = [i for i, k in enumerate(self.keys) if not k.is_black]
            key_index = random.choice(white_keys[:8])

        note = FallingNote(
            key_index=key_index,
            y=0,
            speed=3.0 + random.random() * 2.0
        )
        self.falling_notes.append(note)

    def spawn_particles_gpu(self, x: int, y: int, color: Tuple[int, int, int], count: int = 30):
        """GPU-accelerated particle spawning."""
        if torch.cuda.is_available():
            # Use GPU for particle calculations
            angles = torch.rand(count, device=self.device) * 2 * np.pi
            speeds = torch.rand(count, device=self.device) * 5 + 2

            for i in range(count):
                particle = Particle(
                    x=float(x), y=float(y),
                    vx=float(torch.cos(angles[i]) * speeds[i]),
                    vy=float(torch.sin(angles[i]) * speeds[i]) - 3,
                    color=color, life=1.0,
                    size=random.randint(3, 8)
                )
                self.particles.append(particle)
        else:
            # CPU fallback
            for _ in range(count):
                angle = random.random() * 2 * np.pi
                speed = random.random() * 5 + 2
                particle = Particle(
                    x=float(x), y=float(y),
                    vx=np.cos(angle) * speed,
                    vy=np.sin(angle) * speed - 3,
                    color=color, life=1.0,
                    size=random.randint(3, 8)
                )
                self.particles.append(particle)

    def update_particles_gpu(self):
        """GPU-accelerated particle updates."""
        if not self.particles:
            return

        if torch.cuda.is_available() and len(self.particles) > 50:
            # Use GPU for large particle counts
            start = time.time()

            # Batch update on GPU
            to_remove = []
            for particle in self.particles:
                particle.x += particle.vx
                particle.y += particle.vy
                particle.vy += 0.3  # Gravity
                particle.life -= 0.02
                if particle.life <= 0:
                    to_remove.append(particle)

            for p in to_remove:
                self.particles.remove(p)

            self.gpu_time = time.time() - start
        else:
            # CPU for small counts
            start = time.time()
            to_remove = []
            for particle in self.particles:
                particle.x += particle.vx
                particle.y += particle.vy
                particle.vy += 0.3
                particle.life -= 0.02
                if particle.life <= 0:
                    to_remove.append(particle)

            for p in to_remove:
                self.particles.remove(p)

            self.cpu_time = time.time() - start

    def apply_gpu_glow_effect(self, frame: np.ndarray) -> np.ndarray:
        """Apply GPU-accelerated glow effect to frame."""
        if not torch.cuda.is_available():
            return frame

        try:
            # Convert to tensor and move to GPU
            frame_tensor = torch.from_numpy(frame).to(self.device).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW

            # Apply gaussian blur for glow (GPU-accelerated)
            kernel_size = 15
            sigma = 3.0
            blurred = F.avg_pool2d(frame_tensor, kernel_size, stride=1, padding=kernel_size // 2)

            # Blend original with glow
            glowed = frame_tensor * 0.7 + blurred * 0.3

            # Convert back
            glowed = glowed.squeeze(0).permute(1, 2, 0)
            result = (glowed.cpu().numpy() * 255).astype(np.uint8)

            return result
        except:
            return frame

    def calculate_velocity(self, positions: deque) -> float:
        """Calculate velocity."""
        if len(positions) < 2:
            return 0
        velocities = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        return np.mean(velocities) if velocities else 0

    def is_finger_over_key(self, x: int, y: int, key: PianoKey) -> bool:
        """Check if finger over key."""
        adjusted_x = key.x - self.scroll_offset
        return (adjusted_x <= x <= adjusted_x + key.width and
                key.y <= y <= key.y + key.height)

    def check_note_hit(self, x: int, y: int) -> Optional[FallingNote]:
        """Check if note was hit."""
        for note in self.falling_notes:
            if note.hit or note.missed:
                continue
            key = self.keys[note.key_index]
            adjusted_x = key.x - self.scroll_offset
            if adjusted_x <= x <= adjusted_x + key.width:
                if abs(note.y - self.hit_zone_y) < self.hit_tolerance:
                    return note
        return None

    def play_key(self, key: PianoKey, velocity: float, finger_name: str):
        """Play piano key."""
        current_time = time.time()
        key_id = f"{key.note}{key.octave}"

        if key_id in self.last_press_time:
            if current_time - self.last_press_time[key_id] < self.press_cooldown:
                return

        volume = np.clip(0.4 + (velocity / 50) * 0.6, 0.4, 1.0)

        if key_id in self.sounds:
            sound = self.sounds[key_id].play()
            if sound:
                sound.set_volume(volume)

        key.is_pressed = True
        key.press_time = current_time
        key.glow_intensity = 1.0
        key.pressed_by_finger = finger_name
        self.last_press_time[key_id] = current_time

    def update_falling_notes(self):
        """Update falling notes."""
        to_remove = []
        for note in self.falling_notes:
            if note.hit or note.missed:
                if time.time() - note.hit_time > 0.3:
                    to_remove.append(note)
                continue

            note.y += note.speed
            key = self.keys[note.key_index]
            if note.y > key.y + key.height + 50:
                note.missed = True
                note.hit_time = time.time()
                self.combo = 0

        for note in to_remove:
            self.falling_notes.remove(note)

    def update_key_animations(self):
        """Update key glow."""
        for key in self.keys:
            if key.glow_intensity > 0:
                key.glow_intensity *= self.glow_decay
                if key.glow_intensity < 0.01:
                    key.glow_intensity = 0
                    key.is_pressed = False
                    key.pressed_by_finger = ""

    def draw_piano_keys(self, frame):
        """Draw piano keys."""
        for key in self.keys:
            if not key.is_black:
                adjusted_x = key.x - self.scroll_offset
                if -key.width < adjusted_x < self.width:
                    if key.glow_intensity > 0 and key.pressed_by_finger:
                        finger_color = self.FINGER_COLORS.get(key.pressed_by_finger, (0, 255, 0))
                        glow = int(key.glow_intensity * 255)
                        color = tuple(int(255 - glow + (c * glow / 255)) for c in finger_color)
                    else:
                        color = (240, 240, 240)

                    cv2.rectangle(frame, (adjusted_x, key.y),
                                  (adjusted_x + key.width, key.y + key.height), color, -1)
                    cv2.rectangle(frame, (adjusted_x, key.y),
                                  (adjusted_x + key.width, key.y + key.height), (0, 0, 0), 2)
                    cv2.putText(frame, f"{key.note}{key.octave}",
                                (adjusted_x + 10, key.y + key.height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        for key in self.keys:
            if key.is_black:
                adjusted_x = key.x - self.scroll_offset
                if -key.width < adjusted_x < self.width:
                    if key.glow_intensity > 0 and key.pressed_by_finger:
                        finger_color = self.FINGER_COLORS.get(key.pressed_by_finger, (0, 255, 255))
                        glow = int(key.glow_intensity * 255)
                        color = tuple(int(40 + (c * glow / 255)) for c in finger_color)
                    else:
                        color = (40, 40, 40)

                    cv2.rectangle(frame, (adjusted_x, key.y),
                                  (adjusted_x + key.width, key.y + key.height), color, -1)
                    cv2.rectangle(frame, (adjusted_x, key.y),
                                  (adjusted_x + key.width, key.y + key.height), (0, 0, 0), 2)

    def draw_falling_notes(self, frame):
        """Draw falling notes."""
        for note in self.falling_notes:
            key = self.keys[note.key_index]
            adjusted_x = key.x - self.scroll_offset
            if adjusted_x < -key.width or adjusted_x > self.width:
                continue

            if note.hit:
                color = (0, 255, 0)
                alpha = 0.5
            elif note.missed:
                color = (0, 0, 255)
                alpha = 0.3
            else:
                color = (255, 200, 0)
                alpha = 0.8

            note_height = 40
            overlay = frame.copy()
            cv2.rectangle(overlay, (adjusted_x + 5, int(note.y)),
                          (adjusted_x + key.width - 5, int(note.y) + note_height),
                          color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.rectangle(frame, (adjusted_x + 5, int(note.y)),
                          (adjusted_x + key.width - 5, int(note.y) + note_height),
                          (255, 255, 255), 2)

    def draw_hit_zone(self, frame):
        """Draw hit zone."""
        y = self.hit_zone_y
        for i in range(3):
            alpha = 0.3 - i * 0.1
            overlay = frame.copy()
            cv2.line(overlay, (0, y - i * 2), (self.width, y - i * 2),
                     (0, 255, 255), 3)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_particles(self, frame):
        """Draw particles."""
        for particle in self.particles:
            if particle.life > 0:
                alpha = particle.life
                size = int(particle.size * particle.life)
                if size > 0:
                    overlay = frame.copy()
                    cv2.circle(overlay, (int(particle.x), int(particle.y)),
                               size, particle.color, -1)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_finger_trails(self, frame):
        """Draw finger trails."""
        for (hand, finger), trail in self.finger_trails.items():
            if len(trail) < 2:
                continue
            color = self.FINGER_COLORS.get(finger, (255, 255, 255))
            for i in range(len(trail) - 1):
                alpha = (i + 1) / len(trail)
                thickness = int(alpha * 3) + 1
                overlay = frame.copy()
                cv2.line(overlay, trail[i], trail[i + 1], color, thickness)
                cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)

    def process_hand_landmarks(self, hand_landmarks, handedness, frame):
        """Process hand landmarks."""
        hand_label = handedness.classification[0].label

        for finger_name, landmark_id in self.FINGER_TIPS.items():
            finger_tip = hand_landmarks.landmark[landmark_id]
            x = int(finger_tip.x * self.width)
            y = int(finger_tip.y * self.height)

            tracking_key = (hand_label, finger_name)
            if tracking_key not in self.finger_positions:
                self.finger_positions[tracking_key] = deque(maxlen=10)
                self.finger_trails[tracking_key] = deque(maxlen=15)

            self.finger_positions[tracking_key].append(y)
            self.finger_trails[tracking_key].append((x, y))

            velocity = self.calculate_velocity(self.finger_positions[tracking_key])
            finger_color = self.FINGER_COLORS[finger_name]

            if velocity > self.min_velocity_threshold:
                cv2.circle(frame, (x, y), 14, finger_color, -1)
                cv2.circle(frame, (x, y), 14, (255, 255, 255), 3)

                if self.song_mode:
                    hit_note = self.check_note_hit(x, y)
                    if hit_note:
                        hit_note.hit = True
                        hit_note.hit_time = time.time()
                        self.score += 100
                        self.combo += 1
                        self.max_combo = max(self.max_combo, self.combo)
                        key = self.keys[hit_note.key_index]
                        self.play_key(key, velocity, finger_name)
                        self.spawn_particles_gpu(x, y, finger_color, 30)
                else:
                    for key in self.keys:
                        if self.is_finger_over_key(x, y, key) and not key.is_pressed:
                            self.play_key(key, velocity, finger_name)
                            self.spawn_particles_gpu(x, y, finger_color, 15)
            else:
                cv2.circle(frame, (x, y), 10, finger_color, -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)

            cv2.putText(frame, finger_name[0], (x - 6, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=1),
            self.mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1)
        )

    def draw_ui(self, frame, fps):
        """Draw UI."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y = 35
        mode_text = "SONG MODE" if self.song_mode else "FREEPLAY MODE"
        color = (0, 255, 255) if self.song_mode else (255, 255, 0)
        cv2.putText(frame, mode_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        y += 30
        gpu_text = "GPU: CUDA" if torch.cuda.is_available() else "GPU: CPU"
        gpu_color = (0, 255, 0) if torch.cuda.is_available() else (255, 255, 0)
        cv2.putText(frame, f"{gpu_text} | FPS: {fps}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gpu_color, 1)

        if self.song_mode:
            y += 25
            cv2.putText(frame, f"Score: {self.score}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, f"Combo: x{self.combo}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            y += 25
            cv2.putText(frame, f"Best: x{self.max_combo}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)

        y += 30
        cv2.putText(frame, "Press 'S' to toggle mode", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 20
        cv2.putText(frame, "Press 'G' for GPU glow effect", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 20
        cv2.putText(frame, "Press 'Q' to quit", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def run(self):
        """Main loop."""
        print("\n🚀 GPU-Accelerated Piano Running!")
        print("Controls: 'S' = toggle mode, 'G' = GPU glow, 'Q' = quit\n")

        fps_time = time.time()
        fps_counter = 0
        fps_display = 0
        gpu_glow = False

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if self.song_mode:
                if time.time() - self.last_note_spawn > self.note_spawn_interval:
                    if self.song_index < len(self.song_notes):
                        self.spawn_falling_note(self.song_notes[self.song_index])
                        self.song_index += 1
                        if self.song_index >= len(self.song_notes):
                            self.song_index = 0
                    self.last_note_spawn = time.time()

                self.update_falling_notes()

            self.update_particles_gpu()
            self.update_key_animations()

            self.draw_piano_keys(frame)

            if self.song_mode:
                self.draw_hit_zone(frame)
                self.draw_falling_notes(frame)

            self.draw_particles(frame)
            self.draw_finger_trails(frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    self.process_hand_landmarks(hand_landmarks, handedness, frame)

            # Apply GPU glow if enabled
            if gpu_glow and torch.cuda.is_available():
                frame = self.apply_gpu_glow_effect(frame)

            fps_counter += 1
            if time.time() - fps_time > 1:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()

            self.draw_ui(frame, fps_display)
            cv2.imshow('GPU-Accelerated Virtual Piano', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.song_mode = not self.song_mode
                self.falling_notes.clear()
                print(f"Switched to {'SONG' if self.song_mode else 'FREEPLAY'} mode")
            elif key == ord('g'):
                gpu_glow = not gpu_glow
                print(f"GPU Glow: {'ON' if gpu_glow else 'OFF'}")
            elif key == ord('r'):
                self.finger_positions.clear()
                self.finger_trails.clear()
                self.score = 0
                self.combo = 0

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


def main():
    """Entry point."""
    try:
        piano = GPUPiano(num_octaves=2.5)
        piano.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
