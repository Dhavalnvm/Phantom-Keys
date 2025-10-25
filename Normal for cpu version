#!/usr/bin/env python3
"""
Virtual Piano - ALL FINGERS Hand Gesture Controlled Piano
Control a virtual piano using ALL 5 FINGERS via webcam.
Each finger can play independently!

Controls:
- Use ANY finger to play keys (thumb, index, middle, ring, pinky)
- Each finger has its own color
- Move finger down quickly over a key to play it
- Press LEFT/RIGHT arrows to scroll
- Press 'q' to quit
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class PianoKey:
    """Represents a piano key with its position and state."""
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
    pressed_by_finger: str = ""  # Which finger pressed it
    

class MultiFingerPiano:
    """Piano that tracks all 5 fingers independently."""
    
    # MediaPipe hand landmarks for finger tips
    FINGER_TIPS = {
        'Thumb': 4,
        'Index': 8,
        'Middle': 12,
        'Ring': 16,
        'Pinky': 20
    }
    
    # Color for each finger (BGR format)
    FINGER_COLORS = {
        'Thumb': (255, 150, 0),    # Orange
        'Index': (0, 255, 0),      # Green
        'Middle': (0, 255, 255),   # Yellow
        'Ring': (255, 0, 255),     # Magenta
        'Pinky': (255, 100, 255)   # Pink
    }
    
    def __init__(self, num_octaves=2.5):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize Pygame for audio
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual camera dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Piano configuration
        self.num_octaves = num_octaves
        self.scroll_offset = 0
        self.scroll_speed = 50
        
        # Piano keys
        self.keys: List[PianoKey] = []
        self.create_full_keyboard()
        
        # Generate sounds
        self.sounds = {}
        self.generate_piano_sounds()
        
        # Track each finger independently
        # Format: {(hand_label, finger_name): deque of y positions}
        self.finger_positions: Dict[Tuple[str, str], deque] = {}
        self.last_press_time: Dict[str, float] = {}
        self.press_cooldown = 0.15
        
        # Animation settings
        self.glow_decay = 0.90
        self.min_velocity_threshold = 12
        
        # Visual effects
        self.particle_effects = []  # Store particle effects
        
    def create_full_keyboard(self):
        """Create a full piano keyboard with multiple octaves."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_types = ['W', 'B', 'W', 'B', 'W', 'W', 'B', 'W', 'B', 'W', 'B', 'W']
        
        base_frequency = 130.81  # C3
        
        white_key_width = 80
        white_key_height = 200
        black_key_width = 50
        black_key_height = 120
        key_y = self.height - white_key_height - 20
        
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
                    x=white_key_index * white_key_width,
                    y=key_y,
                    width=white_key_width,
                    height=white_key_height,
                    note=note_name,
                    frequency=frequency,
                    octave=octave,
                    is_black=False
                )
                self.keys.append(key)
                white_key_index += 1
            else:
                key = PianoKey(
                    x=white_key_index * white_key_width - black_key_width // 2,
                    y=key_y,
                    width=black_key_width,
                    height=black_key_height,
                    note=note_name,
                    frequency=frequency,
                    octave=octave,
                    is_black=True
                )
                self.keys.append(key)
    
    def generate_piano_sounds(self):
        """Generate piano-like sounds for each key."""
        sample_rate = 22050
        duration = 1.2
        
        for key in self.keys:
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Fundamental + harmonics
            wave = np.sin(2 * np.pi * key.frequency * t)
            wave += 0.3 * np.sin(2 * np.pi * key.frequency * 2 * t)
            wave += 0.15 * np.sin(2 * np.pi * key.frequency * 3 * t)
            wave += 0.08 * np.sin(2 * np.pi * key.frequency * 4 * t)
            wave += 0.04 * np.sin(2 * np.pi * key.frequency * 5 * t)
            
            # ADSR envelope
            attack = int(0.01 * sample_rate)
            decay = int(0.1 * sample_rate)
            sustain_level = 0.6
            release = int(0.4 * sample_rate)
            
            envelope = np.ones_like(t)
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)
            envelope[attack+decay:-release] = sustain_level
            envelope[-release:] = np.linspace(sustain_level, 0, release)
            
            wave *= envelope
            wave = wave / np.max(np.abs(wave))
            wave = (wave * 32767).astype(np.int16)
            stereo_wave = np.column_stack((wave, wave))
            
            sound = pygame.sndarray.make_sound(stereo_wave)
            key_id = f"{key.note}{key.octave}"
            self.sounds[key_id] = sound
    
    def calculate_velocity(self, positions: deque) -> float:
        """Calculate downward velocity of finger movement."""
        if len(positions) < 2:
            return 0
        
        velocities = []
        for i in range(len(positions) - 1):
            dy = positions[i+1] - positions[i]
            velocities.append(dy)
        
        return np.mean(velocities) if velocities else 0
    
    def is_finger_over_key(self, x: int, y: int, key: PianoKey) -> bool:
        """Check if finger position is over a key."""
        adjusted_x = key.x - self.scroll_offset
        return (adjusted_x <= x <= adjusted_x + key.width and 
                key.y <= y <= key.y + key.height)
    
    def play_key(self, key: PianoKey, velocity: float, finger_name: str):
        """Play a piano key with velocity sensitivity."""
        current_time = time.time()
        key_id = f"{key.note}{key.octave}"
        
        # Debounce
        if key_id in self.last_press_time:
            if current_time - self.last_press_time[key_id] < self.press_cooldown:
                return
        
        # Volume based on velocity
        volume = np.clip(0.3 + (velocity / 50) * 0.7, 0.3, 1.0)
        
        # Play sound
        if key_id in self.sounds:
            sound = self.sounds[key_id].play()
            if sound:
                sound.set_volume(volume)
        
        # Update key state
        key.is_pressed = True
        key.press_time = current_time
        key.glow_intensity = 1.0
        key.pressed_by_finger = finger_name
        self.last_press_time[key_id] = current_time
    
    def draw_piano_keys(self, frame):
        """Draw the piano keyboard on the frame."""
        # Draw white keys first
        for key in self.keys:
            if not key.is_black:
                adjusted_x = key.x - self.scroll_offset
                
                if -key.width < adjusted_x < self.width:
                    # Color based on which finger pressed it
                    if key.glow_intensity > 0 and key.pressed_by_finger:
                        finger_color = self.FINGER_COLORS.get(key.pressed_by_finger, (0, 255, 0))
                        glow = int(key.glow_intensity * 255)
                        # Mix white with finger color
                        color = tuple(int(255 - glow + (c * glow / 255)) for c in finger_color)
                    else:
                        color = (255, 255, 255)
                    
                    # Draw key
                    cv2.rectangle(frame, (adjusted_x, key.y),
                                (adjusted_x + key.width, key.y + key.height),
                                color, -1)
                    cv2.rectangle(frame, (adjusted_x, key.y),
                                (adjusted_x + key.width, key.y + key.height),
                                (0, 0, 0), 2)
                    
                    # Draw note label
                    label = f"{key.note}{key.octave}"
                    cv2.putText(frame, label, (adjusted_x + 10, key.y + key.height - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw black keys
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
                                (adjusted_x + key.width, key.y + key.height),
                                color, -1)
                    cv2.rectangle(frame, (adjusted_x, key.y),
                                (adjusted_x + key.width, key.y + key.height),
                                (0, 0, 0), 2)
                    
                    label = f"{key.note}{key.octave}"
                    label_color = (255, 255, 255) if key.glow_intensity > 0.3 else (200, 200, 200)
                    cv2.putText(frame, label, (adjusted_x + 5, key.y + key.height - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
    
    def update_key_animations(self):
        """Update glow animations for all keys."""
        for key in self.keys:
            if key.glow_intensity > 0:
                key.glow_intensity *= self.glow_decay
                if key.glow_intensity < 0.01:
                    key.glow_intensity = 0
                    key.is_pressed = False
                    key.pressed_by_finger = ""
    
    def process_hand_landmarks(self, hand_landmarks, handedness, frame):
        """Process all 5 fingers from hand landmarks."""
        hand_label = handedness.classification[0].label
        
        # Process each finger
        for finger_name, landmark_id in self.FINGER_TIPS.items():
            finger_tip = hand_landmarks.landmark[landmark_id]
            x = int(finger_tip.x * self.width)
            y = int(finger_tip.y * self.height)
            
            # Create tracking key
            tracking_key = (hand_label, finger_name)
            
            # Initialize tracking if needed
            if tracking_key not in self.finger_positions:
                self.finger_positions[tracking_key] = deque(maxlen=10)
            
            # Track position
            self.finger_positions[tracking_key].append(y)
            
            # Calculate velocity
            velocity = self.calculate_velocity(self.finger_positions[tracking_key])
            
            # Get finger color
            finger_color = self.FINGER_COLORS[finger_name]
            
            # Draw finger tip
            if velocity > self.min_velocity_threshold:
                # Active (playing)
                cv2.circle(frame, (x, y), 12, finger_color, -1)
                cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
                
                # Draw velocity indicator
                cv2.putText(frame, f"{int(velocity)}", (x + 15, y - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, finger_color, 1)
            else:
                # Inactive
                cv2.circle(frame, (x, y), 8, finger_color, -1)
                cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)
            
            # Label finger
            cv2.putText(frame, finger_name[0], (x - 5, y - 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Check if playing a key
            if velocity > self.min_velocity_threshold:
                pressed_key = None
                
                # Check black keys first
                for key in self.keys:
                    if key.is_black and self.is_finger_over_key(x, y, key):
                        pressed_key = key
                        break
                
                # Check white keys
                if pressed_key is None:
                    for key in self.keys:
                        if not key.is_black and self.is_finger_over_key(x, y, key):
                            pressed_key = key
                            break
                
                # Play key
                if pressed_key and not pressed_key.is_pressed:
                    self.play_key(pressed_key, velocity, finger_name)
        
        # Draw hand skeleton
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=2),
            self.mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1)
        )
    
    def handle_keyboard_scroll(self, key):
        """Handle keyboard arrow keys for scrolling."""
        if key == 81 or key == 2:  # Left arrow
            self.scroll_offset = max(0, self.scroll_offset - self.scroll_speed)
        elif key == 83 or key == 3:  # Right arrow
            max_scroll = max(0, (len([k for k in self.keys if not k.is_black]) * 80) - self.width)
            self.scroll_offset = min(max_scroll, self.scroll_offset + self.scroll_speed)
    
    def draw_finger_legend(self, frame):
        """Draw a legend showing which color = which finger."""
        legend_x = self.width - 200
        legend_y = 20
        
        # Background
        cv2.rectangle(frame, (legend_x - 10, legend_y - 5),
                     (self.width - 10, legend_y + 155), (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x - 10, legend_y - 5),
                     (self.width - 10, legend_y + 155), (255, 255, 255), 1)
        
        cv2.putText(frame, "Finger Colors:", (legend_x, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset = 40
        for finger_name, color in self.FINGER_COLORS.items():
            cv2.circle(frame, (legend_x + 10, legend_y + y_offset), 8, color, -1)
            cv2.circle(frame, (legend_x + 10, legend_y + y_offset), 8, (255, 255, 255), 1)
            cv2.putText(frame, finger_name, (legend_x + 25, legend_y + y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 25
    
    def run(self):
        """Main loop."""
        print("Virtual Piano - ALL 5 FINGERS Control")
        print("=" * 60)
        print(f"Keyboard: {self.num_octaves} octaves ({len(self.keys)} keys)")
        print("=" * 60)
        print("Instructions:")
        print("- ALL 5 FINGERS can play independently!")
        print("- Each finger has its own color")
        print("- Move ANY finger down quickly to play")
        print("- Use LEFT/RIGHT arrows to scroll")
        print("- Press 'q' to quit")
        print("=" * 60)
        
        fps_time = time.time()
        fps_counter = 0
        fps_display = 0
        
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            self.draw_piano_keys(frame)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    self.process_hand_landmarks(hand_landmarks, handedness, frame)
            
            self.update_key_animations()
            
            # FPS
            fps_counter += 1
            if time.time() - fps_time > 1:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Info panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, f"FPS: {fps_display}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"ALL 5 FINGERS ACTIVE", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Move fingers DOWN to play", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw finger legend
            self.draw_finger_legend(frame)
            
            # Scroll bar
            total_keys = len([k for k in self.keys if not k.is_black])
            max_scroll = max(1, (total_keys * 80) - self.width)
            cv2.rectangle(frame, (10, self.height - 40), (self.width - 10, self.height - 10),
                         (50, 50, 50), -1)
            scroll_bar_width = int((self.width - 20) * (self.width / (total_keys * 80)))
            scroll_bar_x = int(10 + (self.scroll_offset / max_scroll) * (self.width - 20 - scroll_bar_width)) if max_scroll > 0 else 10
            cv2.rectangle(frame, (scroll_bar_x, self.height - 40),
                         (scroll_bar_x + scroll_bar_width, self.height - 10), (0, 255, 0), -1)
            
            cv2.imshow('Virtual Piano - All Fingers', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.finger_positions.clear()
                self.last_press_time.clear()
            else:
                self.handle_keyboard_scroll(key)
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


def main():
    """Entry point."""
    print("\n🎹 Virtual Piano - ALL FINGERS Edition")
    print("\nChoose keyboard size:")
    print("1. Small (1 octave)")
    print("2. Medium (2 octaves)")
    print("3. Large (2.5 octaves) [Recommended]")
    print("4. Extra Large (3 octaves)")
    
    try:
        choice = input("\nEnter choice (1-4) or press Enter for default [3]: ").strip()
        if not choice:
            choice = "3"
        
        octave_map = {"1": 1.0, "2": 2.0, "3": 2.5, "4": 3.0}
        num_octaves = octave_map.get(choice, 2.5)
        
        print(f"\n🎵 Starting piano with {num_octaves} octaves...")
        print("🖐️  ALL 5 FINGERS will be tracked!\n")
        
        piano = MultiFingerPiano(num_octaves=num_octaves)
        piano.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
