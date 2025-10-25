#!/usr/bin/env python3
"""
Setup script for Virtual Piano
Handles dependency installation and fixes version conflicts
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error occurred:")
        print(e.stderr)
        return False

def main():
    """Main setup function."""
    print("""
╔════════════════════════════════════════════════════════════╗
║         🎹 Virtual Piano - Setup Script 🎹                 ║
║                                                            ║
║  This script will fix dependency conflicts and install     ║
║  all required packages with compatible versions.          ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📋 Current Python version:")
    print(f"   {sys.version}\n")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("⚠️  WARNING: You are not in a virtual environment!")
        print("   It's recommended to use a virtual environment.")
        print("\n   To create one:")
        print("   python -m venv piano_env")
        if os.name == 'nt':  # Windows
            print("   piano_env\\Scripts\\activate")
        else:  # Unix/Linux/Mac
            print("   source piano_env/bin/activate")
        print("\n   Then run this script again.\n")
        
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Step 1: Uninstall conflicting packages
    print("\n" + "="*60)
    print("Step 1: Removing conflicting packages")
    print("="*60)
    packages_to_remove = ['numpy', 'protobuf', 'mediapipe', 'opencv-python', 'pygame']
    
    for package in packages_to_remove:
        print(f"Uninstalling {package}...")
        subprocess.run(f"{sys.executable} -m pip uninstall -y {package}", 
                      shell=True, capture_output=True)
    
    print("✅ Old packages removed")
    
    # Step 2: Install compatible versions
    print("\n" + "="*60)
    print("Step 2: Installing compatible versions")
    print("="*60)
    
    packages = [
        "numpy==1.26.4",
        "protobuf==4.25.5",
        "opencv-python==4.10.0.84",
        "mediapipe==0.10.21",
        "pygame==2.6.1"
    ]
    
    for package in packages:
        success = run_command(
            f"{sys.executable} -m pip install {package}",
            f"Installing {package}"
        )
        if not success:
            print(f"\n⚠️  Warning: Failed to install {package}")
            print("Continuing with other packages...")
    
    # Step 3: Verify installation
    print("\n" + "="*60)
    print("Step 3: Verifying installation")
    print("="*60)
    
    try:
        import numpy
        import cv2
        import mediapipe
        import pygame
        
        print("\n✅ All packages imported successfully!")
        print(f"\n📦 Installed versions:")
        print(f"   NumPy: {numpy.__version__}")
        print(f"   OpenCV: {cv2.__version__}")
        print(f"   MediaPipe: {mediapipe.__version__}")
        print(f"   Pygame: {pygame.__version__}")
        
        # Test camera
        print("\n" + "="*60)
        print("🎥 Testing camera access...")
        print("="*60)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera detected and accessible!")
            cap.release()
        else:
            print("⚠️  Warning: Could not access camera")
            print("   Make sure your webcam is connected and not in use")
        
        print("\n" + "="*60)
        print("🎉 Setup Complete!")
        print("="*60)
        print("\nYou can now run the piano:")
        print("  python virtual_piano.py          # Original (1 octave)")
        print("  python virtual_piano_full.py     # Full keyboard (2.5 octaves)")
        
    except ImportError as e:
        print(f"\n❌ Error: Could not import required package")
        print(f"   {e}")
        print("\nTry running setup again or install packages manually:")
        print("  pip install numpy==1.26.4 protobuf==4.25.5")
        print("  pip install opencv-python==4.10.0.84")
        print("  pip install mediapipe==0.10.21 pygame==2.6.1")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
