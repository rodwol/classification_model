import os
import numpy as np
from preprocessing import AudioPreprocessor
from pathlib import Path
import cv2

def get_label_from_annotation(txt_path):
    """Extract label from annotation text file"""
    try:
        with open(txt_path, 'r') as f:
            content = f.read().lower()
            
        # Look for respiratory sound labels in the annotation
        if 'crackle' in content and 'wheeze' in content:
            return 3  # both
        elif 'crackle' in content:
            return 1  # crackle
        elif 'wheeze' in content:
            return 2  # wheeze
        elif 'normal' in content or 'healthy' in content:
            return 0  # normal
        else:
            # If no specific labels found, check for other indicators
            if 'adventitious' not in content and 'abnormal' not in content:
                return 0  # assume normal if no abnormal indicators
            return None  # unknown
            
    except Exception as e:
        print(f"Error reading annotation file {txt_path}: {str(e)}")
        return None

def build_dataset(raw_audio_dir, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = AudioPreprocessor(
        target_sr=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        duration=3.0,
        target_shape=(224, 224)
    )

    X = []
    y = []

    audio_files = [f for f in os.listdir(raw_audio_dir) if f.endswith(".wav")]
    
    print(f"Found {len(audio_files)} audio files")
    
    processed_count = 0
    skipped_count = 0
    
    for i, filename in enumerate(audio_files):
        wav_path = os.path.join(raw_audio_dir, filename)
        txt_path = wav_path.replace(".wav", ".txt")

        if not os.path.exists(txt_path):
            print(f"Missing annotation: {txt_path}")
            skipped_count += 1
            continue

        # Get label from annotation file
        label = get_label_from_annotation(txt_path)

        if label is None:
            print(f"Skipping unknown label: {filename}")
            skipped_count += 1
            continue

        print(f"Processing {i+1}/{len(audio_files)}: {filename} (label: {label})")

        # Convert to spectrogram
        try:
            spec = preprocessor.generate_mel_spectrogram(wav_path)
            
            if spec is not None:
                # Normalize spectrogram if needed
                if spec.max() > 1.0:
                    spec = spec / spec.max()
                
                X.append(spec)
                y.append(label)
                processed_count += 1
                print(f"Successfully processed: {filename} (label: {label})")
            else:
                print(f"Failed to generate spectrogram for: {filename}")
                skipped_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            skipped_count += 1
            continue

    # Check if we have any data before saving
    if len(X) == 0:
        print("No data was processed successfully!")
        print(f"Processed: {processed_count}, Skipped: {skipped_count}")
        return
    
    X = np.array(X)
    y = np.array(y)

    print("\nFinal dataset summary:")
    print(f"Total files found: {len(audio_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    label_names = {0: "normal", 1: "crackle", 2: "wheeze", 3: "both"}
    print("\nClass distribution:")
    for lbl, count in zip(unique, counts):
        print(f"  {label_names[lbl]}: {count} samples")
    
    # Add channel dimension if needed for CNN
    if len(X.shape) == 3:
        X = X.reshape(X.shape + (1,))
        print("X reshaped to:", X.shape)

    # Normalize the entire dataset
    if X.max() > 1.0:
        X = X / 255.0

    # Save processed data
    np.save(save_dir / "X.npy", X)
    np.save(save_dir / "y.npy", y)

    print(f"\nSaved {len(X)} samples to: {save_dir}")

# Let's also create a function to inspect annotation files to understand their structure
def inspect_annotations(raw_audio_dir, num_files=5):
    """Inspect a few annotation files to understand their structure"""
    audio_files = [f for f in os.listdir(raw_audio_dir) if f.endswith(".wav")]
    
    print("Inspecting annotation files structure:")
    print("-" * 50)
    
    for i, filename in enumerate(audio_files[:num_files]):
        txt_path = os.path.join(raw_audio_dir, filename.replace(".wav", ".txt"))
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    content = f.read()
                print(f"\n{filename}:")
                print(f"Annotation content: {content.strip()}")
                label = get_label_from_annotation(txt_path)
                print(f"Extracted label: {label}")
            except Exception as e:
                print(f"Error reading {txt_path}: {str(e)}")

if __name__ == "__main__":
    # First, let's inspect some annotation files to understand their structure
    raw_audio_dir = "data/Respiratory_Sound_Database/audio_and_txt_files"
    inspect_annotations(raw_audio_dir)
    
    # Then build the dataset
    build_dataset(
        raw_audio_dir=raw_audio_dir,
        save_dir="data/processed"
    )