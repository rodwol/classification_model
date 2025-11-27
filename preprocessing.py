import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import cv2

class AudioPreprocessor:
    def __init__(self, target_sr=22050, n_mels=128, n_fft=2048, hop_length=512, 
                 duration=3.0, target_shape=(224, 224)):
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.target_shape = target_shape
    
    def load_and_fix_audio(self, audio_path):
        """Load audio and ensure consistent duration"""
        try:
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Fix duration by padding or trimming
            target_length = int(self.duration * self.target_sr)
            if len(y) > target_length:
                y = y[:target_length]
            elif len(y) < target_length:
                padding = target_length - len(y)
                y = np.pad(y, (0, padding), mode='constant')
            
            return y, sr
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            return None, None
    
    def generate_mel_spectrogram(self, audio_path, save_path=None, visualize=False):
        """Generate Mel spectrogram from audio file"""
        y, sr = self.load_and_fix_audio(audio_path)
        if y is None:
            return None
        
        # Generate Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Resize to target shape
        mel_spec_resized = cv2.resize(mel_spec_normalized, self.target_shape)
        
        if visualize:
            self._plot_spectrogram(mel_spec_db, sr, save_path)
        
        if save_path:
            # Save as image for CNN training
            plt.imsave(save_path, mel_spec_resized, cmap='viridis')
        
        return mel_spec_resized
    
    def _plot_spectrogram(self, spectrogram, sr, save_path=None):
        """Plot and save spectrogram visualization"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        librosa.display.specshow(spectrogram, sr=sr, hop_length=self.hop_length, 
                                x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (dB)')
        
        plt.subplot(1, 2, 2)
        # Show normalized version
        spectrogram_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        librosa.display.specshow(spectrogram_norm, sr=sr, hop_length=self.hop_length, 
                                x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar()
        plt.title('Normalized Mel Spectrogram')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_visualization.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
