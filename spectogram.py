#!/usr/bin/env python3
"""
Audio Spectral Analysis Tool

Generates spectograms to visualize frequency content over time.
Supports basic and advanced analysis with silence detection and spectral features.

Examples:
    # Basic spectogram
    python3 spectogram.py audio.wav
    
    # Mel-scale spectogram with spectral centroid
    python3 spectogram.py audio.mp3 --mel --centroid
    
    # Save without display, custom FFT size
    python3 spectogram.py song.flac --n-fft 4096 --no-display -o output.png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# Audio Processing

def load_audio(path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio file and return signal + sample rate."""
    try:
        signal, sample_rate = librosa.load(path, sr=sr, mono=True)
        return signal, sample_rate
    except Exception as e:
        print(f"Error loading audio: {e}", file=sys.stderr)
        sys.exit(1)


def remove_silence(signal: np.ndarray, top_db: float = 60) -> np.ndarray:
    """Remove silent sections below threshold."""
    intervals = librosa.effects.split(signal, top_db=top_db)
    
    if len(intervals) == 0:
        print(f"Warning: No audio above {top_db}dB threshold", file=sys.stderr)
        return np.array([])
    
    # Concatenate non-silent intervals
    trimmed = np.concatenate([signal[start:end] for start, end in intervals])
    
    silence_removed = len(signal) - len(trimmed)
    pct = (silence_removed / len(signal)) * 100
    print(f"Silence removed: {pct:.1f}% ({silence_removed:,} samples)")
    
    return trimmed


# Spectrogram Computation

def compute_stft(
    signal: np.ndarray,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    window: str = 'hann'
) -> np.ndarray:
    """Compute Short-Time Fourier Transform magnitude."""
    if hop_length is None:
        hop_length = n_fft // 4
    
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window=window)
    return np.abs(D)


def compute_mel_spectrogram(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: Optional[int] = None
) -> np.ndarray:
    """Compute Mel-scale spectrogram."""
    if hop_length is None:
        hop_length = n_fft // 4
    
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mel_spec


def to_db(magnitude: np.ndarray) -> np.ndarray:
    """Convert magnitude to decibels."""
    return librosa.amplitude_to_db(magnitude, ref=np.max)


# Spectral Features

def compute_spectral_centroid(
    signal: np.ndarray,
    sr: int,
    magnitude: np.ndarray,
    hop_length: int
) -> np.ndarray:
    """Compute spectral centroid (center of mass of spectrum)."""
    return librosa.feature.spectral_centroid(
        y=signal,
        sr=sr,
        S=magnitude,
        hop_length=hop_length
    )[0]


def compute_spectral_bandwidth(
    signal: np.ndarray,
    sr: int,
    magnitude: np.ndarray,
    hop_length: int
) -> np.ndarray:
    """Compute spectral bandwidth (spread around centroid)."""
    return librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr,
        S=magnitude,
        hop_length=hop_length
    )[0]


def compute_spectral_rolloff(
    signal: np.ndarray,
    sr: int,
    magnitude: np.ndarray,
    hop_length: int,
    roll_percent: float = 0.85
) -> np.ndarray:
    """Compute spectral roll-off (frequency below which X% of energy is contained)."""
    return librosa.feature.spectral_rolloff(
        y=signal,
        sr=sr,
        S=magnitude,
        hop_length=hop_length,
        roll_percent=roll_percent
    )[0]


# Visualization

def plot_spectrogram(
    spec_db: np.ndarray,
    sr: int,
    hop_length: int,
    title: str,
    use_mel: bool = False,
    fmax: Optional[float] = None,
    features: Optional[dict] = None,
    output_path: Optional[str] = None,
    display: bool = True
) -> None:
    """
    Plot spectrogram with optional spectral features.
    
    Args:
        spec_db: Spectrogram in dB
        sr: Sample rate
        hop_length: Hop length used in STFT
        title: Plot title
        use_mel: Whether this is a Mel spectrogram
        fmax: Max frequency to display
        features: Dict with 'centroid', 'bandwidth', 'rolloff' arrays
        output_path: Save path (if None, don't save)
        display: Show plot interactively
    """
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    
    # Plot spectrogram
    y_axis = 'mel' if use_mel else 'log'
    img = librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis=y_axis,
        cmap='viridis',
        fmax=fmax,
        ax=ax
    )
    
    plt.colorbar(img, format='%+2.0f dB', label='Amplitude [dB]')
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]' if not use_mel else 'Mel Frequency')
    
    # Overlay spectral features
    if features:
        times = librosa.frames_to_time(
            np.arange(spec_db.shape[1]),
            sr=sr,
            hop_length=hop_length
        )
        
        if 'centroid' in features:
            ax.plot(times, features['centroid'], 
                   color='cyan', linestyle='--', linewidth=1.5,
                   label='Spectral Centroid')
        
        if 'bandwidth' in features and 'centroid' in features:
            centroid = features['centroid']
            bandwidth = features['bandwidth']
            ax.fill_between(times, 
                           centroid - bandwidth,
                           centroid + bandwidth,
                           color='yellow', alpha=0.2,
                           label='Bandwidth')
        
        if 'rolloff' in features:
            ax.plot(times, features['rolloff'],
                   color='red', linestyle='-.', linewidth=1.5,
                   label='Spectral Rolloff (85%)')
        
        if features:
            ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error saving plot: {e}", file=sys.stderr)
    
    # Display if requested
    if display:
        plt.show()
    else:
        plt.close()


# Main

def analyze_audio(
    input_path: str,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    window: str = 'hann',
    silence_threshold: Optional[float] = 60,
    use_mel: bool = False,
    fmax: Optional[float] = None,
    show_centroid: bool = False,
    show_bandwidth: bool = False,
    show_rolloff: bool = False,
    output_path: Optional[str] = None,
    display: bool = True
) -> None:
    
    # Load audio
    print(f"Loading: {input_path}")
    signal, sr = load_audio(input_path)
    duration = len(signal) / sr
    print(f"Duration: {duration:.2f}s, Sample rate: {sr} Hz")
    
    # Remove silence
    if silence_threshold is not None:
        signal = remove_silence(signal, top_db=silence_threshold)
        if len(signal) == 0:
            print("Error: No audio remaining after silence removal", file=sys.stderr)
            sys.exit(1)
    
    # Set hop_length default
    if hop_length is None:
        hop_length = n_fft // 4
    
    # Compute spectrogram
    if use_mel:
        print(f"Computing Mel spectrogram (n_fft={n_fft}, hop={hop_length})...")
        magnitude = compute_mel_spectrogram(signal, sr, n_fft, hop_length)
        spec_db = to_db(magnitude)
        y_label = "Mel Spectrogram"
    else:
        print(f"Computing STFT (n_fft={n_fft}, hop={hop_length}, window={window})...")
        magnitude = compute_stft(signal, n_fft, hop_length, window)
        spec_db = to_db(magnitude)
        y_label = "Spectrogram"
    
    # Compute spectral features if requested
    features = {}
    
    if show_centroid or show_bandwidth:
        print("Computing spectral centroid...")
        # For Mel, recompute STFT magnitude for accurate features
        if use_mel:
            stft_mag = compute_stft(signal, n_fft, hop_length, window)
        else:
            stft_mag = magnitude
        
        centroid = compute_spectral_centroid(signal, sr, stft_mag, hop_length)
        features['centroid'] = centroid
    
    if show_bandwidth and 'centroid' in features:
        print("Computing spectral bandwidth...")
        if use_mel:
            stft_mag = compute_stft(signal, n_fft, hop_length, window)
        else:
            stft_mag = magnitude
        bandwidth = compute_spectral_bandwidth(signal, sr, stft_mag, hop_length)
        features['bandwidth'] = bandwidth
    
    if show_rolloff:
        print("Computing spectral rolloff...")
        if use_mel:
            stft_mag = compute_stft(signal, n_fft, hop_length, window)
        else:
            stft_mag = magnitude
        rolloff = compute_spectral_rolloff(signal, sr, stft_mag, hop_length)
        features['rolloff'] = rolloff
    
    # Generate title
    filename = Path(input_path).name
    feature_str = ""
    if features:
        feature_names = list(features.keys())
        feature_str = f" + {', '.join(feature_names)}"
    
    title = f"{y_label} - {filename}\n(n_fft={n_fft}, hop={hop_length}, window={window}){feature_str}"
    
    # Plot
    plot_spectrogram(
        spec_db,
        sr,
        hop_length,
        title,
        use_mel=use_mel,
        fmax=fmax,
        features=features if features else None,
        output_path=output_path,
        display=display
    )


def main():
    parser = argparse.ArgumentParser(
        description='Audio spectral analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('input', help='Input audio file (WAV, MP3, FLAC, etc.)')
    
    # STFT parameters
    parser.add_argument('--n-fft', type=int, default=2048,
                       help='FFT window size (default: 2048)')
    parser.add_argument('--hop-length', type=int, default=None,
                       help='Hop length in samples (default: n_fft/4)')
    parser.add_argument('--window', default='hann',
                       choices=['hann', 'hamming', 'blackman', 'blackmanharris'],
                       help='Window function (default: hann)')
    
    # Processing options
    parser.add_argument('--silence-threshold', type=float, default=60,
                       help='Silence removal threshold in dB (default: 60, use 0 to disable)')
    parser.add_argument('--mel', action='store_true',
                       help='Use Mel-scale spectrogram')
    parser.add_argument('--fmax', type=float, default=None,
                       help='Maximum frequency to display (Hz)')
    
    # Spectral features
    parser.add_argument('--centroid', action='store_true',
                       help='Show spectral centroid')
    parser.add_argument('--bandwidth', action='store_true',
                       help='Show spectral bandwidth')
    parser.add_argument('--rolloff', action='store_true',
                       help='Show spectral rolloff (85%)')
    
    # Output options
    parser.add_argument('-o', '--output', default=None,
                       help='Save plot to file (PNG, PDF, etc.)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plot interactively')
    
    args = parser.parse_args()
    
    # Validation
    if args.n_fft < 64:
        parser.error('n_fft must be >= 64')
    
    if args.silence_threshold == 0:
        silence_threshold = None
    else:
        silence_threshold = args.silence_threshold
    
    # Run analysis
    analyze_audio(
        args.input,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        window=args.window,
        silence_threshold=silence_threshold,
        use_mel=args.mel,
        fmax=args.fmax,
        show_centroid=args.centroid,
        show_bandwidth=args.bandwidth,
        show_rolloff=args.rolloff,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == '__main__':
    main()