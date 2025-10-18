#!/usr/bin/env python3
"""
APNG Steganography Tool

Hide and extract messages in APNG frame delay metadata.
Messages are encoded in the delay_num (numerator) values of each frame.

Examples:
    # Decode message from APNG
    python3 apng_steg.py decode animation.apng
    
    # Encode message into APNG
    python3 apng_steg.py encode frames/ "SECRET" -o hidden.apng
"""

import sys
import argparse
import subprocess
import glob
import re
import shutil
from pathlib import Path
from typing import List, Optional

try:
    from apng import APNG
except ImportError:
    print("Error: apng library not found", file=sys.stderr)
    print("Install: pip install apng", file=sys.stderr)
    sys.exit(1)


# APNG Frame Extraction (uses apngdis)

def extract_frames(apng_path: str, output_dir: str, prefix: str = 'frame') -> None:
    """
    Extract frames from APNG using apngdis tool.
    Creates frame PNG files and delay metadata TXT files.
    """
    print(f"Extracting frames from: {apng_path}")
    
    # Clean output directory
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True)
    
    # Run apngdis in temporary directory
    abs_path = Path(apng_path).resolve()
    
    try:
        subprocess.run(
            ['apngdis', str(abs_path), prefix],
            capture_output=True,
            text=True,
            check=True
        )
    except FileNotFoundError:
        print("Error: 'apngdis' not found", file=sys.stderr)
        print("Install: sudo apt-get install apngdis", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running apngdis: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    
    # Move extracted files to output directory
    moved = 0
    for ext in ('png', 'txt'):
        for file in glob.glob(f"{prefix}*.{ext}"):
            shutil.move(file, Path(output_dir) / file)
            moved += 1
    
    print(f"Extracted {moved} files to: {output_dir}")


# Delay Metadata Parsing

def parse_delay_files(directory: str, prefix: str = 'frame') -> List[int]:
    """
    Parse delay metadata files and extract delay numerators.
    
    Returns list of delay_num values in frame order.
    """
    pattern = str(Path(directory) / f"{prefix}*.txt")
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(re.search(rf"{prefix}(\d+)", Path(f).name).group(1))
    )
    
    if not files:
        print(f"Error: No delay files found in {directory}", file=sys.stderr)
        return []
    
    delay_nums = []
    for filepath in files:
        with open(filepath, 'r') as f:
            line = f.read().strip()
        
        # Parse "delay=NUM/DEN" format
        match = re.match(r"delay=(\d+)/(\d+)", line)
        if not match:
            print(f"Warning: Invalid delay format in {filepath}: {line}", 
                  file=sys.stderr)
            continue
        
        delay_nums.append(int(match.group(1)))
    
    return delay_nums


# Message Encoding/Decoding

def decode_message(delay_nums: List[int], 
                   min_ascii: int = 32, 
                   max_ascii: int = 126) -> str:
    """
    Decode message from delay numerators.
    Only includes printable ASCII characters (32-126).
    """
    chars = []
    for num in delay_nums:
        if min_ascii <= num <= max_ascii:
            chars.append(chr(num))
    
    return ''.join(chars)


def encode_message_to_delays(message: str) -> List[int]:
    """Convert message string to list of ASCII values."""
    ascii_vals = [ord(char) for char in message]
    
    # Warn about non-printable characters
    if any(val < 32 or val > 126 for val in ascii_vals):
        print("Warning: Message contains non-printable characters", 
              file=sys.stderr)
    
    return ascii_vals


# APNG Creation

def create_apng(
    frames_dir: str,
    delay_nums: List[int],
    output_path: str,
    delay_den: int = 10
) -> None:
    """
    Create APNG with message encoded in frame delays.
    
    Args:
        frames_dir: Directory containing frame PNG files
        delay_nums: List of delay numerators (ASCII values)
        output_path: Output APNG file path
        delay_den: Delay denominator (default: 10)
    """
    # Get frame files
    frames = sorted(glob.glob(str(Path(frames_dir) / '*.png')))
    
    if not frames:
        print(f"Error: No PNG files in {frames_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Creating APNG with {len(frames)} frames")
    
    # Create APNG object
    apng = APNG()
    
    # Add frames with encoded delays
    for i, frame_path in enumerate(frames):
        # Loop message if more frames than characters
        delay_num = delay_nums[i % len(delay_nums)]
        
        try:
            apng.append_file(frame_path, delay=delay_num, delay_den=delay_den)
        except Exception as e:
            print(f"Error adding frame {frame_path}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Save APNG
    try:
        apng.save(output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error saving APNG: {e}", file=sys.stderr)
        sys.exit(1)


def decode_apng(
    apng_path: str,
    output_dir: str = 'apng_frames',
    output_file: Optional[str] = None
) -> None:
    """Extract and decode message from APNG."""
    # Extract frames
    extract_frames(apng_path, output_dir)
    
    # Parse delays
    delay_nums = parse_delay_files(output_dir)
    
    if not delay_nums:
        print("Error: No delay data found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Delay numerators: {delay_nums}")
    
    # Decode message
    message = decode_message(delay_nums)
    
    if not message:
        print("Warning: No printable characters found", file=sys.stderr)
        return
    
    print(f"Hidden message: {message}")
    
    # Save to file if requested
    if output_file:
        try:
            Path(output_file).write_text(message)
            print(f"Message saved: {output_file}")
        except Exception as e:
            print(f"Error saving message: {e}", file=sys.stderr)


def encode_apng(
    frames_dir: str,
    message: str,
    output_path: str = 'hidden.apng',
    delay_den: int = 10
) -> None:
    """Create APNG with hidden message in frame delays."""
    if not Path(frames_dir).is_dir():
        print(f"Error: Directory not found: {frames_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Convert message to delays
    delay_nums = encode_message_to_delays(message)
    
    print(f"Encoding message: {message}")
    print(f"ASCII values: {delay_nums}")
    
    # Create APNG
    create_apng(frames_dir, delay_nums, output_path, delay_den)


def main():
    parser = argparse.ArgumentParser(
        description='APNG steganography tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True,
                                       help='Operation mode')
    
    # Decode command
    decode_parser = subparsers.add_parser('decode',
                                          help='Extract hidden message from APNG')
    decode_parser.add_argument('apng', help='Input APNG file')
    decode_parser.add_argument('--out-dir', default='apng_frames',
                              help='Frame extraction directory (default: apng_frames)')
    decode_parser.add_argument('-o', '--output',
                              help='Save message to file')
    
    # Encode command
    encode_parser = subparsers.add_parser('encode',
                                          help='Hide message in APNG')
    encode_parser.add_argument('frames_dir',
                              help='Directory containing frame PNG files')
    encode_parser.add_argument('message',
                              help='Message to hide')
    encode_parser.add_argument('-o', '--output', default='output.apng',
                              help='Output APNG file (default: output.apng)')
    encode_parser.add_argument('--delay-den', type=int, default=10,
                              help='Delay denominator (default: 10)')
    
    args = parser.parse_args()
    
    # Execute command
    try:
        if args.command == 'decode':
            decode_apng(args.apng, args.out_dir, args.output)
        
        elif args.command == 'encode':
            encode_apng(args.frames_dir, args.message, 
                       args.output, args.delay_den)
    
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()