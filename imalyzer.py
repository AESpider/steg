#!/usr/bin/env python3
"""
imalyzer - Image Analysis Tool

Detects printer tracking dots (yellow dots) and hidden steganographic data.
Supports channel extraction and LSB analysis.

Examples:
    # Detect yellow tracking dots
    python imalyzer.py -i scan.png -o output.png --yellow-dots --scale 8
    
    # Extract LSB from blue channel
    python imalyzer.py -i image.png -o qr.png --lsb-extract -c b
    
    # Isolate red channel with contrast adjustment
    python imalyzer.py -i photo.jpg -o red.png -c r --interactive
"""

import argparse
import sys
import random
from typing import Optional, List

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def load_image(path: str) -> Image.Image:
    """Load and convert image to RGB."""
    try:
        return Image.open(path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)


def extract_channel(img: Image.Image, channel: str) -> Image.Image:
    """
    Extract specific color channel(s).
    
    Args:
        img: RGB image
        channel: 'r', 'g', 'b', 'y' (yellow from CMYK), or comma-separated combo
    
    Returns:
        Grayscale or RGB image with selected channels
    """
    channels = [c.strip().lower() for c in channel.split(',') if c.strip()]
    
    # Yellow channel (CMYK)
    if 'y' in channels:
        cmyk = img.convert('CMYK')
        _, _, yellow, _ = cmyk.split()
        return yellow.convert('L')
    
    # RGB channel(s)
    rgb_map = {'r': 0, 'g': 1, 'b': 2}
    valid_channels = [c for c in channels if c in rgb_map]
    
    if not valid_channels:
        return img.convert('L')  # Default to grayscale
    
    # Single channel
    if len(valid_channels) == 1:
        idx = rgb_map[valid_channels[0]]
        return img.split()[idx].convert('L')
    
    # Multiple channels - create RGB with selected channels
    r, g, b = img.split()
    black = Image.new('L', img.size, 0)
    
    return Image.merge('RGB', (
        r if 'r' in valid_channels else black,
        g if 'g' in valid_channels else black,
        b if 'b' in valid_channels else black
    ))


def normalize_contrast(img: Image.Image) -> Image.Image:
    """Stretch contrast to full 0-255 range (grayscale only)."""
    if img.mode != 'L':
        return img
    
    arr = np.array(img)
    min_val, max_val = arr.min(), arr.max()
    
    if max_val == min_val:
        print("Warning: Uniform channel, no variation detected", file=sys.stderr)
        return Image.fromarray(np.zeros_like(arr), mode='L')
    
    normalized = ((arr - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return Image.fromarray(normalized, mode='L')


def apply_yellow_dots_visualization(img: Image.Image) -> Image.Image:
    """Convert grayscale to yellow visualization (R+G channels, B=0)."""
    if img.mode != 'L':
        return img
    
    arr = np.array(img)
    yellow_rgb = Image.merge('RGB', (
        Image.fromarray(arr),      # Red
        Image.fromarray(arr),      # Green
        Image.fromarray(np.zeros_like(arr))  # Blue = 0
    ))
    return yellow_rgb


def extract_lsb(img: Image.Image, channels: Optional[List[str]] = None) -> Image.Image:
    """
    Extract Least Significant Bit from specified channels.
    
    Args:
        img: RGB image
        channels: List of channels ('r', 'g', 'b'). Defaults to ['b']
    
    Returns:
        Binary image (mode '1')
    """
    if channels is None:
        channels = ['b']
    
    arr = np.array(img)
    h, w = arr.shape[:2]
    
    # Combine LSBs from all specified channels using OR
    lsb_combined = np.zeros((h, w), dtype=np.uint8)
    
    channel_map = {'r': 0, 'g': 1, 'b': 2}
    for ch in channels:
        if ch in channel_map:
            idx = channel_map[ch]
            lsb_combined |= (arr[:, :, idx] & 1)
    
    # Convert to binary image (0 or 255)
    binary = (lsb_combined * 255).astype(np.uint8)
    return Image.fromarray(binary, mode='L').convert('1')


def apply_random_colormap(img: Image.Image) -> Image.Image:
    """Map each grayscale intensity to a random RGB color."""
    if img.mode != 'L':
        img = img.convert('L')
    
    arr = np.array(img)
    
    # Create random color lookup table
    colormap = np.array([
        [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        for _ in range(256)
    ], dtype=np.uint8)
    
    colored = colormap[arr]
    return Image.fromarray(colored, 'RGB')


def scale_image(img: Image.Image, factor: int) -> Image.Image:
    """Resize image by integer factor using nearest-neighbor."""
    if factor <= 1:
        return img
    
    w, h = img.size
    new_size = (w * factor, h * factor)
    return img.resize(new_size, Image.NEAREST)


# Interactive Adjustment (OpenCV)

def interactive_adjust(img: Image.Image) -> Image.Image:
    """Open window with brightness/contrast sliders."""
    if not HAS_OPENCV:
        print("Error: OpenCV not installed", file=sys.stderr)
        return img
    
    # Convert PIL to OpenCV format (BGR)
    if img.mode == 'L':
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
    else:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    window = 'Adjust (press any key to save)'
    cv2.namedWindow(window)
    
    def update(val):
        alpha = cv2.getTrackbarPos('Contrast', window) / 100.0
        beta = cv2.getTrackbarPos('Brightness', window) - 100
        adjusted = cv2.convertScaleAbs(cv_img, alpha=alpha, beta=beta)
        cv2.imshow(window, adjusted)
    
    cv2.createTrackbar('Contrast', window, 100, 200, update)
    cv2.createTrackbar('Brightness', window, 100, 200, update)
    
    update(0)
    cv2.waitKey(0)
    
    # Get final values
    alpha = cv2.getTrackbarPos('Contrast', window) / 100.0
    beta = cv2.getTrackbarPos('Brightness', window) - 100
    
    cv2.destroyAllWindows()
    
    final = cv2.convertScaleAbs(cv_img, alpha=alpha, beta=beta)
    
    # Convert back to PIL
    if img.mode == 'L':
        return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY))
    else:
        return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))


# Main

def process_image(
    input_path: str,
    output_path: str,
    channel: str = '',
    scale: int = 1,
    invert: bool = False,
    yellow_dots: bool = False,
    lsb_extract: bool = False,
    interactive: bool = False,
    colormap: bool = False
) -> None:
    """Main image processing pipeline."""
    
    # Load image
    img = load_image(input_path)
    print(f"Processing: {input_path} ({img.size[0]}x{img.size[1]})")
    
    # Mode selection
    if lsb_extract:
        # LSB extraction mode
        channels = [c.strip().lower() for c in channel.split(',') if c.strip()] if channel else None
        result = extract_lsb(img, channels)
        print(f"LSB extraction from channels: {channels or ['b']}")
    
    elif yellow_dots:
        # Yellow dots detection mode
        result = extract_channel(img, 'y')
        result = normalize_contrast(result)
        result = apply_yellow_dots_visualization(result)
        print("Yellow dots mode: CMYK yellow channel -> yellow RGB")
    
    else:
        # Standard channel extraction mode
        result = extract_channel(img, channel or '')
        
        # Normalize contrast for grayscale
        if result.mode == 'L':
            result = normalize_contrast(result)
    
    # Post-processing
    if invert:
        result = ImageOps.invert(result)
        print("Applied color inversion")
    
    if scale > 1:
        result = scale_image(result, scale)
        print(f"Scaled {scale}x")
    
    if interactive:
        result = interactive_adjust(result)
        print("Interactive adjustment completed")
    
    if colormap:
        result = apply_random_colormap(result)
        print("Applied random colormap")
    
    # Save
    result.save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Image forensics tool for tracking dots and steganography',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input image path')
    parser.add_argument('-o', '--output', required=True,
                        help='Output image path')
    
    # Processing modes
    parser.add_argument('-c', '--channel', default='',
                        help='Channel(s) to extract: r, g, b, y, or r,g,b')
    parser.add_argument('--yellow-dots', action='store_true',
                        help='Yellow dots detection mode (CMYK yellow)')
    parser.add_argument('--lsb-extract', action='store_true',
                        help='Extract LSB for steganography')
    
    # Enhancements
    parser.add_argument('-s', '--scale', type=int, default=1,
                        help='Scale factor (default: 1)')
    parser.add_argument('-v', '--invert', action='store_true',
                        help='Invert colors')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive brightness/contrast (requires OpenCV)')
    parser.add_argument('--colormap', action='store_true',
                        help='Apply random color mapping')
    
    args = parser.parse_args()
    
    # Validation
    if args.scale < 1:
        parser.error('Scale must be >= 1')
    
    if args.interactive and not HAS_OPENCV:
        parser.error('Interactive mode requires OpenCV: pip install opencv-python')
    
    if args.yellow_dots and args.lsb_extract:
        parser.error('--yellow-dots and --lsb-extract are mutually exclusive')
    
    # Run
    process_image(
        args.input,
        args.output,
        args.channel,
        args.scale,
        args.invert,
        args.yellow_dots,
        args.lsb_extract,
        args.interactive,
        args.colormap
    )


if __name__ == '__main__':
    main()