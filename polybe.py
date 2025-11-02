#!/usr/bin/env python3
"""
Polybe Cipher Encoder/Decoder

A command-line tool for encoding and decoding text using the Polybe square cipher.
The Polybe cipher uses a 5x5 grid to represent letters as coordinate pairs.

Principle:
  - Each letter maps to a coordinate (row + column)
  - Example: 't' -> A1, 'e' -> A3, 'x' -> B5

Examples:
    # Decode from file to stdout
    python3 polybe.py decode -i code_polybe.txt
    
    # Encode from stdin to file
    echo "hello world" | python3 polybe.py encode -o encoded.txt
    
    # Decode with input and output files
    python3 polybe.py decode -i encoded.txt -o decoded.txt
"""

import argparse
import re
import sys

# Polybe matrix (customizable)
MATRIX = {
    'A1':'a', 'A2':'b', 'A3':'c', 'A4':'d', 'A5':'e',
    'B1':'f', 'B2':'g', 'B3':'h', 'B4':'i', 'B5':'k',
    'C1':'l', 'C2':'m', 'C3':'n', 'C4':'o', 'C5':'p',
    'D1':'q', 'D2':'r', 'D3':'s', 'D4':'t', 'D5':'u',
    'E1':'v', 'E2':'w', 'E3':'x', 'E4':'y', 'E5':'z'
}

# Build reverse matrix for encoding
REVERSE_MATRIX = {v: k for k, v in MATRIX.items()}
# Row mapping for lowercase input
ROW_MAP = {'a':'A', 'b':'B', 'c':'C', 'd':'D', 'e':'E'}

def encode_text(text):
    """Encode plaintext to Polybe cipher"""
    result = []
    for line in text.strip().split('\n'):
        encoded_words = []
        for word in line.split():
            encoded_chars = []
            for char in word.lower():
                if char in REVERSE_MATRIX:
                    encoded_chars.append(REVERSE_MATRIX[char].lower())
                else:
                    # Keep non-alphabetic chars as-is
                    encoded_chars.append(char)
            encoded_words.append(''.join(encoded_chars))
        result.append(' '.join(encoded_words))
    return '\n'.join(result)

def decode_text(text):
    """Decode Polybe cipher to plaintext"""
    result = []
    for line in text.strip().split('\n'):
        decoded_words = []
        for word in line.split():
            # Extract all valid Polybe pairs (e.g., a1, b3)
            pairs = re.findall(r'[a-e][1-5]', word.lower())
            decoded_chars = []
            for pair in pairs:
                row = ROW_MAP[pair[0]]
                key = row + pair[1]
                decoded_chars.append(MATRIX[key])
            decoded_words.append(''.join(decoded_chars))
        result.append(' '.join(decoded_words))
    return '\n'.join(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Polybe cipher encoder/decoder tool'
    )
    parser.add_argument(
        'mode',
        choices=['encode', 'decode'],
        help='Operation mode: encode or decode'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input file path (default: stdin)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (default: stdout)'
    )
    
    args = parser.parse_args()
    
    # Read input
    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.input}' not found", file=sys.stderr)
            sys.exit(1)
    else:
        text = sys.stdin.read()
    
    # Process
    if args.mode == 'encode':
        output = encode_text(text)
    else:
        output = decode_text(text)
    
    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
    else:
        print(output)
