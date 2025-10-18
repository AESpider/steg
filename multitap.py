#!/usr/bin/env python3
"""
T9 Multi-Tap Encoder/Decoder

Encode and decode text using old phone keypad multi-tap method.

Examples:
    # Encode text
    python3 multitap.py encode "hello"
    # Output: 44-33-555-555-666
    
    # Encode with space
    python3 multitap.py encode "hello world"
    # Output: 44-33-555-555-666--9-666-777-555-3
    
    # Decode
    python3 multitap.py decode "44-33-555-555-666"
    # Output: hello
"""

import sys
import argparse
from pathlib import Path
from typing import Optional


# T9 Keypad Mapping

# Map each key to its letters
KEYPAD = {
    '2': 'abc',
    '3': 'def',
    '4': 'ghi',
    '5': 'jkl',
    '6': 'mno',
    '7': 'pqrs',
    '8': 'tuv',
    '9': 'wxyz',
}

# Build reverse mapping: letter -> (key, position)
LETTER_MAP = {}
for key, letters in KEYPAD.items():
    for pos, letter in enumerate(letters, 1):
        LETTER_MAP[letter] = (key, pos)


# Encoding

def encode_char(char: str) -> str:
    """Encode single character to multi-tap sequence."""
    if char not in LETTER_MAP:
        raise ValueError(f"Unsupported character: '{char}'")
    
    key, count = LETTER_MAP[char]
    return key * count


def encode(text: str, separator: str = '-') -> str:
    """
    Encode text to multi-tap sequence.
    
    Spaces are encoded as double separator (e.g., '--').
    """
    parts = []
    
    for char in text.lower():
        if char == ' ':
            # Empty string will create double separator when joined
            parts.append('')
        else:
            parts.append(encode_char(char))
    
    return separator.join(parts)


# Decoding

def decode_group(group: str) -> str:
    """Decode a single multi-tap group (e.g., '444' -> 'i')."""
    if not group:
        return ''
    
    # All characters must be the same digit
    key = group[0]
    if not all(c == key for c in group):
        raise ValueError(f"Invalid group (mixed digits): '{group}'")
    
    if key not in KEYPAD:
        raise ValueError(f"Invalid key: '{key}'")
    
    # Get letter at position (count - 1)
    letters = KEYPAD[key]
    count = len(group)
    
    if count > len(letters):
        raise ValueError(
            f"Too many presses for key '{key}': {count} (max {len(letters)})"
        )
    
    return letters[count - 1]


def decode(sequence: str) -> str:
    """
    Decode multi-tap sequence to text.
    
    Double separator represents space.
    """
    normalized = sequence.strip()
    
    # Detect separator (non-digit character)
    separator = None
    for char in normalized:
        if not char.isdigit():
            separator = char
            break
    
    if separator is None:
        # No separator, treat as single group
        return decode_group(normalized)
    
    # Split by separator first
    groups = normalized.split(separator)
    
    # Decode each group, treating empty strings as spaces
    chars = []
    for group in groups:
        if not group:  # Empty group = space (from double separator)
            chars.append(' ')
        else:
            chars.append(decode_group(group))
    
    return ''.join(chars)


# File I/O

def read_input(path: str) -> str:
    """Read input from file or stdin."""
    if path == '-':
        return sys.stdin.read().strip()
    
    try:
        return Path(path).read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Error: Invalid UTF-8 encoding: {path}", file=sys.stderr)
        sys.exit(1)


def write_output(content: str, path: Optional[str]) -> None:
    """Write output to file or stdout."""
    if path:
        try:
            Path(path).write_text(content + '\n', encoding='utf-8')
            print(f"Saved: {path}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(content)


def main():
    parser = argparse.ArgumentParser(
        description='T9 multi-tap encoder/decoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True,
                                       help='Operation mode')
    
    # Encode command
    encode_parser = subparsers.add_parser('encode',
                                          help='Encode text to multi-tap')
    encode_parser.add_argument('text', nargs='?',
                              help='Text to encode (or use -i for file)')
    encode_parser.add_argument('-i', '--input',
                              help='Input file (use - for stdin)')
    encode_parser.add_argument('-o', '--output',
                              help='Output file (default: stdout)')
    encode_parser.add_argument('-s', '--separator', default='-',
                              help='Separator character (default: -)')
    
    # Decode command
    decode_parser = subparsers.add_parser('decode',
                                          help='Decode multi-tap to text')
    decode_parser.add_argument('sequence', nargs='?',
                              help='Sequence to decode (or use -i for file)')
    decode_parser.add_argument('-i', '--input',
                              help='Input file (use - for stdin)')
    decode_parser.add_argument('-o', '--output',
                              help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    try:
        # Get input
        if args.command == 'encode':
            if args.input:
                input_data = read_input(args.input)
            elif args.text:
                input_data = args.text
            else:
                parser.error('Provide text or use -i for file input')
            
            result = encode(input_data, args.separator)
        
        else:  # decode
            if args.input:
                input_data = read_input(args.input)
            elif args.sequence:
                input_data = args.sequence
            else:
                parser.error('Provide sequence or use -i for file input')
            
            result = decode(input_data)
        
        # Output result
        write_output(result, args.output)
    
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()