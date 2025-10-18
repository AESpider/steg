#!/usr/bin/env python3
"""
Whitespace Steganography and Whitespace Language Tool

Hide secret messages in whitespace (spaces/tabs) or execute Whitespace programs.

Modes:
    hide    : Hide text message in whitespace (EOF or mid-text encoding)
    reveal  : Extract hidden text message from whitespace
    exec    : Execute Whitespace esoteric language program

Examples:
    # Hide message in trailing spaces
    python3 white_space.py hide -i cover.txt -s "Secret" -o hidden.txt
    
    # Reveal hidden message
    python3 white_space.py reveal -i hidden.txt
    
    # Execute Whitespace program embedded in file
    python3 white_space.py exec -i program.ws
"""

import sys
import argparse
import tempfile
import subprocess
import os
from typing import List, Optional


# Constants

BIT_TO_WS = {'0': ' ', '1': '\t'}
WS_TO_BIT = {' ': '0', '\t': '1'}


# File I/O

def read_lines(path: str) -> List[str]:
    """Read file and return lines."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)


def write_file(path: str, content: str) -> None:
    """Write content to file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Error writing file: {e}", file=sys.stderr)
        sys.exit(1)


# Binary Encoding/Decoding

def text_to_bits(text: str) -> str:
    """Convert text to binary string (8 bits per character)."""
    return ''.join(f"{ord(c):08b}" for c in text)


def bits_to_text(bits: str) -> str:
    """Convert binary string to text (pad to multiple of 8)."""
    # Pad to multiple of 8
    if len(bits) % 8:
        bits += '0' * (8 - len(bits) % 8)
    
    # Convert each 8-bit chunk to character
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        chars.append(chr(int(byte, 2)))
    
    return ''.join(chars)


# EOF (End-of-Line)

def hide_eof(lines: List[str], secret: str) -> str:
    """
    Hide message in trailing whitespace at end of lines.
    Each bit encoded as space (0) or tab (1).
    """
    bits = text_to_bits(secret)
    result = []
    bit_idx = 0
    
    # Encode bits in existing lines
    for line in lines:
        core = line.rstrip('\r\n')
        newline = line[len(core):]
        
        # Add whitespace bit if available
        ws = BIT_TO_WS[bits[bit_idx]] if bit_idx < len(bits) else ''
        result.append(core + ws + newline)
        
        if bit_idx < len(bits):
            bit_idx += 1
    
    # Add new lines if needed
    while bit_idx < len(bits):
        ws = BIT_TO_WS[bits[bit_idx]]
        result.append(ws + '\n')
        bit_idx += 1
    
    return ''.join(result)


def reveal_eof(lines: List[str]) -> str:
    """Extract message from trailing whitespace."""
    bits = []
    
    for line in lines:
        core = line.rstrip('\r\n')
        # Get trailing spaces/tabs
        trimmed = core.rstrip(' \t')
        trailing = core[len(trimmed):]
        
        # Convert whitespace to bits
        for char in trailing:
            if char in WS_TO_BIT:
                bits.append(WS_TO_BIT[char])
    
    return bits_to_text(''.join(bits))


# Mid-Text

def hide_mid_text(lines: List[str], secret: str) -> str:
    """
    Hide message in spaces between words.
    Single space = 0, double space = 1.
    
    Warning: Very fragile, easily broken by text editors.
    """
    bits = text_to_bits(secret)
    result = []
    bit_idx = 0
    
    for line in lines:
        newline = line[len(line.rstrip('\r\n')):]
        clean = line.rstrip('\r\n')
        
        # Skip empty lines
        if not clean.strip():
            result.append(line)
            continue
        
        words = clean.split(' ')
        parts = []
        
        for i, word in enumerate(words):
            parts.append(word)
            
            # Add space between words
            if i < len(words) - 1:
                if bit_idx < len(bits):
                    # Encode bit: 0=single space, 1=double space
                    parts.append(' ' if bits[bit_idx] == '0' else '  ')
                    bit_idx += 1
                else:
                    parts.append(' ')
        
        result.append(''.join(parts) + newline)
    
    if bit_idx < len(bits):
        print(f"Warning: Message truncated ({bit_idx}/{len(bits)} bits encoded)", 
              file=sys.stderr)
    
    return ''.join(result)


def reveal_mid_text(lines: List[str]) -> str:
    """Extract message from inter-word spaces."""
    bits = []
    
    for line in lines:
        i = 0
        while i < len(line):
            if line[i] == ' ':
                # Check if double space
                if i + 1 < len(line) and line[i + 1] == ' ':
                    bits.append('1')
                    i += 2
                else:
                    bits.append('0')
                    i += 1
            else:
                i += 1
    
    return bits_to_text(''.join(bits))


# Whitespace Language Execution

def extract_whitespace_code(lines: List[str]) -> str:
    """Extract only trailing whitespace from each line (Whitespace program)."""
    code_parts = []
    
    for line in lines:
        stripped = line.rstrip('\r\n')
        trimmed = stripped.rstrip(' \t')
        trailing = stripped[len(trimmed):]
        code_parts.append(trailing + '\n')
    
    return ''.join(code_parts)


def execute_whitespace(code: str) -> str:
    """Execute Whitespace esoteric language code."""
    # Write code to temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                     encoding='utf-8', suffix='.ws') as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    
    try:
        # Try multiple interpreters in order of preference
        interpreters = [
            ['whitespace', tmp_path],           # pip install whitespace
            ['whitepycli', tmp_path],           # pip install whitepy
            [sys.executable, '-m', 'whitespace', tmp_path]  # fallback
        ]
        
        for cmd in interpreters:
            try:
                output = subprocess.check_output(
                    cmd, 
                    stderr=subprocess.PIPE, 
                    timeout=5
                )
                return output.decode('utf-8', errors='ignore')
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                continue
        
        # All interpreters failed
        print("Error: No Whitespace interpreter found", file=sys.stderr)
        print("Install one of:", file=sys.stderr)
        print("  pip install whitespace", file=sys.stderr)
        print("  pip install whitepy", file=sys.stderr)
        sys.exit(1)
    
    except subprocess.TimeoutExpired:
        print("Error: Whitespace execution timeout (>5s)", file=sys.stderr)
        sys.exit(1)
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Main

def hide_message(
    input_path: str,
    secret: str,
    output_path: Optional[str],
    method: str = 'eof'
) -> None:
    """Hide secret message in cover text."""
    lines = read_lines(input_path)
    
    if method == 'eof':
        print(f"Hiding message in trailing whitespace ({len(secret)} chars)")
        result = hide_eof(lines, secret)
    else:  # mid-text
        print(f"Hiding message in inter-word spaces ({len(secret)} chars)")
        print("Warning: Mid-text encoding is fragile", file=sys.stderr)
        result = hide_mid_text(lines, secret)
    
    if output_path:
        write_file(output_path, result)
        print(f"Saved: {output_path}")
    else:
        print("\nEncoded content:")
        print(result)


def reveal_message(
    input_path: str,
    output_path: Optional[str],
    method: str = 'eof'
) -> None:
    """Reveal hidden message from file."""
    lines = read_lines(input_path)
    
    if method == 'eof':
        print("Extracting message from trailing whitespace")
        message = reveal_eof(lines)
    else:  # mid-text
        print("Extracting message from inter-word spaces")
        message = reveal_mid_text(lines)
    
    # Clean non-printable characters
    message = message.rstrip('\x00')
    
    if output_path:
        write_file(output_path, message)
        print(f"Saved: {output_path}")
    else:
        print("\nRevealed message:")
        print(message)


def execute_program(input_path: str, output_path: Optional[str]) -> None:
    """Execute Whitespace program embedded in file."""
    print(f"Extracting Whitespace code from: {input_path}")
    lines = read_lines(input_path)
    code = extract_whitespace_code(lines)
    
    print(f"Executing Whitespace program ({len(code)} chars)...")
    output = execute_whitespace(code)
    
    if output_path:
        write_file(output_path, output)
        print(f"Output saved: {output_path}")
    else:
        print("\nProgram output:")
        print(output)

def main():
    parser = argparse.ArgumentParser(
        description='Whitespace steganography and Whitespace language tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True,
                                       help='Operation mode')
    
    # Hide command
    hide_parser = subparsers.add_parser('hide', 
                                         help='Hide message in whitespace')
    hide_parser.add_argument('-i', '--input', required=True,
                            help='Cover text file')
    hide_parser.add_argument('-s', '--secret', required=True,
                            help='Secret message to hide')
    hide_parser.add_argument('-o', '--output',
                            help='Output file (default: stdout)')
    hide_parser.add_argument('-m', '--method', 
                            choices=['eof', 'mid-text'], default='eof',
                            help='Encoding method (default: eof)')
    
    # Reveal command
    reveal_parser = subparsers.add_parser('reveal',
                                          help='Reveal hidden message')
    reveal_parser.add_argument('-i', '--input', required=True,
                              help='File with hidden message')
    reveal_parser.add_argument('-o', '--output',
                              help='Output file (default: stdout)')
    reveal_parser.add_argument('-m', '--method',
                              choices=['eof', 'mid-text'], default='eof',
                              help='Decoding method (default: eof)')
    
    # Execute command
    exec_parser = subparsers.add_parser('exec',
                                        help='Execute Whitespace program')
    exec_parser.add_argument('-i', '--input', required=True,
                            help='File containing Whitespace code')
    exec_parser.add_argument('-o', '--output',
                            help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Dispatch commands
    try:
        if args.command == 'hide':
            hide_message(args.input, args.secret, args.output, args.method)
        
        elif args.command == 'reveal':
            reveal_message(args.input, args.output, args.method)
        
        elif args.command == 'exec':
            execute_program(args.input, args.output)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()