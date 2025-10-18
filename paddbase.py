#!/usr/bin/env python3
"""
paddbase - Padding Decoder

Extracts hidden data from base32/base64 encoded strings using padding bits.
The technique exploits unused bits in the last character before padding.

Principle:
  - Base64: Each char = 6 bits, but encodes 8-bit bytes
  - Padding '=' indicates unused bits in the last character
  - One '=' = 2 hidden bits, two '==' = 4 hidden bits
  - Base32: Similar principle with 5 bits per character

Examples:
    # Decode base64 from file
    python3 paddbase.py -i encoded.txt -o decoded.bin
    
    # Decode base32 with custom separator
    python3 paddbase.py -i data.txt -o output.bin -e base32 -s ","
    
    # Decode from stdin
    python3 paddbase.py -i data.txt | python3 paddbase.py -e base32


Reference: https://inshallhack.org/paddbasey/
"""

import argparse
import sys
import math
from typing import Optional
from pathlib import Path


BASE64_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
BASE32_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

# Padding Functions

def gcd(a: int, b: int) -> int:
    """Calculate Greatest Common Divisor."""
    while b > 0:
        a, b = b, a % b
    return a


def extract_hidden_bits(token: bytes, charset: str, pad_char: str = "=", n_pad: int = 8) -> Optional[bytes]:
    """
    Extract hidden bits from a single base-encoded token.
    
    Args:
        token: Base32/64 encoded token
        charset: Character set used for encoding
        pad_char: Padding character (default '=')
        n_pad: Number of bits per padding unit (default 8)
    
    Returns:
        Hidden bits as bytes, or None if no hidden data
    """
    token_str = token.decode('utf-8', errors='ignore').strip()
    
    # Count padding characters
    padding_count = token_str.count(pad_char)
    if not token_str or padding_count == 0:
        return None
    
    # Calculate encoding parameters
    n_repr = int(math.ceil(math.log(len(charset), 2)))  # Bits per character
    w_len = n_repr * n_pad / gcd(n_repr, n_pad)         # Quantum length
    n_char = int(math.ceil(w_len / n_repr))             # Characters per quantum
    
    # Validate token structure
    if len(token_str) % n_char != 0:
        return None
    
    # Calculate unused bits for each padding position
    unused_bits = {n: int(w_len - n * n_repr) % n_pad for n in range(n_char)}
    
    # Get the last character before padding
    last_char = token_str.rstrip(pad_char)[-1]
    
    try:
        char_index = charset.index(last_char)
    except ValueError:
        return None
    
    # Extract binary value of last character
    binary_value = bin(char_index)[2:].zfill(n_repr)
    
    # Return only the hidden bits (rightmost unused bits)
    hidden_bit_count = unused_bits[padding_count]
    if hidden_bit_count == 0:
        return None
    
    return binary_value[-hidden_bit_count:].encode('utf-8')


def decode_paddbase(
    encoded_data: bytes,
    encoding: str = "base64",
    charset: Optional[str] = None,
    separator: str = "\n",
    pad_char: str = "=",
    n_pad: int = 8
) -> bytes:
    """
    Decode hidden data from base-encoded strings.
    
    Args:
        encoded_data: Base32/64 encoded data with hidden bits
        encoding: 'base32' or 'base64'
        charset: Custom character set (optional)
        separator: Token separator (default newline)
        pad_char: Padding character (default '=')
        n_pad: Bits per padding unit (default 8)
    
    Returns:
        Decoded hidden data as bytes
    """
    # Select charset
    if charset is None:
        charset = BASE64_CHARSET if encoding == "base64" else BASE32_CHARSET
    
    # Extract hidden bits from each token
    all_bits = []
    sep_bytes = separator.encode('utf-8')
    
    for token in encoded_data.split(sep_bytes):
        token = token.strip()
        if not token:
            continue
        
        hidden_bits = extract_hidden_bits(token, charset, pad_char, n_pad)
        if hidden_bits:
            all_bits.append(hidden_bits)
    
    # Concatenate all bits
    bit_string = b''.join(all_bits).decode('utf-8')
    
    # Convert bit string to bytes (8 bits per byte)
    result = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_bits = bit_string[i:i+8]
        if len(byte_bits) == 8:
            result.append(int(byte_bits, 2))
    
    return bytes(result)


# I/O Functions

def read_input(input_path: Optional[str]) -> bytes:
    """Read data from file or stdin."""
    if input_path:
        path = Path(input_path)
        if not path.exists():
            print(f"Error: File not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        
        with path.open('rb') as f:
            return f.read()
    else:
        # Read from stdin
        return sys.stdin.buffer.read()


def write_output(data: bytes, output_path: Optional[str]) -> None:
    """Write data to file or stdout."""
    if output_path:
        path = Path(output_path)
        with path.open('wb') as f:
            f.write(data)
        print(f"Decoded data saved to: {output_path}", file=sys.stderr)
    else:
        # Write to stdout
        sys.stdout.buffer.write(data)


def main():
    parser = argparse.ArgumentParser(
        description='Decode hidden data from base32/base64 padding bits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decode from file
  python paddbase.py -i encoded.txt -o decoded.bin
  
  # Decode base32 from stdin
  python3 paddbase.py -i data.txt | python3 paddbase.py -e base32
  
  # Custom separator
  python paddbase.py -i data.txt -o out.bin -s "," -e base64

For more info: https://inshallhack.org/paddbasey/
        """
    )
    
    # I/O arguments
    parser.add_argument('-i', '--input', 
                        help='Input file (default: stdin)')
    parser.add_argument('-o', '--output',
                        help='Output file (default: stdout)')
    
    # Decoding parameters
    parser.add_argument('-e', '--encoding', 
                        choices=['base32', 'base64'], 
                        default='base64',
                        help='Encoding type (default: base64)')
    parser.add_argument('-c', '--charset',
                        help='Custom character set')
    parser.add_argument('-s', '--separator',
                        default='\n',
                        help='Token separator (default: newline)')
    parser.add_argument('-p', '--pad-char',
                        default='=',
                        help='Padding character (default: =)')
    
    # Verbose mode
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Read input data
    if args.verbose:
        print(f"Reading from: {args.input or 'stdin'}...", file=sys.stderr)
    
    encoded_data = read_input(args.input)
    
    if not encoded_data:
        print("Error: No input data", file=sys.stderr)
        sys.exit(1)
    
    # Decode hidden data
    if args.verbose:
        print(f"Decoding {args.encoding} with separator: {repr(args.separator)}...", file=sys.stderr)
    
    try:
        decoded = decode_paddbase(
            encoded_data,
            args.encoding,
            args.charset,
            args.separator,
            args.pad_char
        )
    except Exception as e:
        print(f"Error during decoding: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not decoded:
        print("Warning: No hidden data found", file=sys.stderr)
        sys.exit(0)
    
    # Write output
    if args.verbose:
        print(f"Extracted {len(decoded)} bytes", file=sys.stderr)
    
    write_output(decoded, args.output)


if __name__ == '__main__':
    main()