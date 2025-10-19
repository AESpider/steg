#!/usr/bin/env python3
"""
yEnc Encoder/Decoder
"""
import sys
import argparse


def yenc_encode(data):
    """Encode bytes using yEnc"""
    result = bytearray()
    for byte in data:
        encoded = (byte + 42) & 0xff
        # Escape special chars
        if encoded in (0x00, 0x0a, 0x0d, 0x3d):
            result.append(0x3d)
            result.append((encoded + 64) & 0xff)
        else:
            result.append(encoded)
    return bytes(result)


def yenc_decode(data):
    """Decode yEnc bytes"""
    result = bytearray()
    i = 0
    while i < len(data):
        if data[i] == 0x3d:  # escape char
            i += 1
            if i < len(data):
                result.append((data[i] - 64 - 42) & 0xff)
        else:
            result.append((data[i] - 42) & 0xff)
        i += 1
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description="yEnc encoder/decoder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    # Encode
    enc = sub.add_parser("encode")
    enc.add_argument("-i", "--input")
    enc.add_argument("-m", "--message")
    enc.add_argument("-o", "--output")
    
    # Decode
    dec = sub.add_parser("decode")
    dec.add_argument("-i", "--input")
    dec.add_argument("-m", "--message")
    dec.add_argument("-o", "--output")
    
    args = parser.parse_args()
    
    # Get input
    if args.input:
        with open(args.input, "rb") as f:
            data = f.read()
        # For decode: check if hex format
        if args.cmd == "decode":
            try:
                text = data.decode('ascii').strip()
                if all(c in '0123456789abcdefABCDEF \n\r\t' for c in text):
                    data = bytes.fromhex(text.replace('\n', '').replace(' ', ''))
            except:
                pass
    elif args.message:
        if args.cmd == "encode":
            data = args.message.encode('utf-8')
        else:  # decode
            data = bytes.fromhex(args.message)
    else:
        print("Error: use -i FILE or -m MESSAGE", file=sys.stderr)
        return 1
    
    # Process
    result = yenc_encode(data) if args.cmd == "encode" else yenc_decode(data)
    
    # Output
    if args.output:
        with open(args.output, "wb") as f:
            f.write(result)
        print(f"Written to: {args.output}", file=sys.stderr)
    else:
        if args.cmd == "encode":
            # Encode output: always hex
            print(result.hex())
        else:
            # Decode output: try text, else hex
            try:
                sys.stdout.buffer.write(result)
            except:
                print(result.hex())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())