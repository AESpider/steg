#!/usr/bin/env python3
import subprocess
import sys
import re

def run_binwalk(filename):
    """Executes binwalk and returns the output."""
    print(f"[*] Analyzing {filename} with binwalk...")
    try:
        # Run binwalk and capture output
        result = subprocess.run(
            ['binwalk', filename],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing binwalk: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'binwalk' command not found or installed.")
        sys.exit(1)


def parse_binwalk_output(output):
    """Parses binwalk output for offsets and descriptions."""
    lines = output.strip().split('\n')
    
    # Find the header separator (---)
    try:
        header_index = next(i for i, line in enumerate(lines) if '---' in line)
    except StopIteration:
        print("Error: Unexpected binwalk output format.")
        return []

    entries = []
    # Process lines after the header
    for line in lines[header_index + 1:]:
        # Use regex to capture the three columns
        match = re.match(r'(\d+)\s+(0x[0-9A-Fa-f]+)\s+(.+)', line)
        if match:
            # Store decimal offset, hex offset, and description
            entries.append({
                'dec_offset': int(match.group(1)),
                'hex_offset': match.group(2),
                'description': match.group(3).strip()
            })
    return entries


def extract_file_with_dd(filename, offset, description):
    """Uses dd to extract data starting from the given offset."""
    
    # Create a clean output filename
    clean_desc = re.sub(r'[^a-zA-Z0-9]', '_', description)
    output_filename = f"extracted_{offset}_{clean_desc}.bin"

    print(f"\n[*] Extracting '{description}' (Offset: {offset}) to end of file...")
    
    try:
        # Execute dd: bs=1 (byte by byte), skip=OFFSET
        subprocess.run(
            ['dd', f'if={filename}', f'of={output_filename}', 'bs=1', f'skip={offset}'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE # Capture dd stats (bytes copied)
        )
        print(f"[+] Success: Data saved to '{output_filename}'")

    except subprocess.CalledProcessError as e:
        print(f"Error executing dd: {e.stderr.decode()}")
    except FileNotFoundError:
        print("Error: 'dd' command not found or installed.")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    
    # 1. Run Binwalk
    binwalk_output = run_binwalk(filename)
    
    # 2. Parse output
    entries = parse_binwalk_output(binwalk_output)

    if not entries:
        print("No file signatures found by binwalk.")
        return

    # 3. Display selection menu
    print("\nSelect file to extract:")
    
    # Display table header
    print("Index | DECIMAL       | HEXADECIMAL   | DESCRIPTION")
    print("-" * 50)
    
    for i, entry in enumerate(entries):
        print(f"[{i:^3}] | {entry['dec_offset']:<13} | {entry['hex_offset']:<13} | {entry['description']}")

    # 4. Get user choice
    try:
        choice = input("\nEnter the index of the file to extract or 'q' to quit: ")
        if choice.lower() == 'q':
            return
        
        index = int(choice)
        if 0 <= index < len(entries):
            selected_entry = entries[index]
            
            # 5. Execute dd
            extract_file_with_dd(
                filename,
                selected_entry['dec_offset'],
                selected_entry['description']
            )
        else:
            print("Invalid index.")
    except ValueError:
        print("Invalid input.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()