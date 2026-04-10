import os
import sys

def split_xyz_trajectory(input_filename, output_prefix="frame_"):
    """
    Splits a multi-frame XYZ trajectory file into individual, sequentially
    numbered XYZ files.

    The XYZ format is assumed to be:
    1. Line 1: Number of atoms (N)
    2. Line 2: Comment/Metadata
    3. Lines 3 to N+2: Coordinates
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_filename):
            print(f"Error: Input file not found: {input_filename}")
            sys.exit(1)

        print(f"Starting to process trajectory file: {input_filename}")

        # Open the input file for reading
        with open(input_filename, 'r') as infile:
            all_lines = infile.readlines()

        if not all_lines:
            print("The input file is empty.")
            return

        # 1. Determine the number of atoms from the very first line
        try:
            num_atoms = int(all_lines[0].strip())
            # Block size = N (atoms) + 2 (header lines)
            block_size = num_atoms + 2
            print(f"Detected {num_atoms} atoms per frame. Block size is {block_size} lines.")
        except ValueError:
            print(f"Error: Could not parse the number of atoms from the first line: '{all_lines[0].strip()}'")
            print("Please ensure the first line of the file contains only the integer count of atoms.")
            return

        total_lines = len(all_lines)
        if total_lines % block_size != 0:
            print(f"Warning: Total lines ({total_lines}) is not an exact multiple of the block size ({block_size}).")
            print("This suggests the file may be incomplete or corrupted. Processing frames found.")

        frame_count = 0
        current_frame_lines = []

        # 2. Iterate through all lines, chunking them by block_size
        for i, line in enumerate(all_lines):
            current_frame_lines.append(line)

            # Check if we have collected a full block
            if len(current_frame_lines) == block_size:
                frame_count += 1
                
                # 3. Define the output filename with leading zeros (e.g., 0001)
                # We use 4 digits for up to 9999 frames, adjust if needed.
                output_filename = f"{output_prefix}{frame_count:04d}.xyz"
                
                # 4. Write the current block to the new file
                with open(output_filename, 'w') as outfile:
                    outfile.writelines(current_frame_lines)

                print(f"  -> Written frame {frame_count} to {output_filename}")

                # Reset the lines buffer for the next frame
                current_frame_lines = []

        if frame_count == 0:
            print("No complete frames were found in the file.")
        else:
            print(f"\nProcessing complete. Total frames written: {frame_count}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

# --- Main execution block ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_trajectory.py <input_xyz_file>")
        print("Example: python split_trajectory.py trajectory.xyz")
        sys.exit(1)
        
    input_file = sys.argv[1]
    split_xyz_trajectory(input_file)