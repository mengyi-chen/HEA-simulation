#!/usr/bin/env python3
"""
Convert POSCAR files from kMC simulation to trajectory file for OVITO visualization
No external dependencies required (pure Python)
"""
import os
import glob
import re
import argparse

def extract_step_number(filename):
    """Extract step number from POSCAR filename"""
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def read_vasp_poscar(filename):
    """Read VASP POSCAR file manually
    
    Returns:
        dict with keys: comment, scale, cell, elements, counts, coords_type, positions
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = {}
    data['comment'] = lines[0].strip()
    data['scale'] = float(lines[1].strip())
    
    # Cell vectors
    data['cell'] = []
    for i in range(2, 5):
        data['cell'].append([float(x) for x in lines[i].split()])
    
    # Elements and counts
    data['elements'] = lines[5].split()
    data['counts'] = [int(x) for x in lines[6].split()]
    
    # Coordinate type
    data['coords_type'] = lines[7].strip()[0].upper()  # D or C
    
    # Positions
    n_atoms = sum(data['counts'])
    data['positions'] = []
    for i in range(8, 8 + n_atoms):
        pos = [float(x) for x in lines[i].split()[:3]]
        data['positions'].append(pos)
    
    return data

def fractional_to_cartesian(frac_pos, cell, scale):
    """Convert fractional coordinates to Cartesian"""
    cart = [0.0, 0.0, 0.0]
    for i in range(3):
        for j in range(3):
            cart[i] += frac_pos[j] * cell[j][i] * scale
    return cart

def write_xyz_frame(f, step, elements, counts, positions, cell, scale, coords_type):
    """Write one frame to XYZ file"""
    n_atoms = sum(counts)
    
    # Header
    f.write(f"{n_atoms}\n")
    f.write(f"Step {step}\n")
    
    # Expand element list
    atom_types = []
    for elem, count in zip(elements, counts):
        atom_types.extend([elem] * count)
    
    # Write atoms
    for atom_type, pos in zip(atom_types, positions):
        # Convert to Cartesian if needed
        if coords_type == 'D':  # Direct (fractional)
            cart_pos = fractional_to_cartesian(pos, cell, scale)
        else:  # Cartesian
            cart_pos = [x * scale for x in pos]
        
        f.write(f"{atom_type} {cart_pos[0]:.6f} {cart_pos[1]:.6f} {cart_pos[2]:.6f}\n")

def convert_to_trajectory(configs_folder, output_file="trajectory.xyz"):
    """Convert all POSCAR files to a single XYZ trajectory file"""
    
    # Find all POSCAR files
    pattern = os.path.join(configs_folder, "POSCAR_step_*.vasp")
    poscar_files = glob.glob(pattern)
    
    if len(poscar_files) == 0:
        print(f"No POSCAR files found in {configs_folder}")
        return
    
    print(f"Found {len(poscar_files)} POSCAR files")
    
    # Sort by step number
    poscar_files.sort(key=extract_step_number)
    
    # Convert
    print(f"Converting to {output_file}...")
    
    successful = 0
    with open(output_file, 'w') as f:
        for i, poscar_file in enumerate(poscar_files):
            step = extract_step_number(poscar_file)
            
            try:
                data = read_vasp_poscar(poscar_file)
                write_xyz_frame(f, step, data['elements'], data['counts'], 
                              data['positions'], data['cell'], data['scale'],
                              data['coords_type'])
                successful += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(poscar_files)} files...")
            
            except Exception as e:
                print(f"Warning: Failed to read {poscar_file}: {e}")
                continue
    
    if successful > 0:
        print(f"\n✓ Success! Trajectory saved to: {output_file}")
        print(f"  Total frames: {successful}")
        print(f"\n=== How to use in OVITO ===")
        print(f"1. Open OVITO")
        print(f"2. File -> Load File -> Select {output_file}")
        print(f"3. Use timeline slider to view animation")
        print(f"4. Tips:")
        print(f"   - Add 'Color coding' modifier to color by element type")
        print(f"   - Add 'Slice' modifier to view cross-section")
        print(f"   - Render -> Export Animation to create video (MP4)")
        print(f"\n=== Alternative: Direct Python animation ===")
        print(f"You can also use the 'animate_trajectory.py' script (if available)")
    else:
        print("Error: No valid structures found")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert POSCAR files to XYZ trajectory')
    parser.add_argument('--configs_folder', type=str, default='../configs_2025_11_26_14_20_16',
                       help='Folder containing POSCAR_step_*.vasp files')
    parser.add_argument('--output', type=str, default='configs_2025_11_26_14_20_16/trajectory_24x24x24.xyz',
                       help='Output trajectory filename (default: trajectory.xyz)')
    
    args = parser.parse_args()
    
    # Make output path absolute if needed
    if not os.path.isabs(args.output):
        args.output = os.path.join(os.path.dirname(args.configs_folder), args.output)
    
    convert_to_trajectory(args.configs_folder, args.output)
