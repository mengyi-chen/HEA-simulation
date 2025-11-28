#!/usr/bin/env python3
"""
Generate a 24x24x24 supercell of spinel structure with random Al, Cr, Fe, Co, Ni, Cu distribution
and a 6x6x24 cavity through the entire z direction, centered in xy plane
"""
import numpy as np

# Read original POSCAR
with open('POSCAR.POSCAR.vasp', 'r') as f:
    lines = f.readlines()

# Parse original structure
comment = lines[0].strip()
scale = float(lines[1].strip())
lattice_vectors = []
for i in range(2, 5):
    lattice_vectors.append([float(x) for x in lines[i].split()])
lattice_vectors = np.array(lattice_vectors)

element_line = lines[5].split()
count_line = [int(x) for x in lines[6].split()]
coordinate_type = lines[7].strip()

# Parse atomic positions
n_fe = count_line[0]  # 24 Fe atoms
n_o = count_line[1]   # 32 O atoms
total_atoms = n_fe + n_o

positions = []
for i in range(8, 8 + total_atoms):
    pos = [float(x) for x in lines[i].split()[:3]]
    positions.append(pos)

fe_positions = np.array(positions[:n_fe])
o_positions = np.array(positions[n_fe:])

print(f"Original unit cell: {n_fe} metal atoms, {n_o} O atoms")
print(f"Lattice constant: {lattice_vectors[0,0]:.4f} Å")

# Create supercell (nx * ny * nz)
# nx, ny, nz = 24, 24, 24
nx, ny, nz = 6, 6, 6
print(f"Creating {nx}x{ny}x{nz} supercell...")

# Define cavity parameters (6x6x24 unit cells - through entire z direction)
# Centered in xy, spanning full z
# cavity_size_xy = 6 # (cavity_size_xy * cavity_size_xy * nz )
cavity_size_xy = 2 # (cavity_size_xy * cavity_size_xy * nz )

# New lattice vectors
new_lattice = lattice_vectors * np.array([nx, ny, nz])[:, np.newaxis]


cavity_x_start = (nx - cavity_size_xy) / 2 / nx  # Center in x
cavity_x_end = (nx + cavity_size_xy) / 2 / nx
cavity_y_start = (ny - cavity_size_xy) / 2 / ny  # Center in y
cavity_y_end = (ny + cavity_size_xy) / 2 / ny
cavity_z_start = 0.0                              # From bottom
cavity_z_end = 1.0                                # To top (entire z direction)

print(f"\nCavity region (fractional coordinates):")
print(f"  x: {cavity_x_start:.4f} to {cavity_x_end:.4f}")
print(f"  y: {cavity_y_start:.4f} to {cavity_y_end:.4f}")
print(f"  z: {cavity_z_start:.4f} to {cavity_z_end:.4f} (entire height)")

def is_in_cavity(pos):
    """Check if position is inside the cavity region"""
    x, y, z = pos
    return (cavity_x_start <= x < cavity_x_end and
            cavity_y_start <= y < cavity_y_end and
            cavity_z_start <= z < cavity_z_end)

# Generate all metal atom positions in supercell
all_metal_positions = []
metal_in_cavity = []  # Track which atoms are in cavity
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            for pos in fe_positions:
                # Position in fractional coordinates of supercell
                new_pos = (pos + np.array([ix, iy, iz])) / np.array([nx, ny, nz])
                all_metal_positions.append(new_pos)
                metal_in_cavity.append(is_in_cavity(new_pos))

# Generate all oxygen positions in supercell
all_o_positions = []
o_in_cavity = []  # Track which atoms are in cavity
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            for pos in o_positions:
                # Position in fractional coordinates of supercell
                new_pos = (pos + np.array([ix, iy, iz])) / np.array([nx, ny, nz])
                all_o_positions.append(new_pos)
                o_in_cavity.append(is_in_cavity(new_pos))

all_metal_positions = np.array(all_metal_positions)
all_o_positions = np.array(all_o_positions)
metal_in_cavity = np.array(metal_in_cavity)
o_in_cavity = np.array(o_in_cavity)

n_total_metal = len(all_metal_positions)
n_total_o = len(all_o_positions)
n_metal_removed = np.sum(metal_in_cavity)
n_o_removed = np.sum(o_in_cavity)

print(f"\nTotal metal atoms: {n_total_metal}")
print(f"Metal atoms in cavity: {n_metal_removed}")
print(f"Metal atoms remaining: {n_total_metal - n_metal_removed}")
print(f"Total O atoms: {n_total_o}")
print(f"O atoms in cavity: {n_o_removed}")
print(f"O atoms remaining: {n_total_o - n_o_removed}")

# Randomly assign 6 elements with equal probability to metal sites (only non-cavity)
elements = ['Al', 'Cr', 'Fe', 'Co', 'Ni', 'Cu']
np.random.seed(42)  # For reproducibility

# Initialize element assignment with vacancy (0) for all atoms
element_assignment = np.zeros(n_total_metal, dtype=object)
element_assignment[:] = 'X'  # Vacancy placeholder

# Assign elements only to non-cavity atoms
non_cavity_indices = np.where(~metal_in_cavity)[0]
element_assignment[non_cavity_indices] = np.random.choice(elements, size=len(non_cavity_indices))

# Count atoms per element
element_counts = {'X': n_metal_removed}  # Vacancy count
for elem in elements:
    element_counts[elem] = np.sum(element_assignment == elem)

print("\nElement distribution (metal atoms):")
print(f"  Vacancy (X): {element_counts['X']} atoms")
for elem in elements:
    count = element_counts[elem]
    percentage = count / (n_total_metal - n_metal_removed) * 100 if n_total_metal > n_metal_removed else 0
    print(f"  {elem}: {count} atoms ({percentage:.2f}% of non-cavity)")

# Group positions by element
element_positions = {}
element_positions['X'] = all_metal_positions[metal_in_cavity]
for elem in elements:
    mask = element_assignment == elem
    element_positions[elem] = all_metal_positions[mask]

# Separate oxygen positions
o_positions_cavity = all_o_positions[o_in_cavity]
o_positions_regular = all_o_positions[~o_in_cavity]

print(f"\nOxygen distribution:")
print(f"  XO (oxygen vacancy sites in cavity): {len(o_positions_cavity)} atoms")
print(f"  O (regular oxygen): {len(o_positions_regular)} atoms")

# Write new POSCAR
# output_file = 'POSCAR_24x24x24_with_cavity.vasp'
output_file = f'POSCAR_{nx}x{ny}x{nz}_with_cavity.vasp'
with open(output_file, 'w') as f:
    # Comment line
    f.write(f"HEA spinel {nx}x{ny}x{nz} with cavity (Al,Cr,Fe,Co,Ni,Cu)O, XO for O vacancies\n")

    # Scale
    f.write("1.0\n")

    # Lattice vectors
    for vec in new_lattice:
        f.write(f"  {vec[0]:20.16f}  {vec[1]:20.16f}  {vec[2]:20.16f}\n")

    # Element names - X for metal vacancy, then elements, then XO for oxygen vacancy, then O
    all_elements = ['X'] + elements + ['XO', 'O']
    f.write("  " + "  ".join(all_elements) + "\n")

    # Element counts
    counts = [str(element_counts['X'])]
    counts += [str(element_counts[elem]) for elem in elements]
    counts += [str(len(o_positions_cavity))]  # XO count
    counts += [str(len(o_positions_regular))]  # O count
    f.write("  " + "  ".join(counts) + "\n")

    # Coordinate type
    f.write("Direct\n")

    # Write metal vacancy positions (metal sites in cavity)
    for pos in element_positions['X']:
        f.write(f"  {pos[0]:18.15f}  {pos[1]:18.15f}  {pos[2]:18.15f}\n")

    # Write metal atom positions by element
    for elem in elements:
        for pos in element_positions[elem]:
            f.write(f"  {pos[0]:18.15f}  {pos[1]:18.15f}  {pos[2]:18.15f}\n")

    # Write oxygen vacancy positions (oxygen sites in cavity) as XO
    for pos in o_positions_cavity:
        f.write(f"  {pos[0]:18.15f}  {pos[1]:18.15f}  {pos[2]:18.15f}\n")

    # Write regular oxygen positions (non-cavity)
    for pos in o_positions_regular:
        f.write(f"  {pos[0]:18.15f}  {pos[1]:18.15f}  {pos[2]:18.15f}\n")

print(f"\nSupercell with cavity written to {output_file}")
print(f"New lattice constant: {new_lattice[0,0]:.4f} Å")
print(f"Total atoms (including vacancies as X and XO): {n_total_metal + n_total_o}")
print(f"Total vacancy sites: {n_metal_removed} metal (X) + {n_o_removed} oxygen (XO)")
