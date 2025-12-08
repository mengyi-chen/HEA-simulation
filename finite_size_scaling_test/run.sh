# python finite_size_scaling.py --batch_size 16 --energy_model mace --device 0
python finite_size_scaling.py --batch_size 16 --energy_model chgnet --device 0

python finite_size_scaling_full_structure.py --batch_size 1 --energy_model mace --device 0
python finite_size_scaling_full_structure.py --batch_size 1 --energy_model chgnet --device 0