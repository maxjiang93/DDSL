export CUDA_VISIBLE_DEVICES=1

rm -r processed_data
mkdir processed_data

python3 process_airfoil_shapes.py

python3 process_airfoil_cfd_data.py
