export CUDA_VISIBLE_DEVICES=0

python3 optim.py --airfoil n0012 --target 95 --C example.txt --savefile n0012_to_95 --step_size 4e-5
