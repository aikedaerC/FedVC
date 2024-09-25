@echo on  # Enables command echoing so you can see what is being run
python virtual_clients.py --dataset=nvidia ^
    --model=FADNet ^
    --alg=fedvc ^
    --lr=1e-6 ^
    --vir_clients_num=20 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.5 ^
    --decay sqrt ^
    --logdir=logs4\fedvc05 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia ^
    --device="cuda:0"

python virtual_clients.py --dataset=carla ^
    --model=FADNet ^
    --alg=fedvc ^
    --lr=1e-6 ^
    --vir_clients_num=20 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.5 ^
    --decay sqrt ^
    --logdir=logs4\fedvc05 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_carla\gaia ^
    --device="cuda:0"

python virtual_clients.py --dataset=gazebo ^
    --model=FADNet ^
    --alg=fedvc ^
    --lr=1e-6 ^
    --vir_clients_num=20 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.5 ^
    --decay sqrt ^
    --logdir=logs4\fedvc05 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_gazebo\gaia ^
    --device="cuda:0"
