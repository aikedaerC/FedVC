@echo on  # Enables command echoing so you can see what is being run
python main.py --dataset=nvidia --model=FADNet --alg=fedavg --lr=1e-4 --mu=5 --epochs=1 --comm_round=100 --n_parties=10 --partition=noniid --beta=0.5 --decay=sqrt --logdir=logs5\fedavg05 --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia --device=cuda:0

python main.py --dataset=nvidia --model=FADNet --alg=fedprox --lr=1e-4 --mu=5 --epochs=1 --comm_round=100 --n_parties=10 --partition=noniid --beta=0.5 --decay=sqrt --logdir=logs5\fedprox05 --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia --device=cuda:0

python main.py --dataset=nvidia --model=FADNetFFA --alg=fedfa --lr=1e-6 --mu=5 --epochs=1 --comm_round=100 --n_parties=10 --partition=noniid --beta=0.5 --decay=sqrt --logdir=logs5\fedfa05 --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia --device=cuda:0

python main.py --dataset=nvidia --model=FADNet --alg=moon --lr=1e-4 --mu=5 --epochs=1 --comm_round=100 --n_parties=10 --partition=noniid --beta=0.5 --decay=sqrt --logdir=logs5\moon05 --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia --device=cuda:0

python main.py --dataset=nvidia --model=FADNet --alg=scaffold --lr=1e-4 --mu=5 --epochs=1 --comm_round=100 --n_parties=10 --partition=noniid --beta=0.5 --decay=sqrt --logdir=logs5\scaffold05 --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia --device=cuda:0

python main.py --dataset=nvidia --model=FADNet --alg=fedsam --lr=1e-3 --mu=5 --epochs=1 --comm_round=100 --n_parties=10 --partition=noniid --beta=0.5 --decay=sqrt --logdir=logs5\fedsam05 --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia --device=cuda:0

python virtual_clients.py --dataset=nvidia --model=FADNet --alg=fedvc --lr=1e-4 --vir_clients_num=20 --epochs=1 --comm_round=100 --n_parties=10 --partition=noniid --beta=0.5 --decay sqrt --logdir=logs5\fedvc05 --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia --device="cuda:0"


