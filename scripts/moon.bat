@echo on  # Enables command echoing so you can see what is being run

REM Run for dataset nvidia
python main.py --dataset=nvidia ^
    --model=FADNetFFA ^
    --alg=moon ^
    --lr=0.0001 ^
    --mu=5 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.5 ^
    --decay=sqrt ^
    --logdir=logs4\moon05 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia ^
    --device=cuda:0

REM Run for dataset carla
python main.py --dataset=carla ^
    --model=FADNetFFA ^
    --alg=moon ^
    --lr=0.0001 ^
    --mu=5 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.5 ^
    --decay=sqrt ^
    --logdir=logs4\moon05 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_carla\gaia ^
    --device=cuda:0

REM Run for dataset gazebo
python main.py --dataset=gazebo ^
    --model=FADNetFFA ^
    --alg=moon ^
    --lr=0.0001 ^
    --mu=5 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.5 ^
    --decay=sqrt ^
    --logdir=logs4\moon05 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_gazebo\gaia ^
    --device=cuda:0

REM Run another batch with different beta
python main.py --dataset=nvidia ^
    --model=FADNetFFA ^
    --alg=moon ^
    --lr=0.0001 ^
    --mu=5 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.1 ^
    --decay=sqrt ^
    --logdir=logs4\moon01 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia ^
    --device=cuda:0

REM Repeat for other datasets
python main.py --dataset=carla ^
    --model=FADNetFFA ^
    --alg=moon ^
    --lr=0.0001 ^
    --mu=5 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.1 ^
    --decay=sqrt ^
    --logdir=logs4\moon01 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_carla\gaia ^
    --device=cuda:0

python main.py --dataset=gazebo ^
    --model=FADNetFFA ^
    --alg=moon ^
    --lr=0.0001 ^
    --mu=5 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.1 ^
    --decay=sqrt ^
    --logdir=logs4\moon01 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_gazebo\gaia ^
    --device=cuda:0
