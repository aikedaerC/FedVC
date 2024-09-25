@echo on  # Enables command echoing so you can see what is being run

REM Run for dataset nvidia
python main.py --dataset=nvidia ^
    --model=FADNetFFA ^
    --alg=fednova ^
    --lr=1e-6 ^
    --mu=5 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.5 ^
    --decay=sqrt ^
    --logdir=logs4\fednova05 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia ^
    --device=cuda:0

@REM REM Run for dataset carla
@REM python main.py --dataset=carla ^
@REM     --model=FADNetFFA ^
@REM     --alg=fednova ^
@REM     --lr=1e-6 ^
@REM     --mu=5 ^
@REM     --epochs=1 ^
@REM     --comm_round=100 ^
@REM     --n_parties=10 ^
@REM     --partition=noniid ^
@REM     --beta=0.5 ^
@REM     --decay=sqrt ^
@REM     --logdir=logs4\fednova05 ^
@REM     --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_carla\gaia ^
@REM     --device=cuda:0

@REM REM Run for dataset gazebo
@REM python main.py --dataset=gazebo ^
@REM     --model=FADNetFFA ^
@REM     --alg=fednova ^
@REM     --lr=1e-6 ^
@REM     --mu=5 ^
@REM     --epochs=1 ^
@REM     --comm_round=100 ^
@REM     --n_parties=10 ^
@REM     --partition=noniid ^
@REM     --beta=0.5 ^
@REM     --decay=sqrt ^
@REM     --logdir=logs4\fednova05 ^
@REM     --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_gazebo\gaia ^
@REM     --device=cuda:0

REM Run another batch with different beta
python main.py --dataset=nvidia ^
    --model=FADNetFFA ^
    --alg=fednova ^
    --lr=1e-6 ^
    --mu=5 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.1 ^
    --decay=sqrt ^
    --logdir=logs4\fednova01 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_nvidia\gaia ^
    --device=cuda:0

REM Repeat for other datasets
python main.py --dataset=carla ^
    --model=FADNetFFA ^
    --alg=fednova ^
    --lr=1e-6 ^
    --mu=5 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.1 ^
    --decay=sqrt ^
    --logdir=logs4\fednova01 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_carla\gaia ^
    --device=cuda:0

python main.py --dataset=gazebo ^
    --model=FADNetFFA ^
    --alg=fednova ^
    --lr=1e-6 ^
    --mu=5 ^
    --epochs=1 ^
    --comm_round=100 ^
    --n_parties=10 ^
    --partition=noniid ^
    --beta=0.1 ^
    --decay=sqrt ^
    --logdir=logs4\fednova01 ^
    --datadir=C:\Users\aikedaer\Desktop\NonIIDbench\datasets\driving_gazebo\gaia ^
    --device=cuda:0
