
nohup python main_distributed.py --dataset=crosstask_how --name=how_base --gpu=0 \
      --base_model=base --horizon=3 > out/how_base.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=how_pre --gpu=1 \
      --base_model=predictor --horizon=3 --ifMask=True > out/how_pre.log 2>&1 &

nohup python main_distributed.py --dataset=NIV --name=NIV_base --gpu=2 \
      --base_model=base --horizon=3 > out/NIV_base.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV_pre --gpu=3 \
      --base_model=predictor --transformer_num=2 --dim_feedforward=1024 \
      --dropout=0.4 --horizon=3 > out/NIV_pre.log 2>&1 &

nohup python main_distributed.py --dataset=coin --name=coin_base --gpu=4 \
      --base_model=base --horizon=3 > out/coin_base.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin_how --gpu=5 \
      --base_model=predictor --transformer_num=7 --dim_feedforward=2048 \
      --dropout=0.7 --horizon=3 > out/coin_pre.log 2>&1 &