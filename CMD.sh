nohup python main_distributed.py --transformer_num=1 --name=trans_num1 --gpu=2 > out/output_trans_num1.log 2>&1 &

nohup python main_distributed.py --transformer_num=2 --name=trans_num2 --gpu=3 > out/output_trans_num2.log 2>&1 &

nohup python main_distributed.py --transformer_num=3 --name=trans_num3 --gpu=4 > out/output_trans_num3.log 2>&1 &

nohup python main_distributed.py --transformer_num=4 --name=trans_num4 --gpu=7 > out/output_trans_num4.log 2>&1 &

nohup python main_distributed.py --transformer_num=5 --name=trans_num5 --gpu=2 > out/output_trans_num5.log 2>&1 &

nohup python main_distributed.py --transformer_num=6 --name=trans_num6 --gpu=3 > out/output_trans_num6.log 2>&1 &