nohup python main_distributed.py --name=end_trans4 --gpu=1 --transformer_num=4 > out/output_end_trans4.log 2>&1 &
nohup python main_distributed.py --name=end_trans3 --gpu=2 --transformer_num=3 > out/output_end_trans3.log 2>&1 &
nohup python main_distributed.py --name=end_trans2 --gpu=4 --transformer_num=2 > out/output_end_trans2.log 2>&1 &
nohup python main_distributed.py --name=end_trans1 --gpu=5 --transformer_num=1 > out/output_end_trans1.log 2>&1 &