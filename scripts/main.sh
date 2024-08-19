nohup python main_distributed.py --dataset=crosstask_how --name=test_predictor --gpu=6 \
      --base_model=predictor --epoch=1 > out/output_test_predictor.log 2>&1 &

nohup python main_distributed.py --dataset=crosstask_how --name=how1 --gpu=0 \
      --base_model=base > out/output_how1.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=how2 --gpu=1 \
      --base_model=base > out/output_how2.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=how3 --gpu=2 \
      --base_model=base > out/output_how3.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=how4 --gpu=3 \
      --base_model=base > out/output_how4.log 2>&1 &

nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor1 --gpu=4 \
      --base_model=predictor > out/output_howpredictor1.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor2 --gpu=5 \
      --base_model=predictor > out/output_howpredictor2.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor3 --gpu=6 \
      --base_model=predictor > out/output_howpredictor3.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor4 --gpu=3 \
      --base_model=predictor > out/output_howpredictor4.log 2>&1 &

python inference.py --resume --base_model=predictor --ckpt_path=/home/zhouyufan/Projects/PDPP/save_max/epoch_new_interpolation3_
0038_0.pth.tar > output.txt


nohup python main_distributed.py --dataset=crosstask_base --multiprocessing-distributed --name=base1 \
      --base_model=base --batch_size=256 --batch_size_val=256 > out/output_base1.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_base --name=base2 --gpu=1 \
      --base_model=base > out/output_base2.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_base --name=base3 --gpu=3 \
      --base_model=base > out/output_base3.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_base --name=base4 --gpu=4 \
      --base_model=base > out/output_base4.log 2>&1 &

