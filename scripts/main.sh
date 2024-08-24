nohup python main_distributed.py --dataset=crosstask_how --name=test_predictor --gpu=6 \
      --base_model=predictor --epoch=1 > out/output_test_predictor.log 2>&1 &

nohup python main_distributed.py --dataset=crosstask_how --name=how1 --gpu=2 \
      --base_model=base > out/output_how1.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=how2 --gpu=1 \
      --base_model=base > out/output_how2.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=how3 --gpu=2 \
      --base_model=base > out/output_how3.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=how4 --gpu=4 \
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



nohup python main_distributed.py --dataset=coin --name=coin1 --gpu=4 \
      --base_model=base --epochs=600 --resume> out/output_coin1.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin2 --gpu=4 \
      --base_model=base --epochs=600 --resume> out/output_coin2.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin3 --gpu=5 \
      --base_model=base --epochs=600 --resume> out/output_coin3.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin4 --gpu=6 \
      --base_model=base --epochs=600 --resume> out/output_coin4.log 2>&1 &


nohup python main_distributed.py --dataset=coin --name=coin5 --gpu=4 \
      --base_model=base > out/output_coin5.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin6 --gpu=4 \
      --base_model=base > out/output_coin6.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin7 --gpu=6 \
      --base_model=base > out/output_coin7.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin8 --gpu=6 \
      --base_model=base > out/output_coin8.log 2>&1 &


python main_distributed.py --dataset=crosstask_how --name=how1 --gpu=4 \
      --base_model=base


nohup python main_distributed.py --dataset=NIV --name=NIV1 --gpu=4 \
      --base_model=base > out/output_NIV1.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV2 --gpu=3 \
      --base_model=base --epochs=400 > out/output_NIV2.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV3 --gpu=4 \
      --base_model=base --epochs=400 > out/output_NIV3.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV4 --gpu=5 \
      --base_model=base --epochs=400 > out/output_NIV4.log 2>&1 &


nohup python main_distributed.py --dataset=NIV --name=NIV1 --gpu=4 \
      --base_model=predictor --horizon=4 > out/output_NIV1.log 2>&1 &