nohup python main_distributed.py --dataset=crosstask_how --name=test_predictor --gpu=6 \
      --base_model=predictor --epoch=1 > out/output_test_predictor.log 2>&1 &

nohup python main_distributed.py --dataset=coin --name=coin1 --gpu=4 \
      --base_model=base --epochs=600 --resume> out/output_coin1.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin2 --gpu=4 \
      --base_model=base --epochs=600 --resume> out/output_coin2.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin3 --gpu=5 \
      --base_model=base --epochs=600 --resume> out/output_coin3.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin4 --gpu=6 \
      --base_model=base --epochs=600 --resume> out/output_coin4.log 2>&1 &


nohup python main_distributed.py --dataset=coin --name=coin --gpu=7 \
      --base_model=base --horizon=4 > out/output_coin.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin2 --gpu=7 \
      --base_model=base --horizon=4 --seed=3407 > out/output_coin2.log 2>&1 &

nohup python main_distributed.py --dataset=coin --name=coin6 --gpu=4 \
      --base_model=base > out/output_coin6.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin7 --gpu=6 \
      --base_model=base > out/output_coin7.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin8 --gpu=6 \
      --base_model=base > out/output_coin8.log 2>&1 &


python main_distributed.py --dataset=crosstask_how --name=how1 --gpu=4 \
      --base_model=base

nohup python main_distributed.py --dataset=coin --name=coin --gpu=7 \
      --base_model=base --horizon=4 --resume > out/output_coin.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin2 --gpu=7 \
      --base_model=base --horizon=4 --seed=3407 --resume > out/output_coin2.log 2>&1 &



nohup python main_distributed.py --dataset=NIV --name=NIV3 --gpu=4 \
      --base_model=predictor --transformer_num=2 > out/output_NIV3.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV4 --gpu=5 \
      --base_model=predictor --transformer_num=2 > out/output_NIV4.log 2>&1 &


nohup python main_distributed.py --dataset=NIV --name=NIV1 --gpu=4 \
      --base_model=predictor --horizon=4 > out/output_NIV1.log 2>&1 &


nohup bash run.sh > output.log 2>&1 &




nohup python main_distributed.py --dataset=NIV --name=NIV1 --gpu=0 \
      --base_model=predictor --transformer_num=2 --ie_num=1 --seed=3407 --resume > out/NIVtransformer_num2ie_num1_1.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV2 --gpu=3 \
      --base_model=predictor --transformer_num=2 --ie_num=2 --seed=3407 --resume > out/NIVtransformer_num2ie_num2_2.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV3 --gpu=2 \
      --base_model=predictor --transformer_num=2 --ie_num=1 --seed=3406 --resume > out/NIVtransformer_num2ie_num1_3.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV4 --gpu=5 \
      --base_model=predictor --transformer_num=2 --ie_num=2 --seed=3406 --resume > out/NIVtransformer_num2ie_num2_4.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV5 --gpu=4 \
      --base_model=predictor --transformer_num=2 --ie_num=1 --resume > out/NIVtransformer_num2ie_num1_5.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV6 --gpu=0 \
      --base_model=predictor --transformer_num=2 --ie_num=2 --resume > out/NIVtransformer_num2ie_num2_6.log 2>&1 &


nohup python main_distributed.py --dataset=crosstask_how --name=how1 --gpu=0 \
      --base_model=base --horizon=5 > out/output_how1.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=how2 --gpu=1 \
      --base_model=base --horizon=6 > out/output_how2.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=how3 --gpu=2 \
      --base_model=base --horizon=5 --seed=3407 > out/output_how3.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=how4 --gpu=3 \
      --base_model=base --horizon=6 --seed=3407 > out/output_how4.log 2>&1 &


nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor111 --gpu=7 \
      --base_model=predictor > out/output_howpredictor111.log 2>&1 &

nohup python main_distributed.py --dataset=crosstask_how --name=test1 --gpu=0 --base_model=prefull --horizon=4 > out/test1.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=test2 --gpu=1 --base_model=prehalf --horizon=4 > out/test2.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=test3 --gpu=2 --base_model=predictor --horizon=4 > out/test3.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=test4 --gpu=3 --base_model=prefull --horizon=4 --seed=3407 > out/test4.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=test5 --gpu=4 --base_model=prehalf --horizon=4 --seed=3407 > out/test5.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=test6 --gpu=5 --base_model=predictor --horizon=4 --seed=3407 > out/test6.log 2>&1 &

nohup python main_distributed.py --dataset=NIV --name=NIV1 --gpu=6 \
      --base_model=base --horizon=3 --seed=3407 > out/output_NIV1.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV2 --gpu=6 \
      --base_model=base --horizon=3 > out/output_NIV2.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV3 --gpu=7 \
      --base_model=base --horizon=4 --seed=3407 > out/output_NIV3.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV4 --gpu=7 \
      --base_model=base --horizon=4 > out/output_NIV4.log 2>&1 &

python inference.py --resume --base_model=base --dataset=NIV --name=NIV2 --horizon=3 --ckpt_path=/data/zhaobo/zhouyufan/PDPP-Optimize/save_max/epoch_NIV2_0066_0.pth.tar --gpu=0

nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor1 --gpu=0 \
      --base_model=predictor --horizon=3 --seed=3407 --resume > out/output_howpredictor1.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor2 --gpu=1 \
      --base_model=predictor --horizon=3 --resume > out/output_howpredictor2.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor3 --gpu=2 \
      --base_model=predictor --horizon=4 --seed=3407 --resume > out/output_howpredictor3.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor4 --gpu=3 \
      --base_model=predictor --horizon=4 --resume > out/output_howpredictor4.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor5 --gpu=4 \
      --base_model=predictor --horizon=5 --seed=3407 --resume > out/output_howpredictor5.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor6 --gpu=5 \
      --base_model=predictor --horizon=5 --resume > out/output_howpredictor6.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor7 --gpu=6 \
      --base_model=predictor --horizon=6 --seed=3407 --resume > out/output_howpredictor7.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=howpredictor8 --gpu=7 \
      --base_model=predictor --horizon=6 --resume > out/output_howpredictor8.log 2>&1 &
      
nohup python main_distributed.py --dataset=coin --name=coin1 --gpu=0 \
      --base_model=predictor --transformer_num=7 --dim_feedforward=2048 \
      --dropout=0.7 --horizon=3 --seed=3407 --resume > out/coin1.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin2 --gpu=1 \
      --base_model=predictor --transformer_num=7 --dim_feedforward=2048 \
      --dropout=0.7 --horizon=3 --resume > out/coin2.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin3 --gpu=2 \
      --base_model=predictor --transformer_num=7 --dim_feedforward=2048 \
      --dropout=0.7 --horizon=4 --seed=3407 --resume > out/coin3.log 2>&1 &
nohup python main_distributed.py --dataset=coin --name=coin4 --gpu=3 \
      --base_model=predictor --transformer_num=7 --dim_feedforward=2048 \
      --dropout=0.7 --horizon=4 --resume > out/coin4.log 2>&1 &


nohup python main_distributed.py --dataset=NIV --name=NIV1 --gpu=4 \
      --base_model=predictor --transformer_num=2 --dim_feedforward=1024 \
      --dropout=0.4 --horizon=3 --seed=3407 --resume > out/NIV1.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV2 --gpu=5 \
      --base_model=predictor --transformer_num=2 --dim_feedforward=1024 \
      --dropout=0.4 --horizon=3 --resume > out/NIV2.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV3 --gpu=6 \
      --base_model=predictor --transformer_num=2 --dim_feedforward=1024 \
      --dropout=0.4 --horizon=4 --seed=3407 --resume > out/NIV3.log 2>&1 &
nohup python main_distributed.py --dataset=NIV --name=NIV4 --gpu=7 \
      --base_model=predictor --transformer_num=2 --dim_feedforward=1024 \
      --dropout=0.4 --horizon=4 --resume > out/NIV4.log 2>&1 &



nohup python main_distributed.py --dataset=crosstask_how --name=test_loss --gpu=7 \
      --base_model=base --horizon=3 --seed=3407 --loss_type=Sequence_CE > out/test_loss.log 2>&1 &
nohup python main_distributed.py --dataset=crosstask_how --name=test_loss2 --gpu=6 \
      --base_model=base --horizon=3 --loss_type=Sequence_CE > out/test_loss2.log 2>&1 &
