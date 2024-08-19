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


nohup python train_mlp.py --name=note1 --dataset=coin --gpu=1 --epochs=260 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.5 > out/output_note1.log 2>&1 &
nohup python train_mlp.py --name=note2 --dataset=coin --gpu=2 --epochs=260 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.5 > out/output_note2.log 2>&1 &
nohup python train_mlp.py --name=note3 --dataset=coin --gpu=3 --epochs=260 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.5 > out/output_note3.log 2>&1 &
nohup python train_mlp.py --name=note4 --dataset=coin --gpu=4 --epochs=260 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.5 > out/output_note4.log 2>&1 &
nohup python train_mlp.py --name=note5 --dataset=coin --gpu=5 --epochs=260 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.5 > out/output_note5.log 2>&1 &
nohup python train_mlp.py --name=note6 --dataset=coin --gpu=6 --epochs=260 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.5 > out/output_note6.log 2>&1 &
nohup python train_mlp.py --name=note7 --dataset=coin --gpu=1 --epochs=260 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.5 > out/output_note7.log 2>&1 &
nohup python train_mlp.py --name=note8 --dataset=coin --gpu=2 --epochs=260 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.5 > out/output_note8.log 2>&1 &
nohup python train_mlp.py --name=note9 --dataset=coin --gpu=3 --epochs=260 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.5 > out/output_note9.log 2>&1 &

nohup python train_mlp.py --name=note1 --dataset=crosstask_base --gpu=5 --epochs=160 > out/output_note1.log 2>&1 &
nohup python train_mlp.py --name=note4 --dataset=coin --gpu=6  --epochs=260 > out/output_note4.log 2>&1 &
nohup python train_mlp.py --name=note3 --dataset=NIV --gpu=4  --epochs=160 > out/output_note3.log 2>&1 &

nohup python train_mlp.py --name=note5 --dataset=crosstask_how --horizon=4 \
      --gpu=5 --epochs=160 > out/output_note5.log 2>&1 &

python temp.py --num_thread_reader=1 --resume --dataset=coin \
       --batch_size=32 --batch_size_val=32 --horizon=3 \
       --num_heads=4 --num_layers=2 --dim_feedforward=1024 --dropout=0.5 \
       --ckpt_path=/home/zhouyufan/Projects/PDPP/save_max_mlp/epoch_note5_0184.pth.tar --gpu=1


nohup python train_mlp.py --name=test_new_mlp --dataset=crosstask_how \
      --batch_size=8 --batch_size_val=8 \
      --epochs=140 --gpu=6 > out/output_test_new_mlp.log 2>&1 &


nohup python train_mlp.py --multiprocessing-distributed --batch_size=8 \
      --batch_size_val=8 --name=test_new_mlp --dataset=crosstask_base \
      --epochs=140 > out/output_test_new_mlp.log 2>&1 &
      
nohup sshpass -p zhoumou123 scp -r raw zhouyufan@10.0.3.227:/home/shenhaoyu/dataset/pytorch_datasets > output_scp.log 2>&1 &

du -sh */

nohup tar -xvf ILSVRC2012_img_val.tar -C val/ > out2.log 2>&1 &



