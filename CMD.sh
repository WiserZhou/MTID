nohup python main_distributed.py --dataset=crosstask_how --name=new_interpolation3 --gpu=3 \
      --base_model=predictor > out/output_new_interpolation3.log 2>&1 &

python inference.py --resume --base_model=predictor --ckpt_path=/home/zhouyufan/Projects/PDPP/save_max/epoch_new_interpolation3_
0038_0.pth.tar > output.txt


nohup python train_mlp.py --name=note3 --dataset=crosstask_base --gpu=1 > out/output_note3.log 2>&1 &
nohup python train_mlp.py --name=note4 --dataset=crosstask_base --gpu=2 > out/output_note4.log 2>&1 &
nohup python train_mlp.py --name=note5 --dataset=crosstask_base --gpu=3 > out/output_note5.log 2>&1 &
nohup python train_mlp.py --name=note6 --dataset=crosstask_base --gpu=1 > out/output_note6.log 2>&1 &
nohup python train_mlp.py --name=note7 --dataset=crosstask_base --gpu=2 > out/output_note7.log 2>&1 &
nohup python train_mlp.py --name=note8 --dataset=crosstask_base --gpu=3 > out/output_note8.log 2>&1 &

nohup python train_mlp.py --name=note9 --dataset=coin --gpu=7 > out/output_note9.log 2>&1 &
nohup python train_mlp.py --name=note10 --dataset=NIV --gpu=7 > out/output_note10.log 2>&1 &
