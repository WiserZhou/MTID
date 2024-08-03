nohup python main_distributed.py --dataset=crosstask_how --name=new_interpolation3 --gpu=3 \
      --base_model=predictor > out/output_new_interpolation3.log 2>&1 &

python inference.py --resume --base_model=predictor --ckpt_path=/home/zhouyufan/Projects/PDPP/save_max/epoch_new_interpolation3_
0038_0.pth.tar > output.txt


nohup python train_mlp.py --name=note --dataset=crosstask_base --gpu=1 > out/output_note.log 2>&1 &

nohup python train_mlp.py --name=note9 --dataset=coin --gpu=7 > out/output_note9.log 2>&1 &
nohup python train_mlp.py --name=note10 --dataset=NIV --gpu=7 > out/output_note10.log 2>&1 &


python temp.py --num_thread_reader=1 --resume --dataset=crosstask_base \
       --batch_size=32 --batch_size_val=32 \
       --ckpt_path=/home/zhouyufan/Projects/PDPP/save_max_mlp/epoch_test_new_mlp_0068.pth.tar --gpu=6


nohup python train_mlp.py --name=test_new_mlp --dataset=crosstask_base --gpu=4 > out/output_test_new_mlp.log 2>&1 &