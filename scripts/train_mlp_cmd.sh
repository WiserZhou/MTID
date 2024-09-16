nohup python train_mlp.py --name=note1 --dataset=coin --gpu=4 --epochs=800 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.8 > out/output_note1.log 2>&1 &
nohup python train_mlp.py --name=note2 --dataset=coin --gpu=4 --epochs=800 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.9 > out/output_note2.log 2>&1 &
nohup python train_mlp.py --name=note3 --dataset=coin --gpu=4 --epochs=800 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.3 > out/output_note3.log 2>&1 &
nohup python train_mlp.py --name=note4 --dataset=coin --gpu=4 --epochs=800 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.4 > out/output_note4.log 2>&1 &
nohup python train_mlp.py --name=note5 --dataset=coin --gpu=4 --epochs=800 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.5 > out/output_note5.log 2>&1 &
nohup python train_mlp.py --name=note6 --dataset=coin --gpu=4 --epochs=800 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.6 > out/output_note6.log 2>&1 &
nohup python train_mlp.py --name=note7 --dataset=coin --gpu=4 --epochs=800 \
      --num_heads=4 --num_layers=2 --dim_feedforward=2048 --dropout=0.7 > out/output_note7.log 2>&1 &


nohup python train_mlp.py --name=note1 --dataset=crosstask_how --gpu=0 --epochs=160 > out/output_note1.log 2>&1 &
nohup python train_mlp.py --name=note1 --dataset=crosstask_base --gpu=5 --epochs=160 > out/output_note1.log 2>&1 &
nohup python train_mlp.py --name=note4 --dataset=coin --gpu=6  --epochs=160 > out/output_note4.log 2>&1 &
nohup python train_mlp.py --name=note3 --dataset=NIV --gpu=4  --epochs=160 > out/output_note3.log 2>&1 &

nohup python train_mlp.py --name=note5 --dataset=crosstask_how --horizon=3 \
      --gpu=5 --epochs=160 > out/output_note5.log 2>&1 &

python temp.py --num_thread_reader=1 --resume --dataset=crosstask_how \
       --batch_size=32 --batch_size_val=32 --horizon=6 \
       --ckpt_path=/data/zhaobo/zhouyufan/PDPP-Optimize/save_max_mlp/epoch_note1_0022.pth.tar --gpu=7




nohup python train_mlp.py --name=note1 --dataset=crosstask_how --gpu=0 --epochs=160 --horizon=4 > out/horizon4.log 2>&1 &
nohup python train_mlp.py --name=note1 --dataset=crosstask_how --gpu=1 --epochs=160 --horizon=5 > out/horizon5.log 2>&1 &
nohup python train_mlp.py --name=note1 --dataset=crosstask_how --gpu=2 --epochs=160 --horizon=6 > out/horizon6.log 2>&1 &


nohup python train_mlp.py --name=note1 --dataset=crosstask_how --gpu=1 --epochs=160 --classfier_model=resnet32x4 > out/horizon3.log 2>&1 &

python train_mlp.py --name=note1 --dataset=crosstask_how --gpu=0 --epochs=160