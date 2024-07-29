nohup python main_distributed.py --dataset=crosstask_how --name=new_interpolation3 --gpu=3 \
      --base_model=predictor > out/output_new_interpolation3.log 2>&1 &


nohup python train_mlp.py --name=note --dataset=crosstask_base --gpu=3 > out/output_note.log 2>&1 &
