ps aux | grep main_distributed | grep -v grep | awk '{print $2}' | xargs kill
rm /data/zhaobo/zhouyufan/PDPP-Optimize/checkpoint/whl/epoch*
rm /data/zhaobo/zhouyufan/PDPP-Optimize/checkpoint_mlp/whl/epoch*
rm /data/zhaobo/zhouyufan/PDPP-Optimize/save_max/epoch*
rm /data/zhaobo/zhouyufan/PDPP-Optimize/save_max_mlp/epoch*
rm /data/zhaobo/zhouyufan/PDPP-Optimize/out/*
