## For compression with a replacement scheduler

```bash
python ./run_glue.py \
  --model_name_or_path /path/to/saved_predecessor \
  --data_dir /data/ \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006
```

## For compression with a constant replacing rate

```bash
python ./run_glue.py \
  --model_name_or_path /path/to/saved_predecessor \
  --data_dir /data/ \
  --replacing_rate 0.5 \
  --steps_for_replacing 2500
```