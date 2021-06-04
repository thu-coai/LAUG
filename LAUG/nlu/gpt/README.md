# GPT

The code derives from [HuggingFace/Transformers](https://github.com/huggingface/transformers).

## Preprocess

```python
cd $dataset$
python preprocess.py
```

## Train

``` python
python train.py --output_dir=$output_dir$ --model_type=gpt2 --model_name_or_path=gpt2 --do_train --do_eval --eval_data_file=$test_file$ --overwrite_cache --use_tokenize --train_data_file=$train_file$ --overwrite_output_dir
```

## Use

```python
python run.py --model_type=gpt2 --model_name_or_path=$save_dir$ --num_samples 1 --input_file=$test_file$ --top_k 5 --output_file=$output_file$ --length 100 --stop_token '<|endoftext|>' --batch_size 16
```

or

```
python run_single.py --model_type=gpt2 --model_name_or_path=$save_dir$ --num_samples 1 --input_file=$test_file$ --top_k 5 --output_file=$output_file$ --length 80 --stop_token '<|endoftext|>'
```

## Test

```python
cd ..
python evaluate_g.py $test_file$
```

## Data Format

```
dialog history $ user utterance & dialog act seq
```

