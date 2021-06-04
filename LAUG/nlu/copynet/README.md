# CopyNet

The original paper can be found at [ACL Anthology](https://www.aclweb.org/anthology/P16-1154/) 

## Preprocess

```python
cd $dataset$
python preprocess.py
```

## Train

``` python
PYTHONPATH=../../.. allennlp train $config_file$ -s $save_dir$ --include-package convlab2.nlu.copynet
```

## Finetune

```python
PYTHONPATH=../../.. allennlp fine-tune -c $config_file$ -s $save_dir$ -m $model_archive$ --include-package convlab2.nlu.copynet
```

## Use

```python
PYTHONPATH=../../.. allennlp predict $model_archive$ $test_tsv_file$ --predictor copynet --include-package convlab2.nlu.copynet --batch-size 64 --output-file $output_file$ --silent --cuda-device 0
```

## Test

```python
python copy_merge.py $test_file$ $output_file$
```

then

```python
cd ..
python evaluate_g.py $output_file$
```

## Data Format

```
dialog history $ user utterance \t dialog act seq
```

