# attention-text-classification
Text classification based on attention model (transformer).

## Update

When I wrote this code, there was no BERT, so the performance was not good.

Some updates you can do:
- Pretrain the encoder with MLM.
- I use the last token as the final state, which is not good. You can use a '[CLS]' token instead, just like BERT.

## Requirements

Basic requirements:
- tensorflow[-gpu] >= 1.4
- opennmt-tf

## Training data
You need preprocess your training data to use our models here. Basically, you need following files:
- training data file
- evaluation data file
- vocab file
- label file

Data files should contain json string, as the following format:
```json
{
    "content":"content should be preprocessed, using space to seperate words",
    "labels":["lable_name"]
}
```

Vocab file contains a word each line.

Label file contains a label name each line.

## Results
I tested this model on CAIL2018 dataset, which is a imbalanced data. The overall accuracy can reach up to 95%, but the macro f1 score is very low. So I think this model is not good for imbalanced data, I will try ULMFiT next.

## Todo
- try other datasets
- implement ULMFiT
