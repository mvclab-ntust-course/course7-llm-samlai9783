# Homework 7

This homework is about using PEFT(Parameter Efficient Fine-Tuning) to fine-tune a LLM(Large Language Model) and compare the results of using different LoRA(Low-Rank Adaptation) configs.

## Environment Setup
Since the program runs in Colab, we need to install some essentials.
```bash=
pip install transformers
pip install datasets
pip install accelerate
pip install peft
```

## Training Process
+ Load base model(distilbert-base-cased) and classification model
+ Data were truncated to the first 50 tokens to speed up the training process, then randomly select 128 examples for training and 32 examples for validation.
+ Data were batched into size of 16.
+ Applying LoRA to the model and set the parameters of LoRA configs.
+ Use trainer.train() to start the training progress of model
## Result
### Config 1
```
r=8,
lora_alpha=32,
lora_dropout=0.05
trainable model parameters: 813314/66430468 = 1.22%
```
![image](https://hackmd.io/_uploads/B1FzSkuUR.png)
![image](https://hackmd.io/_uploads/B18YSW_UC.png)
---
### Config 2
```
r=32,
lora_alpha=64,
lora_dropout=0.1
trainable model parameters: 147866/66430468 = 2.20%
```
![image](https://hackmd.io/_uploads/Bkg9S-OUC.png)
![image](https://hackmd.io/_uploads/BkV9HWOUA.png)
---
### Config 3
```
r=2,
lora_alpha=8,
lora_dropout=0.05
trainable model parameters: 647426/66430468 = 0.97%
```
![image](https://hackmd.io/_uploads/H1y6BZ_8R.png)

![image](https://hackmd.io/_uploads/HJqiBWOIC.png)
