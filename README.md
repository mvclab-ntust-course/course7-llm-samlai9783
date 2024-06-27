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
![image](https://github.com/mvclab-ntust-course/course7-llm-samlai9783/assets/95666854/930372dc-81e5-45b7-92d8-bdb48dc48c2a)

![image](https://github.com/mvclab-ntust-course/course7-llm-samlai9783/assets/95666854/abd4ccdf-9cab-4495-b7ff-40ea95573749)

---
### Config 2
```
r=32,
lora_alpha=64,
lora_dropout=0.1
trainable model parameters: 147866/66430468 = 2.20%
```
![image](https://github.com/mvclab-ntust-course/course7-llm-samlai9783/assets/95666854/7eae0b96-2390-4801-8ab6-5c5201b02fa1)
![image](https://github.com/mvclab-ntust-course/course7-llm-samlai9783/assets/95666854/38054304-7c0e-4682-a77f-451dbbb3d1de)

---
### Config 3
```
r=2,
lora_alpha=8,
lora_dropout=0.05
trainable model parameters: 647426/66430468 = 0.97%
```
![image](https://github.com/mvclab-ntust-course/course7-llm-samlai9783/assets/95666854/77bb106c-5e6b-47ee-87de-74eca3bbcb30)


![image](https://github.com/mvclab-ntust-course/course7-llm-samlai9783/assets/95666854/95ae4fd3-648b-4300-adfa-d2d4d8e407f2)

