# llama-finetuning-lab-2

## Links

- [Inference UI](https://colab.research.google.com/drive/1CRTV5uNRT-Rk7rsGqL3GEIhFwbuy7lml#scrollTo=jNBFguh2yIoQ)
- [FineTome Finetuning](https://colab.research.google.com/drive/1JQtX5wP8P3R2MpMs4bpaqfX2TGivr2Ya#scrollTo=QmUBVEnvCDJv)
- [Hyperparameter Tuning](https://colab.research.google.com/drive/1LVJRXGl-tCTuCCpqSm9p7SsXQ_DrN5Nb#scrollTo=95_Nn-89DhsL)
- [Emoji Finetuning](https://colab.research.google.com/drive/1WNthcDGTddGWGUju0cBKd2Qh_HXwL8XD#scrollTo=upcOlWe7A1vc)

## List of Models:

- Vanilla Llama 3.1 3B (1000 steps of FineTome)
- Llama 1B - Finetuned w/ Finetome
- Llama + Emojis
- Llama + Hyperparameters (1/2 FineTome)

## Introduction

In the following repository we are going to fine-tune a pre-trained model using the Hugging Face library. The model we are going to use is the `llama-3.1`.

## Task 1

What we did first was to Finetune the `llama-3.1 1B` model using the FineTome dataset. We used the LoRA algorithm and 4bit quantization to fine-tune the model and also we created an Inference UI to test the model using Gradio:

![alt text](imgs/image-6.png)

We can see that we can input a text and a recording using Whisper. The model will then generate a response based on the input text and audio file. There is also a Dropdown list where the user can select the model to use based on all the models that were fine-tuned.

To test the model and access the Inference UI, please click [here](https://colab.research.google.com/drive/1CRTV5uNRT-Rk7rsGqL3GEIhFwbuy7lml#scrollTo=jNBFguh2yIoQ), if you have a GPU you can run the code and test the model using the `inference_ui` file.

The vanilla `llama-3.1 3B` model was trained for 1000 steps using the FineTome dataset. Just to compare it with the fine-tuned model.

### Results

We tested the model using the following exam:

### Questions

1. Who wrote the novel *Pride and Prejudice*?

2. If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?

3. Solve for \( x \): \( 2x + 3 = 7 \).

4. What does the following Python code output?

```python
nums = [1, 2, 3]
print(nums[::-1])
```

5. What is the chemical symbol for water?

6. In the sentence, “After the meeting, she decided to write a follow-up email,” what does “follow-up email” mean?

7. In which year did the Berlin Wall fall?

8. Paraphrase the following sentence: “The quick brown fox jumps over the lazy dog.”

9. What is the time complexity of a binary search algorithm?

10. Is it ethical to use someone’s personal data without their consent?

"""

- Llama 1B - Finetuned w/ Finetome : 0.6 points

## Task 2

### Model-Centric Fine-Tuning

Hyperparameters are the parameters that are set before the learning process begins. They are used to control the learning process and the model's behavior. Hyperparameter tuning is the process of finding the best hyperparameters for a model. In this task, we fine-tuned the `llama-3.1 1B` model using the FineTome dataset and the Hyperparameter tuning dataset. The model was fine-tuned using the LoRA algorithm and 4bit quantization. The hyperparameters that were tuned are:

1. `r`: Changed from 16 to 8, it is the number of bits used to represent the model's weights.
2. `lora_alpha`: Maintained at 16, it is the alpha value used in the LoRA algorithm.
3. `use_rslora`: Changed from False to True, it is a boolean value that determines whether to use the RSLora algorithm.
4. `packing`: Changed from False to True, it is a boolean value that determines whether to use the packing algorithm to combine the input and output embeddings.
5. `warmup_steps`: Changed from 5 to 1250, it is the number of steps used to warm up the model before training ($0.1 \cdot \text{total steps}$).

The model of Task 1 took us 14 hours and 7 different Google accounts to train, using the Hyperparameter tuning, we were able to train the model in 8 hours and 3 Google accounts. For this case we used half of the dataset to train the model and estimate the final time:

- `llama-3.1 1B` FineTome: 14 hours
- `llama-3.1 1B` Hyperparameter Tuning: ~8 hours (4 hours to train half of the dataset)

The results can be tested using the `finetuning-model-centric` file.

### Data-Centric Fine-Tuning: Identifying Emojis 😁

Although, the model was able to process and identify when the text was an emoji, it could not identify the emoji it self:

#### FineTome Dataset Finetuned to Identify Emojis

![alt text](imgs/image-3.png)

![alt text](imgs/image-4.png)

We can see 4 mistakes:

1. The US flag was not adopted in 1932, it was adopted in 1777.
2. This birthday cake does not have a birthday message.
3. The panda emoji is not a red panda.
4. The hand gesture emoji says that the fingers are extended but at the same time they are curled.

Therefore, we decided to fine-tune the model using the Emoji Dataset. The dataset contains 5,000 examples of text and their corresponding emojis. The model was trained for 1 epoch and the results are shown below:

#### Emoji Dataset Finetuned Examples

![alt text](imgs/image.png)


![alt text](imgs/image-2.png)

We can see that the descriptions are more accurate and the model is able to identify the emojis correctly.

However, the model is not perfect and there are still some mistakes:

![alt text](imgs/image-5.png)

Clearly Sweden is not the same country as Suriname, we tried to use the same emoji many times and the model was not able to identify it correctly. We think that if would have trained the model for more epochs, the model would have been able to identify the last emoji correctly.

