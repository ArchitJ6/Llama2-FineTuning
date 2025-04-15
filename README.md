# 🦙 **Llama2-FineTuning**: Fine-tune LLAMA 2 with Custom Datasets Using LoRA and QLoRA Techniques

This project demonstrates how to **fine-tune LLAMA 2** on a custom dataset using **LoRA** and **QLoRA** techniques, leveraging **Google Colab** for the training process and **Hugging Face** for model hosting and sharing. 🌐🚀

## 📚 **Overview**

📓 Notebook Name: Llama2_FineTuning_QLoRA.ipynb

In this notebook, we will fine-tune the **LLAMA 2** model on a custom dataset, applying the **LoRA** (Low-Rank Adaptation) and **QLoRA** (Quantized Low-Rank Adaptation) techniques. These methods allow for efficient and scalable fine-tuning of large models like LLAMA 2, making it possible to perform high-quality training without requiring massive computational resources.

The notebook is designed to run in **Google Colab**, and the fine-tuned model is pushed to the **Hugging Face Model Hub** for easy sharing and deployment.

---

## 🛠️ **Setup Instructions**

### 1. **Clone the Repository**
First, clone the repository and navigate into the project directory:
```bash
git clone https://github.com/ArchitJ6/Llama2-FineTuning.git
cd Llama2-FineTuning
```

### 2. **Install Required Libraries**

Open a **Google Colab** environment and install the required dependencies:
```bash
!pip install -q accelerate==1.6.0 peft==0.15.1 bitsandbytes==0.45.5 transformers==4.51.3 trl==0.8.6
```

---

## 📝 **Notebook Workflow**

The following steps outline the process used in the notebook to fine-tune LLAMA 2 with LoRA and QLoRA:

### Step 1: Install the Required Packages 🛠️
```bash
!pip install -q accelerate==1.6.0 peft==0.15.1 bitsandbytes==0.45.5 transformers==4.51.3 trl==0.8.6
```

### Step 2: Import Libraries 📚
```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
```

### Step 3: Load the Dataset 📊
You can either use the provided dataset or load your own. We will use the dataset from [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) for fine-tuning.

### Step 4: Configure the Fine-Tuning 🏋️
The **LoRA** and **QLoRA** settings, as well as training parameters such as batch size, learning rate, and number of epochs, are configured here.

### Step 5: Fine-Tuning with LoRA & QLoRA 🦙
```python
trainer.train()
```
The model will train for one epoch with **4-bit quantization** and **LoRA** modifications. Training is done using the **SFTTrainer**.

### Step 6: Monitor Training with TensorBoard 📈
You can visualize training progress using TensorBoard:
```bash
%load_ext tensorboard
%tensorboard --logdir results/runs
```

### Step 7: Text Generation 📝
After fine-tuning, you can generate text from the model by using the pipeline for text generation. Here’s an example:
```python
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

### Step 8: Push the Model to Hugging Face 🚀
Once you are satisfied with the fine-tuned model, you can push it to Hugging Face for easy sharing:
```python
model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
```

---

## 📈 **Parameters Used**

### **QLoRA Parameters**
- **LoRA attention dimension**: 64
- **Alpha**: 16
- **Dropout**: 0.1

### **BitsAndBytes Parameters**
- **4-bit Precision**: Enabled (NF4 quantization)
- **Compute dtype**: `float16`
- **Nested Quantization**: Disabled

### **Training Parameters**
- **Epochs**: 1
- **Batch Size**: 1
- **Gradient Accumulation**: 4
- **Learning Rate**: `2e-4`
- **Weight Decay**: 0.001
- **Optimizer**: `paged_adamw_32bit`
- **Max Gradient Norm**: 0.3
- **Warmup Ratio**: 0.03

---

## 🧑‍💻 **Example Usage**

Once the model is trained, you can use it for text generation:

```python
prompt = "Tell me a joke about AI."
generated_text = pipe(f"<s>[INST] {prompt} [/INST]")
print(generated_text[0]['generated_text'])
```

---

## 🌐 **Push Your Model to Hugging Face Hub**

You can easily share your fine-tuned model by pushing it to the Hugging Face Model Hub. Make sure to log in to your Hugging Face account before uploading the model.

1. Login to Hugging Face CLI:
    ```bash
    !huggingface-cli login
    ```
2. Push the model and tokenizer:
    ```python
    model.push_to_hub(new_model, use_temp_dir=False)
    tokenizer.push_to_hub(new_model, use_temp_dir=False)
    ```

---

## 💡 **Contributing**

Feel free to fork this repository and contribute by submitting pull requests, opening issues, or suggesting improvements. Your contributions are always welcome! 🌟

---

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📅 **Acknowledgments**

- [Llama 2](https://huggingface.co/meta/llama-2-7b-chat) for providing the base model.
- [LoRA](https://arxiv.org/abs/2106.09685) for the low-rank adaptation technique.
- [Hugging Face](https://huggingface.co/) for making it easy to share and collaborate on models.
- [Google Colab](https://colab.research.google.com/) for providing free cloud resources for training.