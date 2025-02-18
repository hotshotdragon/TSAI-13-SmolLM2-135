# SmolLM2 Language Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyTorch](https://img.shields.io/badge/PyTorch+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

SmolLM2 is a lightweight LLaMA-inspired language model implementation designed for training on consumer-grade hardware.

## Overview

SmolLM2 implements a smaller version of the LLaMA architecture with:
- 30 transformer layers
- 576 hidden dimension size
- 9 attention heads (with 3 key-value heads for efficient computation)
- Rotary positional embeddings (RoPE)
- SiLU activation function
- Cosine learning rate schedule with warmup

## Features

- **Efficient KV-Caching**: Uses grouped-query attention for faster inference
- **Rotary Position Embeddings**: For better handling of positional information
- **Gradient Clipping**: To stabilize training
- **Top-k and Top-p Sampling**: For controllable text generation
- **Checkpointing**: Regular model checkpoints with timestamp-based naming
- **Rotary Position Embeddings (RoPE)**: Modern positional encoding scheme
- **Weight Tying**: Embedding layer and output layer share weights to reduce parameter count
- **RMS Normalization**: Used instead of traditional LayerNorm for better stability
- **SiLU Activation**: Non-linear activation function in MLP blocks

## Requirements

- Python 3.10+
- PyTorch
- tiktoken
- gradio


## Training

The model is trained on a text corpus using a simple dataloader. The default dataset is 'Lord-of-the-Rings.txt', but you can replace it with your own text file.

```bash
python train.py
```

### Training Parameters

- **Batch Size**: 2
- **Sequence Length**: 1024
- **Max Learning Rate**: 6e-4
- **Min Learning Rate**: 6e-5
- **Warmup Steps**: 2000
- **Max Steps**: 5000

## Generation

The model can generate text using top-k and top-p sampling:

```python
prompt = "Through Rohan over fen and field where the long grass grows"
generated_text = generate_text(model, prompt, 
                              max_new_tokens=50, 
                              temperature=1.2, 
                              top_k=50, 
                              top_p=0.95)
print(generated_text)
```


## Model Architecture

SmolLM2 follows a decoder-only transformer architecture inspired by the Llama model family:

- **Parameters**: ~135M
- **Hidden Size**: 576
- **Intermediate Size (MLP)**: 1536
- **Layers**: 30 transformer decoder layers
- **Attention Heads**: 9
- **Key-Value Heads**: 3 (grouped-query attention)
- **Max Sequence Length**: 2048 tokens
- **Activation Function**: SiLU (Sigmoid Linear Unit)
- **Normalization**: RMS Normalization
- **Vocabulary Size**: 50,257 tokens

### Hyperparameters

```python
@dataclass
class SmolLM2Config:
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-5
    vocab_size: int = 50257
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
```

## Checkpoints

The model automatically saves checkpoints every 500 steps and at the end of training. Checkpoints are stored in the `saved_models` directory with the format:

```
smollm2-135_YYYYMMDD_HHMMSS.pt
```

## Logging

Training progress is logged to both console and a file (`training-v3.log`), including:
- Loss values
- Training speed (tokens/sec)
- Gradient norm
- Generated text samples at checkpoint intervals

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This implementation is inspired by the LLaMA model architecture by Meta AI Research and builds on the broader advances in transformer-based language models.
