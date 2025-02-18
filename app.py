import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import gradio as gr
import os
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import Optional


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
    vocab_size: int = 50257  # GPT-2 standard tokenizer size
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
    bos_token_id: int = 50256  # Standard for GPT-2
    eos_token_id: int = 50256  # Standard for GPT-2
    pad_token_id: int = 50256  # Explicitly set pad_token_id to be the same as eos_token_id
    tie_word_embeddings: bool = True
    use_cache: bool = True
    model_type: str = "gpt2"  # Using gpt2 as the base architecture


def load_model_from_huggingface():
    model_id = "hotshotdragon/SmolLM2-135-custom"
    print("Loading tokenizer...")
    
    # Use the standard GPT-2 tokenizer, which should be compatible with tiktoken's gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Add pad_token (GPT-2 doesn't have pad token by default)
    tokenizer.pad_token = tokenizer.eos_token  # Using EOS token as PAD token
    
    print("Loading model...")
    try:
        # Download model files
        cache_dir = "./model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try to download model.pt
        try:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename="smollm2-135_20250218_135735.pt",
                cache_dir=cache_dir
            )
            print(f"Downloaded model.pt to {model_path}")
            
            # Create GPT2 config from SmolLM2Config
            config_data = SmolLM2Config()
            gpt2_config = GPT2Config(
                vocab_size=config_data.vocab_size,
                n_positions=config_data.max_position_embeddings,
                n_embd=config_data.hidden_size,
                n_layer=config_data.num_hidden_layers,
                n_head=config_data.num_attention_heads,
                n_inner=config_data.intermediate_size,
                activation_function=config_data.hidden_act,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                layer_norm_epsilon=config_data.rms_norm_eps,
                bos_token_id=config_data.bos_token_id,
                eos_token_id=config_data.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Initialize model with custom config
            model = GPT2LMHeadModel(gpt2_config)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location="cpu")
            
            # Check if state dict needs processing
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            
            # Try loading state dict, allowing for missing keys
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            
            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).half()  # Convert to fp16
            
        except Exception as e:
            print(f"Error loading model.pt: {str(e)}")
            print("Trying to load other model files...")
            
            # Try to find other model files
            try:
                model_files = ["pytorch_model.bin", "model.bin", "pytorch_model.pt"]
                for filename in model_files:
                    try:
                        model_path = hf_hub_download(
                            repo_id=model_id,
                            filename=filename,
                            cache_dir=cache_dir
                        )
                        print(f"Found and downloaded {filename} to {model_path}")
                        
                        # Create GPT2 config from SmolLM2Config
                        config_data = SmolLM2Config()
                        gpt2_config = GPT2Config(
                            vocab_size=config_data.vocab_size,
                            n_positions=config_data.max_position_embeddings,
                            n_embd=config_data.hidden_size,
                            n_layer=config_data.num_hidden_layers,
                            n_head=config_data.num_attention_heads,
                            n_inner=config_data.intermediate_size,
                            activation_function=config_data.hidden_act,
                            resid_pdrop=0.0,
                            embd_pdrop=0.0,
                            attn_pdrop=0.0,
                            layer_norm_epsilon=config_data.rms_norm_eps,
                            bos_token_id=config_data.bos_token_id,
                            eos_token_id=config_data.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id
                        )
                        
                        # Initialize model with custom config
                        model = GPT2LMHeadModel(gpt2_config)
                        
                        # Load state dict
                        state_dict = torch.load(model_path, map_location="cpu")
                        
                        # Check if state dict needs processing
                        if "model_state_dict" in state_dict:
                            state_dict = state_dict["model_state_dict"]
                        
                        # Try loading state dict, allowing for missing keys
                        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                        print(f"Missing keys: {missing_keys}")
                        print(f"Unexpected keys: {unexpected_keys}")
                        
                        # Move to appropriate device
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model = model.to(device).half()  # Convert to fp16
                        break
                    except Exception as file_e:
                        print(f"Error loading {filename}: {str(file_e)}")
                        continue
                
                if 'model' not in locals():
                    raise Exception("Couldn't find any compatible model files")
                
            except Exception as alt_e:
                print(f"Error with alternative model files: {str(alt_e)}")
                raise
            
    except Exception as e:
        print(f"All loading attempts failed: {str(e)}")
        raise
    
    return model, tokenizer


class TextGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            self.model, self.tokenizer = load_model_from_huggingface()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> str:
        try:
            # Tokenize the input prompt
            encoded_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Move input_ids and attention_mask to the device
            input_ids = encoded_prompt["input_ids"].to(self.device)
            attention_mask = encoded_prompt["attention_mask"].to(self.device)
            
            # Generate text with the model
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Explicitly provide attention mask
                max_length=max_length + input_ids.shape[1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            # Decode and clean the generated texts
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Optionally, remove the prompt from the generated text
                text = text[len(prompt):].strip()
                generated_texts.append(text)

            return "\n".join(generated_texts)

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return f"An error occurred: {str(e)}"


def create_interface():
    generator = TextGenerator()

    def generate_text(prompt: str, max_length: int, temperature: float, top_k: int, top_p: float) -> str:
        return generator.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    with gr.Blocks() as demo:
        gr.Markdown("# Lord of The Rings Text Generator")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter your Lord of the Rings themed prompt here...",
                    lines=5,
                )
                with gr.Row():
                    with gr.Column():
                        max_length = gr.Slider(
                            minimum=10,
                            maximum=2048,
                            value=100,
                            step=10,
                            label="Maximum Length",
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                        )
                    with gr.Column():
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top K",
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top P",
                        )
                generate_button = gr.Button("Generate")

            with gr.Column():
                output = gr.Textbox(label="Generated Text", lines=10)

        generate_button.click(
            fn=generate_text,
            inputs=[prompt, max_length, temperature, top_k, top_p],
            outputs=output,
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)