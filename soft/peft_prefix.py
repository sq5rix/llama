from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datasets import Dataset
from peft import (
    get_peft_config, 
    PeftModel, 
    PeftConfig,
    PromptTuningConfig, 
    PrefixTuningConfig,
    TaskType
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Literal
from enum import Enum

class TuningType(str, Enum):
    SOFT_PROMPT = "soft_prompt"
    PREFIX = "prefix"

class LLMTuner:
    def __init__(
        self, 
        base_model_name: str = "llama2-3b",
        tuning_type: TuningType = TuningType.SOFT_PROMPT,
        num_virtual_tokens: int = 8,
        num_prefix_tokens: int = 8,
        prefix_projection: bool = True
    ):
        """
        Initialize the tuner with support for both soft prompt and prefix tuning.
        
        Args:
            base_model_name: Name of the base model
            tuning_type: Type of tuning to use (soft_prompt or prefix)
            num_virtual_tokens: Number of virtual tokens for soft prompting
            num_prefix_tokens: Number of prefix tokens for prefix tuning
            prefix_projection: Whether to use prefix projection layer
        """
        self.base_model_name = base_model_name
        self.tuning_type = tuning_type
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"huggingface/{base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            f"huggingface/{base_model_name}",
            torch_dtype=torch.float16
        )
        
        # Configure PEFT based on tuning type
        if tuning_type == TuningType.SOFT_PROMPT:
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=num_virtual_tokens,
                prompt_tuning_init="TEXT",
                tokenizer_name_or_path=f"huggingface/{base_model_name}"
            )
        else:  # PREFIX tuning
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=num_prefix_tokens,
                prefix_projection=prefix_projection,
                token_dim=self.model.config.hidden_size,
                num_transformer_submodules=3,  # typically for attention layers
                num_attention_heads=self.model.config.num_attention_heads,
                num_layers=self.model.config.num_hidden_layers
            )
        
        # Initialize PEFT model
        self.model = PeftModel.from_pretrained(
            self.model, 
            peft_config
        )
        
        # Initialize Ollama for inference
        self.ollama = Ollama(model=base_model_name)

    def prepare_training_data(
        self, 
        examples: List[Dict[str, str]]
    ) -> Dataset:
        """
        Prepare training data in the format required by PEFT.
        
        Args:
            examples: List of dictionaries with 'input' and 'output' keys
        """
        return Dataset.from_dict({
            'input_text': [ex['input'] for ex in examples],
            'target_text': [ex['output'] for ex in examples]
        })

    def train(
        self,
        training_data: List[Dict[str, str]],
        num_epochs: int = 3,
        learning_rate: float = 1e-3,
        batch_size: int = 4,
        grad_accumulation_steps: int = 2
    ):
        """
        Train the model using the specified tuning method.
        
        Args:
            training_data: List of training examples
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            grad_accumulation_steps: Number of steps for gradient accumulation
        """
        dataset = self.prepare_training_data(training_data)
        
        # Training arguments
        training_args = {
            'learning_rate': learning_rate,
            'num_train_epochs': num_epochs,
            'per_device_train_batch_size': batch_size,
            'gradient_accumulation_steps': grad_accumulation_steps,
            'save_strategy': 'epoch',
        }
        
        # Train the model
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx in range(0, len(dataset), batch_size):
                batch_data = dataset[batch_idx:batch_idx + batch_size]
                
                # Prepare input
                inputs = self.tokenizer(
                    batch_data['input_text'],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Prepare target
                targets = self.tokenizer(
                    batch_data['target_text'],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Forward pass
                outputs = self.model(**inputs, labels=targets['input_ids'])
                loss = outputs.loss / grad_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                if (batch_idx + 1) % grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * grad_accumulation_steps
            
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            self.save_model(f"checkpoint_epoch_{epoch+1}")

    def save_model(self, path: str):
        """Save the trained model."""
        self.model.save_pretrained(path)
        # Save tuning type information
        with open(f"{path}/tuning_config.json", "w") as f:
            import json
            json.dump({"tuning_type": self.tuning_type}, f)

    def load_model(self, path: str):
        """Load a trained model."""
        # Load tuning type information
        with open(f"{path}/tuning_config.json", "r") as f:
            import json
            config = json.load(f)
            self.tuning_type = config["tuning_type"]
        
        # Load model
        self.model = PeftModel.from_pretrained(
            self.model,
            path
        )

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text using the trained model.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
        """
        # Create appropriate template based on tuning type
        if self.tuning_type == TuningType.SOFT_PROMPT:
            template = "{soft_prompt}{input_text}"
        else:  # PREFIX
            template = "{prefix}{input_text}"
            
        prompt_template = PromptTemplate(
            input_variables=["input_text"],
            template=template
        )
        
        chain = LLMChain(
            llm=self.ollama,
            prompt=prompt_template
        )
        
        return chain.run(
            input_text=prompt,
            max_tokens=max_length
        )

# Example usage
def example_usage():
    # For soft prompt tuning
    soft_prompt_tuner = LLMTuner(
        base_model_name="llama2-3b",
        tuning_type=TuningType.SOFT_PROMPT,
        num_virtual_tokens=8
    )

    # For prefix tuning
    prefix_tuner = LLMTuner(
        base_model_name="llama2-3b",
        tuning_type=TuningType.PREFIX,
        num_prefix_tokens=8,
        prefix_projection=True
    )

    # Training data
    training_data = [
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a branch of AI..."
        },
        {
            "input": "Explain neural networks",
            "output": "Neural networks are computational models..."
        }
    ]

    # Train models
    soft_prompt_tuner.train(training_data)
    prefix_tuner.train(training_data)

    # Generate text
    soft_prompt_response = soft_prompt_tuner.generate("What is deep learning?")
    prefix_response = prefix_tuner.generate("What is deep learning?")