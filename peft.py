from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datasets import Dataset
from peft import get_peft_config, PeftModel, PeftConfig
from peft import PromptTuningConfig, TaskType
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class SimpleSoftPromptTuner:
    def __init__(
        self, 
        base_model_name: str = "llama2-3b",
        num_virtual_tokens: int = 8
    ):
        """
        Initialize the soft prompt tuner with simplified settings.
        
        Args:
            base_model_name: Name of the base model
            num_virtual_tokens: Number of virtual tokens for soft prompting
        """
        self.base_model_name = base_model_name
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"huggingface/{base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            f"huggingface/{base_model_name}",
            torch_dtype=torch.float16
        )
        
        # Configure PEFT for prompt tuning
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init="TEXT",
            tokenizer_name_or_path=f"huggingface/{base_model_name}"
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
        learning_rate: float = 1e-3
    ):
        """
        Train the soft prompt.
        
        Args:
            training_data: List of training examples
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        dataset = self.prepare_training_data(training_data)
        
        # Training arguments
        training_args = {
            'learning_rate': learning_rate,
            'num_train_epochs': num_epochs,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 2,
            'save_strategy': 'epoch',
        }
        
        # Train the model
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataset:
                # Prepare input
                inputs = self.tokenizer(
                    batch['input_text'],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(dataset)}")

    def save_soft_prompt(self, path: str):
        """Save the trained soft prompt."""
        self.model.save_pretrained(path)

    def load_soft_prompt(self, path: str):
        """Load a trained soft prompt."""
        self.model = PeftModel.from_pretrained(
            self.model,
            path
        )

    def generate(self, prompt: str) -> str:
        """
        Generate text using the trained soft prompt.
        
        Args:
            prompt: Input prompt text
        """
        # Create a chain with the soft-prompted model
        template = "{soft_prompt}{input_text}"
        prompt_template = PromptTemplate(
            input_variables=["input_text"],
            template=template
        )
        
        chain = LLMChain(
            llm=self.ollama,
            prompt=prompt_template
        )
        
        # Generate response
        return chain.run(input_text=prompt)