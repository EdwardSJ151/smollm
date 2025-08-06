import os
os.environ["HF_TOKEN_DUMMY"] = "XXXXXXX"

import shutil
from pathlib import Path
cache_dir = Path("/root/.cache/distilabel")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"Cleared cache directory: {cache_dir}")

from datasets import load_dataset
from distilabel.models import ClientvLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    llm = ClientvLLM(
        base_url="http://localhost:8081/v1",
        tokenizer="Qwen/Qwen3-8B",
        model="helium"
    )
    
    text_generation = TextGeneration(llm=llm)

if __name__ == "__main__":
    dataset = load_dataset("distilabel-internal-testing/instructions", split="test")
    
    # Run with runtime parameters
    distiset = pipeline.run(
        dataset=dataset,
        parameters={
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.6,
                        "max_new_tokens": 2048,
                        "seed": 687590824,
                        "extra_body": {
                            "chat_template_kwargs": {
                                "enable_thinking": False
                            }
                        }
                    }
                }
            }
        }
    )

    distiset.push_to_hub(repo_id="EdwardSJ151/test-distilabel")