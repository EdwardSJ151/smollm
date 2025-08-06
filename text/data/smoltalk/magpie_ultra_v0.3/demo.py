import os
os.environ["HF_TOKEN_DUMMY"] = "XXXXX"

import shutil
from pathlib import Path
cache_dir = Path("/root/.cache/distilabel")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"Cleared cache directory: {cache_dir}")

from datasets import load_dataset

from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    TextGeneration(
        llm=vLLM(
            model="Qwen/Qwen3-8B",
            generation_kwargs={"temperature": 0.7, 
                                "max_new_tokens": 512,
                                "skip_special_tokens": True
            }
        ),
    )

if __name__ == "__main__":
    dataset = load_dataset("distilabel-internal-testing/instructions", split="test")
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="EdwardSJ151/test-distilabel")