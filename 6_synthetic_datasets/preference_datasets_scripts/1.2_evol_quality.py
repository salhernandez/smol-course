# IMPORTANT - EvolQuality - Evolves original prompt!!
# EvolQuality is similar to EvolInstruct - it is a prompting technique but it evolves completions instead of the input prompt. The task takes both a prompt and completion and evolves the completion into a version that better responds to the prompt based on a set of criteria. This better version is defined according to criteria for improving helpfulness, relevance, deepening, creativity, or details. Because this automatically generates a second completion, we can use it to add more completions to a dataset. In theory, we could even assume the evolution is better than the original completion and use it as the chosen completion out of the box.
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import EvolQuality
from huggingface_hub import login

# Login to Hugging Face to access mode
login("")

# Configuration for consistent generation settings across all LLMs
GENERATION_CONFIG = {
    "max_new_tokens": 5000,  # Increase from default (usually 128)
}


# HuggingFaceTB/SmolLM2-135M-Instruct
# HuggingFaceTB/SmolLM2-360M-Instruct
# HuggingFaceTB/SmolLM2-1.7B-Instruct
# The <think/> part can be removed!!
# HuggingFaceTB/SmolLM3-3B
# Qwen/Qwen2.5-1.5B-Instruct
# Qwen/Qwen3-4B-Instruct-2507
# Qwen/Qwen2.5-0.5B-Instruct
llm_model = "Qwen/Qwen2.5-0.5B-Instruct"

llm = TransformersLLM(model=llm_model, generation_kwargs=GENERATION_CONFIG)
total_evolutions= 1
evol_quality = EvolQuality(llm=llm, num_evolutions=5)
evol_quality.load()

instruction = "What is synthetic data?"
completion = "Synthetic data is artificially generated data that mimics real-world usage."

evee = next(evol_quality.process([{
    "instruction": instruction,
    "response": completion
}]))
# Example Output - The process of generating synthetic data through manual prompting involves creating artificial data sets that mimic real-world usage patterns.

print("\n")
print("Original Instruction:","\n", instruction)
print("\n\n")
print("Original Completion:","\n", completion)
print("\n\n")
print("Evolved Completion:","\n", evee[0]["evolved_response"])