from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import TextGeneration
# Login to Hugging Face to access models and push datasets
from huggingface_hub import login

# Fix CUDA multiprocessing issues
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# You can either:
# 1. Pass your token directly: login("your_token_here")
# 2. Use the notebook widget to enter your token interactively
# 3. Set the HF_TOKEN environment variable
login("")

"""
Basic Prompting
Let's start with a basic example and load the HuggingFaceTB/SmolLM2-1.7B-Instruct model using the transformers integration of the distilabel library. We will use the TextGeneration class to generate a synthetic prompt and use that to generate a completion.
"""

# Configuration for consistent generation settings across all LLMs
GENERATION_CONFIG = {
    "max_new_tokens": 5000,  # Increase from default (usually 128)
}

# HuggingFaceTB/SmolLM2-135M-Instruct
# HuggingFaceTB/SmolLM2-360M-Instruct
# HuggingFaceTB/SmolLM2-1.7B-Instruct
# The <think/> part can be removed!!
# HuggingFaceTB/SmolLM3-3B

model_name = "HuggingFaceTB/SSmolLM3-3B"
llm = TransformersLLM(
    model=model_name,
    generation_kwargs=GENERATION_CONFIG
)
gen = TextGeneration(llm=llm)
gen.load()


# original basic prompt
prompt_for_instruction_tune = "Generate a questions about the Hugging Face Smol-Course on small AI models."
## This generates a set of instructions ###
# We will now use the llm to generate a prompt for *instruction tuning*.
result_prompt_for_instruction_tune = next(gen.process([{"instruction": prompt_for_instruction_tune}]))
print("Generated prompt:\n", result_prompt_for_instruction_tune[0]["generation"], "\n\n\n")
# Example Output- What is the purpose of Smol-Course?

## This generates completions for the set of instructions! ###
# We can use that same prompt as input to generate a completion.
prompt_for_completion = result_prompt_for_instruction_tune[0]["generation"]
completion_result = next(gen.process([{"instruction": prompt_for_completion}]))
print("Generated completion:\n", completion_result[0]["generation"], "\n\n\n")
# Example Output - The Smol-Course is a platform designed to learning computer science concepts.

# Cool! We can generated a synthetic prompt and a corresponding completion.

# SelfInstruct
# SelfInstruct is a prompt that generates new instructions based on a seed dataset. This seed data can be a single instruction or a piece of context. The process begins with a pool of initial seed data. The language model is then prompted to generate new instructions based on this seed data using in-context learning.
from distilabel.steps.tasks import SelfInstruct

self_instruct = SelfInstruct(llm=llm)
self_instruct.load()

instruction_seed_prompt = completion_result[0]["generation"]
## This generates NEW instructions for A COMPLETION ###
selfInstruct_result = next(self_instruct.process([{"input": instruction_seed_prompt}]))[0]["instructions"][0]
# Example Output - What is the process of generating synthetic data through manual prompting?

print("Generated SelfInstruct instruction:\n", selfInstruct_result,"\n\n\n")

# EvolInstruct
# EvolInstruct is a prompting technique that takes an input instruction and evolves it into a better version of the same instruction. This better version is defined according to a set of criteria and adds constraints, deepening, concretizing, reasoning or complications to the original instruction. The process can be repeated multiple times to create various evolutions of the same instruction, ideally leading to a better version of the original instruction.
from distilabel.steps.tasks import EvolInstruct

evol_instruct = EvolInstruct(llm=llm, num_evolutions=1)
evol_instruct.load()

initial_instruction = "What is the process of generating synthetic data through manual prompting"

evol_instruct_result = next(evol_instruct.process([{"instruction": initial_instruction}]))
# What is the process of generating synthetic data through manual prompting?
# Example Output - And, how does the artificial intelligence system, GPT4, use machine learning algorithms to manipulate the input data into synthetic data?
print("Before EvolInstruct instruction:\n", initial_instruction, "\n\n\n")
print("After EvolInstruct instruction:\n", evol_instruct_result[0]["evolved_instruction"], "\n\n\n")

# The instruction is now more complex but has lost some of the original meaning. So, take into account that evolving can be a double-edged sword and we need to be careful with the quality of the data we generate.


# Magpie - Generate [prompt/completions] based on a prompt
from distilabel.steps.tasks import Magpie

# Configure the LLM with Magpie template for this specific use case
llm_magpie = TransformersLLM(
    model=model_name,
    magpie_pre_query_template="<|im_start|>user\n",
    generation_kwargs=GENERATION_CONFIG
)

magpie = Magpie(llm=llm_magpie)
magpie.load()

magpie_system_prompt = "You're an AI assistant that will exclusively help users solving math problems."
magpie_result = next(magpie.process([{"system_prompt": magpie_system_prompt}]))
print("Magpie system prompt:\n", magpie_system_prompt, "\n")
print("Generated Magpie instruction:\n", magpie_result[0]["instruction"], "\n")
print("Generated Magpie response:\n", magpie_result[0]["response"], "\n\n\n")
# Example output shows Magpie generates both user instructions and assistant responses
# [{
#   "role": "user",
#   "content": "Can you provide me with a list of the top 3 universities?"
# },
# {
#   "role": "assistant",
#   "content": "The top 3 universities are: MIT, Yale, Stanford."
# }]
gen.unload()

