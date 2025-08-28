from distilabel.steps.tasks import EvolInstruct
# Login to Hugging Face to access models and push datasets
from huggingface_hub import login

# You can either:
# 1. Pass your token directly: login("your_token_here")
# 2. Use the notebook widget to enter your token interactively
# 3. Set the HF_TOKEN environment variable
login("")


from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration


# HuggingFaceTB/SmolLM2-135M-Instruct
# HuggingFaceTB/SmolLM2-360M-Instruct
# HuggingFaceTB/SmolLM2-1.7B-Instruct
# HuggingFaceTB/SmolLM3-3B
model_name = "HuggingFaceTB/SmolLM3-3B"

GENERATION_CONFIG = {
    "max_new_tokens": 5000,  # Increase from default (usually 128)
}

with Pipeline() as pipeline:

    llm = TransformersLLM(model=model_name, generation_kwargs=GENERATION_CONFIG)
    initial_instruction = "Generate a short question about the Hugging Face Smol-Course."
    print("Initial Instruction:\n", initial_instruction, "\n")
    # ???? Generate instruction prompt (A)
    data_a = LoadDataFromDicts(data=[{"instruction": initial_instruction}])
    print("Instruction Prompt:\n", data_a, "\n")

    # Create instructions(B) based on instruction prompt(A)
    gen_b = TextGeneration(llm=llm, output_mappings={"generation": "instruction"})
    print("Generated Instructions:\n", gen_b, "\n")

    # Create a response(C) for the instructions
    gen_c = TextGeneration(llm=llm, output_mappings={"generation": "response"})
    print("Generated Response:\n", gen_c, "\n")
    data_a >> gen_b >> gen_c

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    print("Instruction:\n", distiset["default"]["train"][0]['instruction'], "\n")
    print("Generated Instructions:\n", distiset, "\n")
    print("Response:\n", distiset["default"]["train"][0]['response'], "\n")

    # print(distiset["default"]["train"][0])
    # Example Output
# [{
#   "instruction": "What is the purpose of Smol-Course?",
#   "response": "The Smol-Course is a platform designed to learning computer science concepts."
# }]