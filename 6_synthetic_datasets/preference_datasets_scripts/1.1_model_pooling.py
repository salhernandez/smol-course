from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import TextGeneration
# Login to Hugging Face to access models and push datasets
from huggingface_hub import login

# You can either:
# 1. Pass your token directly: login("your_token_here")
# 2. Use the notebook widget to enter your token interactively
# 3. Set the HF_TOKEN environment variable
login("")


# Model pooling
# You can use models from different model families to generate a second completion, which is called model pooling. To further improve the quality of the second completion, you can use different generation arguments, like tweaking the temperature. Lastly, you can use different prompt templates or system prompts to generate a second completion to ensure diversity based on specific characteristics defined in the template. In theory, we could take two models of varying quality and use the better one as the chosen completion.
from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

# Configuration for consistent generation settings across all LLMs
# GENERATION_CONFIG = {
#     "max_new_tokens": 5000,  # Increase from default (usually 128)
# }

# HuggingFaceTB/SmolLM2-135M-Instruct
# HuggingFaceTB/SmolLM2-360M-Instruct
# HuggingFaceTB/SmolLM2-1.7B-Instruct
# The <think/> part can be removed!!
# HuggingFaceTB/SmolLM3-3B
llm_a_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
llm_b_name = "Qwen/Qwen2.5-1.5B-Instruct"

with Pipeline() as pipeline:
    data = LoadDataFromDicts(data=[{"instruction": "What is synthetic data?"}])
    llm_a = TransformersLLM(model=llm_a_name)
    gen_a = TextGeneration(llm=llm_a)
    llm_b = TransformersLLM(model=llm_b_name)
    gen_b = TextGeneration(llm=llm_b)
    group = GroupColumns(columns=["generation"])
    data >> [gen_a, gen_b] >> group

if __name__ == "__main__":
    distiset = pipeline.run()
    print("\nRESULTS\n\n");
    # print(distiset["default"]["train"]['grouped_generation'])
    print(llm_a_name,":\n\n" ,distiset["default"]["train"]['grouped_generation'][0][0])
    print("\n")
    print(llm_b_name,":\n\n" ,distiset["default"]["train"]['grouped_generation'][0][1])
# EXAMPLE OUTPUT
# {[
#   'Synthetic data is artificially generated data that mimics real-world usage.',
#   'Synthetic data refers to data that has been generated artificially.'
# ]}
