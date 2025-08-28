# Creating Scores
# Scores are a measure of how much one response is preferred over another. In general, these scores can be absolute, subjective, or relative. For this course, we will focus on the first two because they are most valuable for creating preference datasets. This scoring is a way of judging and evaluating using language models and therefore has some overlap with the evaluation techniques we have seen in the chapter on evaluation. As with the other evaluation techniques, scores and evaluations normally require larger models to better align with human preferences.

# UltraFeedback
# UltraFeedback is a technique that generates scores and critiques for a given prompt and its completion.

# The scores are based on the quality of the completion according to a set of criteria. There are four fine-grained criteria: helpfulness, relevance, deepening, and creativity. These are useful but generally speaking, using the overall criteria is a good start, which allows us to simplify the process of generating scores. The scores can be used to determine which completion is the chosen and which is the rejected one. Because they are absolute, they can also be used as interesting filters for outliers in the dataset, either finding the worst completions or the pairs with more or less difference.

# The critiques are added to provide reasoning for the score. They can be used as extra context to help us understand the differences between the scores. The language model generates extensive critiques which is very useful, but this also introduces extra cost and complexity to the process because generating critiques is more expensive than generating a single token to represent a score.

##########
# Evaluate the model's outputs based on various criteria: Helpfulness, Relevance, Deepening, Creativity
# Your role is to provide a holistic assessment based on the above factors.
# Score the output from 1 to 5 on overall quality.

# Answer with the following format: score - rationale

# # Input
# {{ input }}

# # Response
# {{ output }}

# # Score - Rationale
##########


from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import UltraFeedback
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
llm_model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

llm = TransformersLLM(model=llm_model, generation_kwargs=GENERATION_CONFIG)
ultrafeedback = UltraFeedback(llm=llm)
ultrafeedback.load()

instruction = "What is synthetic data?"
completion_a = "Synthetic data is artificially generated data that mimics real-world usage."
completion_b = "Synthetic data refers to data that has been generated artificially."

feedback_results = next(ultrafeedback.process([{
    "instruction": instruction,
    "generations": [completion_a, completion_b]
}]))

# Print the results nicely
print("="*80)
print("ULTRAFEEDBACK EVALUATION RESULTS")
print("="*80)

result = feedback_results[0]  # Get the first (and only) result

print(f"📝 INSTRUCTION: {result['instruction']}")
print("\n" + "─"*60)

# Loop through generations and their evaluations
for i, (generation, rating, rationale) in enumerate(zip(
    result['generations'], 
    result['ratings'], 
    result['rationales']
), 1):
    
    print(f"\n🤖 GENERATION {i}:")
    print(f"   Text: \"{generation}\"")
    print(f"   ⭐ Rating: {rating}/5")
    print(f"   💭 Rationale: {rationale}")

print("\n" + "="*80)
print(f"🏆 WINNER: Generation {result['ratings'].index(max(result['ratings'])) + 1} (Rating: {max(result['ratings'])})")
print("="*80)
# Example Output
# [
#     {
#         'ratings': [4, 5],
#         'rationales': ['could have been more specific', 'good definition'],
#     }
# ]