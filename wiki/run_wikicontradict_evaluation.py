import json
import asyncio
import uuid
from pathlib import Path
from typing import Dict, List, Literal
from dotenv import load_dotenv
from datasets import load_dataset
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(".env")

import sys
sys.path.append(str(Path(__file__).parent.parent / "open_deep_research"))

from open_deep_research.deep_researcher import deep_researcher_builder


# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of instances to evaluate (None = all 253 instances, or set a number for testing)
INSTANCES = 2

# Model for WikiContradictEval judge
JUDGE_MODEL = "openai:gpt-5-mini"  # Paper used Llama-3-70B-Instruct

# Agent configuration
AGENT_CONFIG = {
    "max_structured_output_retries": 3,
    "allow_clarification": False, # Disabled: No clarification needed
    "max_concurrent_research_units": 3,
    "search_api": "none",  # Disabled: Must answer from provided context only
    "max_researcher_iterations": 3,
    "max_react_tool_calls": 6,           
    "summarization_model": "openai:google/gemini-2.5-flash-lite",
    "summarization_model_max_tokens": 8192,
    "research_model": "openai:gpt-4o-mini",
    "research_model_max_tokens": 10000,
    "compression_model": "openai:google/gemini-2.0-flash-lite-001",
    "compression_model_max_tokens": 10000,
    "final_report_model": "openai:gpt-5-mini",
    "final_report_model_max_tokens": 10000,
}

OUTPUT_FILE = Path(__file__).parent / "wikicontradict_results.jsonl"


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def create_template_4_prompt(question: str, passage1: str, passage2: str) -> str:
    """
    Prompt Template 4: Basic context-based answering
    """
    context = f"{passage1}\n\n{passage2}"
    return f"""Provide a short answer for the following question based on the given context.
Question: {question}
Context: {context}"""


def create_template_5_prompt(question: str, passage1: str, passage2: str) -> str:
    """
    Prompt Template 5: Contradiction-aware answering
    """
    context = f"{passage1}\n\n{passage2}"
    return f"""Provide a short answer for the following question based on the given context.
Carefully investigate the given context and provide a concise response that reflects the comprehensive view of the context, even if the answer contains contradictory information reflecting the heterogeneous nature of the context.
Question: {question}
Context: {context}"""


# ============================================================================
# WIKICONTRADICTEVAL JUDGE
# ============================================================================

JUDGE_SYSTEM_PROMPT = """Please evaluate the response to a question under relaxed evaluation, where hallucinations, outdated information are allowed, as long as the primary answer is accurate. Each response is evaluated as correct, partially correct, or incorrect. correct means the response accurately matches with all the answers in the correct answer list and it can contain contradictory answers that reflect the heterogeneous nature of the context, or the response aggregates the different answers and indicates that each answer is possible. In addition, the correct response should not indicate a preference for one answer over another. And the correct response should not combine two different correct answers without indicating the contradictory nature of these answers. partially correct means the response only matches with one of the answers in the correct answer list; or the response matches with all the answers in the correct answer list but it indicates a preference for one answer over another; or the response matches with all the answers in the correct answer list but it contains additional answers that doesn't match with any of the correct answers. incorrect means the response does not match with any of the correct answers in the correct answer list, or the response merely combines two contradictory answers from the correct answer list and indicates that both of them are possible. Note that for each question, there are multiple correct answers based on different sources even though these correct answers contradict each other. Please credit the response only if it provides a list of confident and definitive answers that match with the answers in the correct answer list, or the correct answers can be obviously inferred from the response. The primary or final answers when standing alone must be accurate. Any additional information that is provided must not contradict the primary answers or reshape one's perception of them. For answers that involve names of entities (e.g., people), complete names or commonly recognized names are expected. Regarding numerical answers, approximate numbers are generally not accepted unless explicitly included in the ground-truth answers. We accept responses that contain hallucinated or outdated information that does not significantly impact the primary answers.

Respond with ONLY one of: correct, partially correct, or incorrect"""


async def judge_response(
    question: str,
    model_response: str,
    correct_answers: str,
    judge_llm
) -> Literal["Correct", "Partially Correct", "Incorrect"]:
    """
    Use WikiContradictEval to judge a model response.
    
    Args:
        question: The question asked
        model_response: The model's answer
        correct_answers: Pipe-separated correct answers (e.g., "answer1|answer2")
        judge_llm: LLM to use as judge
        
    Returns:
        One of: "Correct", "Partially Correct", "Incorrect"
    """
    user_prompt = f"""Question: {question}

Correct Answers (separated by |): {correct_answers}

Model Response: {model_response}

Based on the evaluation criteria, classify this response as Correct, Partially Correct, or Incorrect.
Respond with ONLY the classification label."""

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    response = await judge_llm.ainvoke(messages)
    judgment = response.content.strip().lower()
    
    # Normalize judgment (case-insensitive)
    if "partially correct" in judgment:
        return "Partially Correct"
    elif "correct" in judgment and "partially" not in judgment and "incorrect" not in judgment:
        return "Correct"
    elif "incorrect" in judgment:
        return "Incorrect"
    else:
        # Default to Incorrect if unclear
        print(f"Warning: Unclear judgment '{judgment}', defaulting to Incorrect")
        return "Incorrect"


# ============================================================================
# AGENT QUERY
# ============================================================================

async def query_agent(prompt: str, agent_config: Dict) -> str:
    """
    Query the open_deep_research agent with a prompt.
    
    Args:
        prompt: The input prompt
        agent_config: Configuration for the agent
        
    Returns:
        The agent's response as a string
    """
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            **agent_config
        }
    }
    
    try:
        final_state = await graph.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config
        )
        
        # Extract the final response
        if "messages" in final_state and len(final_state["messages"]) > 0:
            last_message = final_state["messages"][-1]
            if hasattr(last_message, "content"):
                return last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                return last_message["content"]
        
        return "[No response generated]"
        
    except Exception as e:
        print(f"Error querying agent: {e}")
        return f"[Error: {str(e)}]"


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

async def evaluate_instance(
    instance: Dict,
    agent_config: Dict,
    judge_llm
) -> Dict:
    """
    Evaluate a single WikiContradict instance.
    
    Args:
        instance: Dataset instance with question, passages, and answers
        agent_config: Configuration for the agent
        judge_llm: LLM to use as judge
        
    Returns:
        Dictionary with prompts, responses, and judgments
    """
    question = instance["question"]
    passage1 = instance["context1"]
    passage2 = instance["context2"]
    correct_answers = instance["ref_answer"]  # e.g., "answer1|answer2"
    
    # Generate prompts
    template4_prompt = create_template_4_prompt(question, passage1, passage2)
    template5_prompt = create_template_5_prompt(question, passage1, passage2)
    
    print(f"\n{'='*80}")
    print(f"Question ID: {instance.get('question_ID', 'N/A')}")
    print(f"Question: {question[:100]}...")
    
    # Query agent with Template 4
    print("\nQuerying with Template 4...")
    template4_response = await query_agent(template4_prompt, agent_config)
    
    # Query agent with Template 5
    print("Querying with Template 5...")
    template5_response = await query_agent(template5_prompt, agent_config)
    
    # Judge responses
    print("Judging responses...")
    template4_judgment = await judge_response(
        question, template4_response, correct_answers, judge_llm
    )
    template5_judgment = await judge_response(
        question, template5_response, correct_answers, judge_llm
    )
    
    print(f"Template 4: {template4_judgment}")
    print(f"Template 5: {template5_judgment}")
    
    return {
        "question_id": instance.get("question_ID"),
        "question": question,
        "context1": passage1,
        "context2": passage2,
        "correct_answers": correct_answers,
        "template4_prompt": template4_prompt,
        "template4_response": template4_response,
        "template4_judgment": template4_judgment,
        "template5_prompt": template5_prompt,
        "template5_response": template5_response,
        "template5_judgment": template5_judgment,
    }


async def main():
    """
    Main evaluation pipeline.
    """
    print("Loading WikiContradict dataset from HuggingFace...")
    dataset = load_dataset("ibm-research/Wikipedia_contradict_benchmark", split="train")
    
    print(f"Dataset loaded: {len(dataset)} instances")
    
    # Limit instances if configured
    if INSTANCES is not None:
        dataset = dataset.select(range(min(INSTANCES, len(dataset))))
        print(f"Limited to first {len(dataset)} instances (INSTANCES={INSTANCES})")
    
    # Initialize judge LLM using init_chat_model (same as agent)
    # This allows us to use the "provider:model" format consistently
    judge_llm = init_chat_model(model=JUDGE_MODEL, temperature=0)
    
    # Create output file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Evaluate each instance
    for i, instance in enumerate(dataset):
        print(f"\n{'#'*80}")
        print(f"Processing instance {i+1}/{len(dataset)}")
        
        try:
            result = await evaluate_instance(instance, AGENT_CONFIG, judge_llm)
            results.append(result)
            
            # Save incrementally
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")
                
        except Exception as e:
            print(f"Error processing instance {i+1}: {e}")
            continue
    
    # Compute summary statistics
    template4_correct = sum(1 for r in results if r["template4_judgment"] == "Correct")
    template4_partial = sum(1 for r in results if r["template4_judgment"] == "Partially Correct")
    template4_incorrect = sum(1 for r in results if r["template4_judgment"] == "Incorrect")
    
    template5_correct = sum(1 for r in results if r["template5_judgment"] == "Correct")
    template5_partial = sum(1 for r in results if r["template5_judgment"] == "Partially Correct")
    template5_incorrect = sum(1 for r in results if r["template5_judgment"] == "Incorrect")
    
    total = len(results)
    
    summary = {
        "total_instances": total,
        "instances_limit": INSTANCES,
        "full_evaluation": INSTANCES is None,
        "template4": {
            "correct": template4_correct,
            "partially_correct": template4_partial,
            "incorrect": template4_incorrect,
            "correct_pct": (template4_correct / total * 100) if total > 0 else 0,
            "partially_correct_pct": (template4_partial / total * 100) if total > 0 else 0,
            "incorrect_pct": (template4_incorrect / total * 100) if total > 0 else 0,
        },
        "template5": {
            "correct": template5_correct,
            "partially_correct": template5_partial,
            "incorrect": template5_incorrect,
            "correct_pct": (template5_correct / total * 100) if total > 0 else 0,
            "partially_correct_pct": (template5_partial / total * 100) if total > 0 else 0,
            "incorrect_pct": (template5_incorrect / total * 100) if total > 0 else 0,
        }
    }
    
    # Save summary
    summary_file = OUTPUT_FILE.parent / "wikicontradict_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Summary saved to: {summary_file}")
    print("\nSUMMARY:")
    print(f"Total instances evaluated: {total}")
    if INSTANCES is not None:
        print(f"⚠️  Partial evaluation: Limited to first {INSTANCES} instances (out of 253 total)")
    else:
        print(f"✓ Full evaluation: All 253 instances evaluated")
    print("\nTemplate 4 (Basic):")
    print(f"  Correct: {template4_correct} ({summary['template4']['correct_pct']:.1f}%)")
    print(f"  Partially Correct: {template4_partial} ({summary['template4']['partially_correct_pct']:.1f}%)")
    print(f"  Incorrect: {template4_incorrect} ({summary['template4']['incorrect_pct']:.1f}%)")
    print("\nTemplate 5 (Contradiction-aware):")
    print(f"  Correct: {template5_correct} ({summary['template5']['correct_pct']:.1f}%)")
    print(f"  Partially Correct: {template5_partial} ({summary['template5']['partially_correct_pct']:.1f}%)")
    print(f"  Incorrect: {template5_incorrect} ({summary['template5']['incorrect_pct']:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
