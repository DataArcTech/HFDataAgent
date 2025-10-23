import re
import ast
import json
from utils import chat_complete

def keyword_extraction(task_description):
    prompt = f"""You can summarize the domain of the task into a few keywords. You can refer to these task examples:

TASK_DESCRIPTION: "You are given a word problem involving basic arithmetic, algebra, or geometry. Your task is to carefully read the problem and provide a step-by-step solution for it."
KEYWORDS: ["math reasoning", "arithmetic", "algebra", "geometry"]

TASK_DESCRIPTION: "You are given a short passage and a multiple-choice question based on it. Your task is to carefully read the passage, reason about the information, and select the most logically correct answer from the provided choices."
KEYWORDS: ["reading comprehension", "logical reasoning", "multiple choice", "text understanding"]

TASK_DESCRIPTION: "Your task is to answer challenging, graduate-level multiple-choice questions spanning Physics, Chemistry, and Biology, requiring deep subject-matter knowledge, complex reasoning, calculation, and synthesis of information."
KEYWORDS: ["multiple choice", "Physics", "Chemistry", "Biology", "scientific reasoning", "complex reasoning", "calculation"]

TASK_DESCRIPTION: "{task_description}"
KEYWORDS: """
    output = chat_complete(prompt)
    match = re.search(r'\[.*\]', output, re.S)
    if match:
        keywords = ast.literal_eval(match.group())
    else:
        keywords = [output.strip()]
        
    return keywords

def format_filter(row, legal_keys):
    example_text = """{'question': 'If an angle measures 120 degrees, what is its reference angle?', 'answer': 'The reference angle is found by subtracting ...', 'topic': 'Trigonometry Basics'}"""
    
    prompt = f"""You are a data field identifier that determines which fields in a JSON-like object represent the instruction's input and output.

Given a JSON structure or similar text, your task is to analyze its keys and contents to decide:
- which field represents the **input** (the user's question, instruction, or request)
- which field represents the **output** (the answer, response, or completion)

### Output format:
Return a JSON object with two keys:
{{
  "input": "<name of the input field>",
  "output": "<name of the output field>"
}}

### Rules:
1. Choose the field names that most likely correspond to the instruction (input) and answer (output).
2. Only select from the provided legal keys.
3. If you cannot identify clear input and output fields, return:
   {{
     "input": null,
     "output": null
   }}
4. Do NOT infer or fabricate field names not present in the text.
5. Only output the field names, not their contents.

### Example
Input: "{example_text}"
Legal Keys: ["question", "answer", "topic"]

Output:
{{
  "input": "question",
  "output": "answer"
}}

Input: {row}
Legal Keys: {legal_keys}

Output:"""

    output = chat_complete(prompt)
    match = re.search(r'\{.*\}', output, re.S)
    if match:
        json_str = match.group().strip()
        try:
            result = json.loads(json_str)
            return result
        except:
            return {"input": None, "output": None}


def instruction_judge(task_description, instruction_sample):
    prompt = f"""You are an expert LLM evaluator for instruction-tuning datasets. Your goal is to assess how helpful and appropriate an instruction sample is for training a model on a specific task.
You will be given:
1. A **Task Definition** – describing the target task the model should learn.
2. An **Instruction Sample** – containing an example instruction (and optionally a response).

You should evaluate the instruction sample across **five criteria**, using the definitions below:

---

### Evaluation Criteria

1. **Relevance (0–10)**  
   How well does the instruction sample align with the task definition and objectives?  
   - 0–2: Completely unrelated or off-task  
   - 3–5: Somewhat relevant but not fully aligned  
   - 6–8: Mostly relevant and supportive of the task goal  
   - 9–10: Perfectly aligned and directly useful for the target task

2. **Correctness (0–10)**  
   Is the response factually accurate and logically valid given the instruction?  
   - 0–2: Incorrect or nonsensical  
   - 3–5: Partially correct or includes minor errors  
   - 6–8: Mostly correct and sound  
   - 9–10: Fully correct, precise, and logically consistent  
   *(If no response is provided, evaluate correctness based on the expected answer type and structure.)*

3. **Helpfulness (0–10)**  
   Does the response provide complete, informative, and useful content that effectively fulfills the instruction?  
   - 0–2: Useless or irrelevant  
   - 3–5: Somewhat useful but incomplete or generic  
   - 6–8: Helpful and reasonably detailed  
   - 9–10: Extremely helpful, comprehensive, and valuable for learning

4. **Clarity (0–10)**  
   Is the instruction easy to understand, unambiguous, and grammatically clear?  
   - 0–2: Confusing, vague, or poorly written  
   - 3–5: Understandable but could be clearer  
   - 6–8: Clear and well-structured  
   - 9–10: Perfectly clear, concise, and unambiguous

5. **Difficulty (0–10)**  
   How challenging is this instruction–response pair relative to the target task?  
   - 0–2: Too trivial or overly complex for the task  
   - 3–5: Basic difficulty, may not stimulate learning  
   - 6–8: Appropriately challenging  
   - 9–10: Ideal level of difficulty to promote robust learning and reasoning

---

### Output Format

Return your evaluation in strict **JSON** format as follows:

{{
  "Relevance": "<0–10>",
  "Correctness": "<0–10>",
  "Helpfulness": "<0–10>",
  "Clarity": "<0–10>",
  "Difficulty": "<0–10>"
}}

---

### Inputs

**Task Definition:**  
{task_description}

**Instruction Sample:**  
{instruction_sample}

---

### Output
"""
    output = chat_complete(prompt)
    match = re.search(r'\{.*\}', output, re.S)
    if match:
        json_str = match.group().strip()
        try:
            result = json.loads(json_str)
            return result
        except:
            return {"Relevance": 5, "Correctness": 5, "Helpfulness": 5, "Clarity": 5, "Difficulty": 5}
    else:
        return {"Relevance": 5, "Correctness": 5, "Helpfulness": 5, "Clarity": 5, "Difficulty": 5}


def solvable_judge(instruction_sample):
    solve_prompt = f"""Please think step by step and answer this question. \n{instruction_sample["input"]}"""
    solution = chat_complete(solve_prompt)
    
    judge_prompt = f"""Given the instruction sample below and the model's attempted solution, determine if the model successfully solved the problem.
Instruction Sample: {instruction_sample}
Model's Solution: {solution}
Return "True" if the model solved it correctly, otherwise return "False"."""
    judge_output = chat_complete(judge_prompt).strip()
    
    if "true" in judge_output.lower():
        return True
    else:
        return False