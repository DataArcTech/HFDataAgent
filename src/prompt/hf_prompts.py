KEYWORD_EXTRACTION_PROMPT = """You can summarize the domain of the task into a few keywords. You can refer to these task examples:

TASK_DESCRIPTION: "You are given a word problem involving basic arithmetic, algebra, or geometry. Your task is to carefully read the problem and provide a step-by-step solution for it."
KEYWORDS: ["math reasoning", "arithmetic", "algebra", "geometry"]

TASK_DESCRIPTION: "You are given a short passage and a multiple-choice question based on it. Your task is to carefully read the passage, reason about the information, and select the most logically correct answer from the provided choices."
KEYWORDS: ["reading comprehension", "logical reasoning", "multiple choice", "text understanding"]

TASK_DESCRIPTION: "Your task is to answer challenging, graduate-level multiple-choice questions spanning Physics, Chemistry, and Biology, requiring deep subject-matter knowledge, complex reasoning, calculation, and synthesis of information."
KEYWORDS: ["multiple choice", "Physics", "Chemistry", "Biology", "scientific reasoning", "complex reasoning", "calculation"]

TASK_DESCRIPTION: "{task_description}"
KEYWORDS:
"""

FIELD_FILTER_PROMPT = """You are a data field identifier that determines which fields in a JSON-like object represent the instruction's input and output.

Given a JSON structure or similar text, your task is to analyze its keys and contents to decide:
- which field represents the **input** (the user's question, instruction, or request)
- which field represents the **output** (the answer, response, or completion)

### Output format:
Return a JSON object with two keys:
{
  "input": "<name of the input field>",
  "output": "<name of the output field>"
}

### Rules:
1. Choose the field names that most likely correspond to the instruction (input) and answer (output).
2. Only select from the provided legal keys.
3. If you cannot identify clear input and output fields, return:
   {
     "input": null,
     "output": null
   }
4. Do NOT infer or fabricate field names not present in the text.
5. Only output the field names, not their contents.

### Example
Input: "{example_text}"
Legal Keys: ["question", "answer", "topic"]

Output:
{
  "input": "question",
  "output": "answer"
}

Input: {row}
Legal Keys: {legal_keys}

Output:
"""

FORMAT_CONVERSION_PROMPT = """You are a format conversion assistant that rewrites given input-output pairs from one format to another.

### Task
You are given:
- **Original Input:** a user's instruction or question.
- **Original Output:** the model's answer or completion.
- **Input Format:** description of how the input is currently formatted.
- **Output Format:** description of how it should be formatted after conversion.

Your task:
- Rewrite BOTH the input and output according to the new format.
- Preserve the original content and meaning of the input and output.
- Follow the target format conventions strictly (tone, structure, delimiters, etc.).
- Do not add explanations, only provide the final formatted content.

### Output format
Return only a JSON object:
{
  "input": "<rewritten input>",
  "output": "<rewritten output>"
}

---

Original Input:
{input}

Original Output:
{output}

Input Format:
{input_format}

Output Format:
{output_format}

Return:
"""

INSTRUCTION_JUDGE_PROMPT = """You are an expert LLM evaluator for instruction-tuning datasets. Your goal is to assess how helpful and appropriate an instruction sample is for training a model on a specific task.
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

{
  "Relevance": "<0–10>",
  "Correctness": "<0–10>",
  "Helpfulness": "<0–10>",
  "Clarity": "<0–10>",
  "Difficulty": "<0–10>"
}

---

### Inputs

**Task Definition:**  
{task_description}

**Instruction Sample:**  
{instruction_sample}

---

### Output
"""

SOLVABLE_JUDGE_PROMPT = """Given the instruction sample below and the model's attempted solution, determine if the model successfully solved the problem.
Instruction Sample: {instruction_sample}
Model's Solution: {solution}
Return "True" if the model solved it correctly, otherwise return "False".
"""

DATA_GENERATOR_ZERO_SHOT_PROMPT = """You are a data generation assistant that creates realistic and high-quality instruction data for fine-tuning language models.

### Task
Generate a **single** input-output pair based on the following description and format rules.

---

**Task Description:**
{task_description}

**Input Format:**
{input_format}

**Output Format:**
{output_format}

---

### Requirements:
1. The generated pair must be **original**, **plausible**, and consistent with the given task description.
2. The **input** should follow the input format and represent a valid instruction or question for this task.
3. The **output** should follow the output format and be a correct, helpful, and complete response to the input.
4. Return only the final JSON object in this format:
{
  "input": "<generated input>",
  "output": "<generated output>"
}

---

Now generate one example:
"""

DATA_GENERATOR_FEW_SHOT_PROMPT = """
You are a data generation assistant that creates diverse, realistic, and high-quality instruction data for fine-tuning large language models.

### Task
Generate a **single** input-output pair based on the task description and few-shot examples below.

---

**Task Description:**
{task_description}

**Input Format:**
{input_format}

**Output Format:**
{output_format}

**Few-shot Examples:**
{example_text}

---

### Requirements:
1. Create a new example that fits the same **task type** and **format** as the given examples.
2. The new pair should be **original** — not a paraphrase or trivial variation.
3. Keep **semantic consistency** with the task, but introduce diversity in topic, difficulty, or structure.
4. The input and output must both follow the described formats.
5. Return only the final JSON object in this format:
{
  "input": "<generated input>",
  "output": "<generated output>"
}

---

Now generate one example:
"""
