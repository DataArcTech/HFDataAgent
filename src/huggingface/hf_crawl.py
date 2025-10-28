import re
import ast
import json

from ..utils.llm_client import chat_complete
from ..prompt.hf_prompts import (
    KEYWORD_EXTRACTION_PROMPT,
    FIELD_FILTER_PROMPT,
    FORMAT_CONVERSION_PROMPT,
    INSTRUCTION_JUDGE_PROMPT,
    SOLVABLE_JUDGE_PROMPT,
    DATA_GENERATOR_ZERO_SHOT_PROMPT,
    DATA_GENERATOR_FEW_SHOT_PROMPT
)

async def keyword_extraction(task_description):
    prompt = KEYWORD_EXTRACTION_PROMPT.format(task_description=task_description)
    output = await chat_complete(prompt)
    match = re.search(r'\[.*\]', output, re.S)
    if match:
        keywords = ast.literal_eval(match.group())
    else:
        keywords = [output.strip()]
    return keywords


async def field_filter(row, legal_keys):
    example_text = """{'question': 'If an angle measures 120 degrees, what is its reference angle?', 'answer': 'The reference angle is found by subtracting ...', 'topic': 'Trigonometry Basics'}"""
    prompt = FIELD_FILTER_PROMPT.format(example_text=example_text, row=row, legal_keys=legal_keys)
    output = await chat_complete(prompt)
    match = re.search(r'\{.*\}', output, re.S)
    if match:
        json_str = match.group().strip()
        try:
            return json.loads(json_str)
        except:
            return {"input": None, "output": None}
    return {"input": None, "output": None}


async def format_conversion(input, output, input_format, output_format):
    prompt = FORMAT_CONVERSION_PROMPT.format(input=input, output=output, input_format=input_format, output_format=output_format)
    output_text = await chat_complete(prompt)
    match = re.search(r'\{.*\}', output_text, re.S)
    if match:
        try:
            return json.loads(match.group().strip())
        except:
            return {"input": None, "output": None}
    return {"input": None, "output": None}


async def instruction_judge(task_description, instruction_sample):
    prompt = INSTRUCTION_JUDGE_PROMPT.format(task_description=task_description, instruction_sample=instruction_sample)
    output = await chat_complete(prompt)
    match = re.search(r'\{.*\}', output, re.S)
    if match:
        try:
            return json.loads(match.group().strip())
        except:
            return {"Relevance": 5, "Correctness": 5, "Helpfulness": 5, "Clarity": 5, "Difficulty": 5}
    return {"Relevance": 5, "Correctness": 5, "Helpfulness": 5, "Clarity": 5, "Difficulty": 5}


async def solvable_judge(instruction_sample):
    solve_prompt = f"Please think step by step and answer this question.\n{instruction_sample['input']}"
    solution = await chat_complete(solve_prompt)
    judge_prompt = SOLVABLE_JUDGE_PROMPT.format(instruction_sample=instruction_sample, solution=solution)
    judge_output = (await chat_complete(judge_prompt)).strip()
    return "true" in judge_output.lower()


async def data_generator_zero_shot(task_description, input_format, output_format):
    prompt = DATA_GENERATOR_ZERO_SHOT_PROMPT.format(task_description=task_description, input_format=input_format, output_format=output_format)
    output_text = await chat_complete(prompt)
    match = re.search(r'\{.*\}', output_text, re.S)
    if match:
        try:
            return json.loads(match.group().strip())
        except:
            return {"input": None, "output": None}
    return {"input": None, "output": None}


async def data_generator_few_shot(task_description, input_format, output_format, examples):
    example_text = "\n\n".join(
        [f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}" for i, ex in enumerate(examples)]
    )
    prompt = DATA_GENERATOR_FEW_SHOT_PROMPT.format(
        task_description=task_description,
        input_format=input_format,
        output_format=output_format,
        example_text=example_text
    )
    output_text = await chat_complete(prompt)
    match = re.search(r'\{.*\}', output_text, re.S)
    if match:
        try:
            return json.loads(match.group().strip())
        except:
            return {"input": None, "output": None}
    return {"input": None, "output": None}
