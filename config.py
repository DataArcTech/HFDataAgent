LLM_API_KEY = "sk-d1468a6c93894e66a510e89b39f68e85"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"
HUGGINGFACE_TOKEN = "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"

# TASK_DESCRIPTION = "Your task is to answer CFA exam questions in a multi-choice form, you should select the correct answer choice (e.g., A, B, C). These question is about asset valuation, applying investment tools and concepts to analyze various investments, portfolio management, wealth planning, ethical and professional standards. It requires skills about Fundamental Knowledge Understanding, Quantitative Analysis and Calculations, Application and Analysis, etc."
TASK_DESCRIPTION = "You are given a word problem involving basic arithmetic, algebra, or geometry. Your task is to carefully read the problem and provide a step-by-step solution for it."

INPUT_FORMAT = "Follow this format: Read the questions and answers carefully, and choose the one you think is appropriate among the three options A, B and C.’ then Q:[Your question here] CHOICES: A: ...,B: ...,C: ..."

OUTPUT_FORMAT = "Your output thinking process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> thinking process here </think> <answer> a single option here </answer>."