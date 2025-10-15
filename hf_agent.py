from utils import chat_complete

def keyword_extraction(task_description):
    prompt = f"""You can summarize the domain of this task: <{task_description}> into a few keywords.
You can refer to these task examples:

TASK_DESCRIPTION: "You are given a word problem involving basic arithmetic, algebra, or geometry. Your task is to carefully read the problem and provide a step-by-step solution for it."
KEYWORDS: ["math reasoning", "arithmetic", "algebra", "geometry"]

TASK_DESCRIPTION: "
"""