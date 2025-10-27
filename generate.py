import logging
import asyncio
from tqdm.asyncio import tqdm_asyncio
from data_agent import data_generator_zero_shot, data_generator_few_shot
from config import TASK_DESCRIPTION, INPUT_FORMAT, OUTPUT_FORMAT, DATA_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

async def main():
    tasks = [
        data_generator_zero_shot(TASK_DESCRIPTION, INPUT_FORMAT, OUTPUT_FORMAT)
        for _ in range(DATA_SIZE)
    ]

    results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=DATA_SIZE, desc="Generating data"):
        result = await coro
        results.append(result)

    for i, result in enumerate(results, 1):
        print(f"\n--- Sample {i} ---\n{result}\n")

if __name__ == "__main__":
    asyncio.run(main())
