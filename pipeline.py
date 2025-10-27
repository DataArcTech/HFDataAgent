import os
import csv
import time
import logging
import asyncio
from tqdm import tqdm
import networkx as nx
from pyvis.network import Network
from hf_client import HFClient
from data_agent import keyword_extraction, instruction_judge, field_filter, format_conversion, data_generator_few_shot, data_generator_zero_shot
from config import HUGGINGFACE_TOKEN, TASK_DESCRIPTION, INPUT_FORMAT, OUTPUT_FORMAT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

async def hf_data_crawl(task_description, client, task_datasets_count=5, task_data_samples=5):
    # 提取任务关键词
    task_keywords = await keyword_extraction(task_description)
    logging.info(f"Extracted Task keywords: {task_keywords}")

    # 初始化 Hugging Face 客户端
    client = HFClient(hf_token=HUGGINGFACE_TOKEN)

    dataset_map = {}
    for task_keyword in tqdm(task_keywords, desc="🔍 Searching keywords", unit="keyword"):
        # Hugging Face 搜索数据集
        task_datasets = client.search_datasets(task_keyword)
        if not task_datasets:
            continue

        dataset_map[task_keyword] = []
        time.sleep(1)
        
        for task_dataset in tqdm(task_datasets[:task_datasets_count], desc=f"📦 {task_keyword}", leave=False, unit="dataset"):
            # 打印数据集基础信息
            dataset_id = task_dataset.id
            logging.info(f"Loading dataset: {dataset_id}")
            dataset_info = client.get_info(dataset_id)
            logging.info(f"Loaded dataset info for {dataset_id}")
            dataset_splits = client.get_splits(dataset_id)["splits"]
            logging.info(f"Available splits for {dataset_id}: {dataset_splits}")
            first_split = dataset_splits[0]["split"]
            first_rows = client.get_first_rows(dataset_id, split=first_split)["rows"][:task_data_samples]
            
            # 整合所有的待处理样本
            dataset_map[task_keyword].append({
                "id": dataset_id,
                "info": dataset_info,
                "rows": first_rows
            })
            
    return dataset_map
            
async def hf_data_process_sample(row, task_description, input_format, output_format):
    # 单个样本过滤和处理
    row_data = row["row"]
    legal_keys = row_data.keys()
    
    # 检查哪些字段对应Input和Output
    fields = await field_filter(str(row_data), legal_keys)
    if fields["input"] is None or fields["output"] is None:
        logging.info("Skipping row due to missing input/output fields.")
        return None
    
    # 获取原始的Input和Output文本
    input_text = row_data.get(fields["input"])
    output_text = row_data.get(fields["output"])
    if input_text is None or output_text is None:
        logging.info("Skipping row due to None input/output values.")
        return None
    
    original_sample = {
        "input": input_text,
        "output": output_text
    }
    
    # 对原始样本进行任务适应性评分，只保留每个分数大于8的
    sample_scores = await instruction_judge(task_description, str(original_sample))
    for criteria, score in sample_scores.items():
        if int(score) < 8:
            logging.info(f"Skipping row due to low score on {criteria}: {score}")
            return None
        
    # 对照任务要求的标准格式进行转换
    formatted_sample = await format_conversion(
        original_sample["input"],
        original_sample["output"],
        input_format,
        output_format
    )
    if formatted_sample.get("input") is None or formatted_sample.get("output") is None:
        logging.info("Skipping row due to None formatted input/output values.")
        return None
    
    return original_sample, sample_scores, formatted_sample

async def hf_data_process(dataset_map, task_description, input_format, output_format):
    all_tasks = []
    index_map = []

    # 打包所有任务
    for keyword, datasets in dataset_map.items():
        for dataset in datasets:
            for row in dataset["rows"]:
                task = asyncio.create_task(hf_data_process_sample(row, task_description, input_format, output_format))
                all_tasks.append(task)
                index_map.append((keyword, dataset["id"], dataset["info"]))

    logging.info(f"🚀 Launching {len(all_tasks)} async sample-processing tasks...")

    # 全部并发执行
    results = await asyncio.gather(*all_tasks)

    processed_data_map = {}
    for (keyword, dataset_id, dataset_info), result in zip(index_map, results):
        if result is None:
            continue
        original_sample, sample_scores, formatted_sample = result
        if keyword not in processed_data_map:
            processed_data_map[keyword] = []
        # 查找现有 dataset entry 或新建
        dataset_entry = next((d for d in processed_data_map[keyword] if d["id"] == dataset_id), None)
        if not dataset_entry:
            dataset_entry = {"id": dataset_id, "info": dataset_info, "samples": []}
            processed_data_map[keyword].append(dataset_entry)
        dataset_entry["samples"].append({
            "original_input": original_sample["input"],
            "original_output": original_sample["output"],
            "scores": sample_scores,
            "formatted_input": formatted_sample["input"],
            "formatted_output": formatted_sample["output"]
        })

    return processed_data_map

def save_data_csv(dataset_map, task_description):
    os.makedirs("hf_logs", exist_ok=True)
    csv_filename = os.path.join("hf_logs", "hf_data_log.csv")
    with open(csv_filename, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # 写入标题行
        writer.writerow([
            "Task_Definition", "Keyword", "Dataset_ID", 
            "Original_Input", "Original_Output", 
            "Judge_Scores", 
            "Formatted_Input", "Formatted_Output"
        ])

        # 遍历 dataset_map 生成每一行
        for kw, datasets in dataset_map.items():
            for dataset in datasets:
                dataset_id = dataset["id"]
                for sample in dataset["samples"]:
                    writer.writerow([
                        task_description, kw, dataset_id,
                        sample["original_input"], sample["original_output"],
                        str(sample["scores"]),
                        sample["formatted_input"], sample["formatted_output"]
                    ])
                    
    logging.info(f"✅ CSV 文件已生成: {csv_filename}")


if __name__ == "__main__":
    hf_client = HFClient(hf_token=HUGGINGFACE_TOKEN)
    dataset_map = asyncio.run(hf_data_crawl(TASK_DESCRIPTION, hf_client))
    processed_data_map = asyncio.run(
        hf_data_process(dataset_map, TASK_DESCRIPTION, INPUT_FORMAT, OUTPUT_FORMAT)
    )
    save_data_csv(processed_data_map, TASK_DESCRIPTION)