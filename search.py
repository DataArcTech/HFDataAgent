import time
import logging
from tqdm import tqdm
import networkx as nx
from pyvis.network import Network
from hf_client import HFClient
from hf_agent import keyword_extraction, instruction_judge, field_filter, solvable_judge
from config import HUGGINGFACE_TOKEN, TASK_DESCRIPTION

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# -------------------------------
# 1️⃣ Hugging Face 数据集抓取逻辑
# -------------------------------
client = HFClient(hf_token=HUGGINGFACE_TOKEN)
task_keywords = keyword_extraction(TASK_DESCRIPTION)
print("Extracted Task keywords:", task_keywords)

dataset_map = {}  # {keyword: [ {id, info, rows}, ... ] }

# 外层进度条：关键词级别
for task_keyword in tqdm(task_keywords, desc="🔍 Searching keywords", unit="keyword"):
    task_datasets = client.search_datasets(task_keyword)
    if not task_datasets:
        continue

    dataset_map[task_keyword] = []
    time.sleep(1)  # 小延迟避免过快请求

    # 内层进度条：数据集级别
    for task_dataset in tqdm(task_datasets[:3], desc=f"📦 {task_keyword}", leave=False, unit="dataset"):
        dataset_id = task_dataset.id
        logging.info(f"Loading dataset: {dataset_id}")
        dataset_info = client.get_info(dataset_id)
        logging.info(f"Loaded dataset info for {dataset_id}")
        dataset_splits = client.get_splits(dataset_id)["splits"]
        logging.info(f"Available splits for {dataset_id}: {dataset_splits}")
        first_split = dataset_splits[0]["split"]
        first_rows = client.get_first_rows(dataset_id, split=first_split)["rows"][:5]

        samples = []
        for row in first_rows:
            row = row["row"]
            print("Processing row:", row)
            legal_keys = row.keys()
            fields = field_filter(str(row), legal_keys)
            print("Filtered fields:", fields)
            if fields["input"] is None or fields["output"] is None:
                print("Skipping row due to missing input/output fields.")
                continue
            
            input_text = row.get(fields["input"])
            output_text = row.get(fields["output"])
            if input_text is None or output_text is None:
                print("Skipping row due to None input/output values.")
                continue

            sample = {
                "input": input_text,
                "output": output_text
            }            

            sample_score = instruction_judge(TASK_DESCRIPTION, str(sample))
            # sample_solvable = solvable_judge(sample)
            sample["score"] = sample_score
            # sample["solvable"] = sample_solvable
            sample["solvable"] = "Unknown"
            samples.append(sample)
        
        dataset_map[task_keyword].append({
            "id": dataset_id,
            "info": dataset_info,
            "samples": samples
        })


# -------------------------------
# 2️⃣ 构建知识图谱
# -------------------------------
G = nx.Graph()

# 添加 task 节点
G.add_node(
    "task",
    label=TASK_DESCRIPTION,
    color="#FFAA00",
    shape="ellipse",
    size=50,  # 加大主任务节点
    physics=True
)

# 为每个 keyword 添加节点并连接 task
for kw in task_keywords:
    kw_node = f"kw::{kw}"
    G.add_node(
        kw_node,
        label=kw,
        color="#00CCFF",
        shape="box",
        size=30  # 关键字节点稍大
    )
    G.add_edge("task", kw_node)

    # 若该 keyword 有匹配到数据集
    if kw in dataset_map:
        for dataset in dataset_map[kw]:
            dataset_id = dataset["id"]
            dataset_info = dataset["info"]

            # 添加 dataset 节点
            ds_node = f"ds::{dataset_id}"
            G.add_node(
                ds_node,
                label=dataset_id,
                color="#33FF66",
                title=f"Info: {dataset_info}",
                size=20
            )
            G.add_edge(kw_node, ds_node)

            # 添加 dataset samples 节点
            for i, sample in enumerate(dataset["samples"]):
                sample_node = f"sample::{dataset_id}::{i}"
                sample_preview = str(sample)[:200]
                G.add_node(
                    sample_node,
                    label=f"Sample {i+1}",
                    title=sample_preview,
                    color="#AAAAAA",
                    size=8
                )
                G.add_edge(ds_node, sample_node)


# -------------------------------
# 3️⃣ 绘制交互式图谱
# -------------------------------
net = Network(
    height="1600px",
    width="100%",
    bgcolor="#222222",
    font_color="white"
)

net.from_nx(G)
net.force_atlas_2based()
net.show("dataset_graph.html", notebook=False)

print("✅ 图谱已生成: dataset_graph.html")


import csv

# -------------------------------
# 4️⃣ 导出结果到 CSV
# -------------------------------
csv_filename = "dataset_summary.csv"

with open(csv_filename, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    # 写入标题行
    writer.writerow(["Task_Definition", "Keyword", "Dataset_ID", "Sample_Input", "Sample_Output", "Judge_Score", "Judge_Solvable"])

    # 遍历 dataset_map 生成每一行
    for kw, datasets in dataset_map.items():
        for dataset in datasets:
            dataset_id = dataset["id"]
            for sample in dataset["samples"]:
                writer.writerow([TASK_DESCRIPTION, kw, dataset_id, sample["input"], sample["output"], sample["score"], sample["solvable"]])

print(f"✅ CSV 文件已生成: {csv_filename}")