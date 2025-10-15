from hf_client import HFClient

client = HFClient()
results = client.search_datasets("math reasoning")
print(results)
first_id = results[0].id
print(f"First dataset ID: {first_id}")
splits = client.get_splits(first_id)
print(splits)
info = client.get_info(first_id)
print(info)
rows = client.get_first_rows(first_id)
print(rows["rows"][:1])
