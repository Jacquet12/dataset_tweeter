import json

with open("merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    mistral = item.get("mistral")
    llama = item.get("llama")
    phi = item.get("phi")
    deepseek = item.get("deepseek")

    support = 0
    for aux in [llama, phi, deepseek]:
        if aux == mistral:
            support += 1

    item["final_sentiment"] = mistral
    item["votes_for_mistral"] = support
    item["agreement_level"] = f"{support + 1}/4"
    item["mistral_supported"] = support >= 2

with open("dataset_final.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ dataset_final.json criado!")