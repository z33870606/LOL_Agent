import json

with open("champions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

champion_names = [champ["name"] for champ in data]

print(champion_names)