import json

with open('intent.json', encoding="utf8") as f:
  data = json.load(f)

print(data)