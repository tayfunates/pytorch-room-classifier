import json

with open('../rooms.json') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

json_content = []
for img_json in content:
    img_json_content = json.loads(img_json)
    json_content.append(img_json_content)

