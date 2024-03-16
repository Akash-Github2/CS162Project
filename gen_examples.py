from utils.file_utils import load_jsonl
import json


data = load_jsonl("data162/train_claims.jsonl")
# print(data)

"""
Baseline for zero-shot:
Overall Accuracy : 0.715119105493437
Overall F1 score : 0.6647597254004577

TODO: FOR FEW_SHOT
use part of the training set (200 or less) as the training set and the rest as the validation set and test on that and record the
performance metrics - also have certain number of support and refute for each category that matches with how much the category already sees
- need to experiment with which examples u choose to add per claim
"""

domains = set()
task_types = set()
langs = set()
for i in range(len(data)):
    if data[i]["domain"] not in domains:
        domains.add(data[i]["domain"])
    if data[i]["task_type"] not in task_types:
        task_types.add(data[i]["task_type"])
        
    if data[i]["language_generated"] not in langs:
        langs.add(data[i]["language_generated"])
        
print(domains)
print(task_types)
print(langs)

m = {}
exampleCnt = {} #maps key to example count

"""
    Format for evidence:
    "Example 1:\nClaim: And all those holes below 40,000 feet are filled with oil instead of water .\nIs the claim fair? \nLabel: SUPPORTS;;;\n Example 2:..."
    
    Fairness: Is the claim fair?
    Fact: Is the claim a fact?
    
    Need per example: Claim, task_type, label
    
    {
        climate: "examplesForDomainIn1Str",
        dom2: ""
    }
    """

for i in range(len(data)):
    k = data[i]["domain"] + "-" + data[i]["task_type"] + "-" + data[i]["label"]
    if k not in m:
        exampleCnt[k] = 1
        m[k] = []
    # if exampleCnt[k] >= 12:
    #     continue
    isFairTT = data[i]["task_type"] == "fairness"
    
    newExample = "Claim: " + data[i]["claim"] + "\n"
    if isFairTT:
        newExample += "Is the claim fair?\n"
    else:
        newExample += "Is the claim factual?\n"
        
    newExample += "Label: " + data[i]["label"] + "\n"
    m[k].append(newExample)
    
    exampleCnt[k] += 1

# print(m)
print(exampleCnt)
# health-fairness-human: "examplestr"


"""
climate-fact-support: 0.72
hsd-fairness-support: 0.5
sbic-fairness-support: 0.28
mgfn-fact-support: 0.5
toxigen-fairness-support: 0.57
health-fact-support: 0.61
"""

with open('gen_examples.jsonl', 'w') as file:
    json.dump(m, file)
    file.write('\n')