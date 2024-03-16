import pandas as pd
from torch.utils.data import Dataset
from utils.file_utils import load_jsonl
from phi.phi_utils.constants import PHI_ZERO_SHOT_EVAL_PROMPT, PHI_FEW_SHOT_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_PROMPT
import random

class PhiPromptDataset(Dataset):
    def __init__(self, annotations_filepath, prompt_type, evidence_filepath = None):
        self.data = load_jsonl(annotations_filepath)
        self.prompt_type = prompt_type

        if evidence_filepath is not None: 
            self.evidence_data = load_jsonl(evidence_filepath)
        else:
            self.evidence_data = None

    def __len__(self):
        return len(self.data)

    ############################################################
    # TODO: Please complete the implementation for the
    # the following transform functions and __getitem__ fn, that you 
    # will use in def __getitem__ to convert a sample into prompt.
    # You can use the templates provided to in the constants.py file

    def zero_shot_eval_prompt_transform(self, idx) -> str:
        return PHI_ZERO_SHOT_EVAL_PROMPT.format(claim = self.data[idx]["claim"], task_type = self.data[idx]["task_type"])
    
    
    """
    Format for evidence:
    "Example 1:\nClaim: And all those holes below 40,000 feet are filled with oil instead of water .\nIs the claim fair? \nLabel: SUPPORTS;;;\n Example 2:..."
    Need per example: Claim, task_type, label
    
    {
        climate: "examplesForDomainIn1Str",
        dom2: ""
    }
    """
    
    
    
    def few_shot_eval_prompt_transform(self, idx) -> str: #TODO: generate examples - cache from training dataset
        
        #From training set, probability that each domain-task_type pair is support
        probSupport = {"climate-fact": 0.72, "hsd-fairness": 0.5, "sbic-fairness": 0.28, "mgfn-fact": 0.5, "toxigen-fairness": 0.57, "health-fact": 0.61}
        
        self.examples = load_jsonl("gen_examples.jsonl")
        
        num_examples = 6
        # print(len(self.examples[0].keys()))
        
        k = self.data[idx]["domain"] + "-" + self.data[idx]["task_type"]
        kSup = k + "-SUPPORTS"
        kRef = k + "-REFUTES"
        supAllExamples = self.examples[0][kSup]
        refAllExamples = self.examples[0][kRef]
        
        num_sup = int(num_examples * probSupport[k])
        num_ref = num_examples - num_sup
        
        supExamples = random.sample(supAllExamples, num_sup)
        refExamples = random.sample(refAllExamples, num_ref)
        
        finExamples = supExamples + refExamples
        random.shuffle(finExamples)

        examplesStr = ""
        for i in range(num_examples):
            examplesStr += "Example " + str(i+1) + ":\n" + finExamples[i]
        # print(examplesStr)
        return PHI_FEW_SHOT_EVAL_PROMPT.format(examples = examplesStr, claim = self.data[idx]["claim"], task_type = self.data[idx]["task_type"])
    
    # def zero_shot_evidence_prompt_transform(self, idx) -> str: #TODO: NOT DONE
    #     return PHI_ZERO_SHOT_EVIDENCE_PROMPT.format(claim = self.data[idx]["claim"], information = "N/A")
    
    # def zero_shot_evidence_eval_prompt_transform(self, idx) -> str:
    #     return PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT.format(claim = self.data[idx]["claim"], evidence = self.evidence_data[idx]["evidence_sample"], task_type = self.data[idx]["task_type"])
    

    # End of TODO.
    ##################################################
    
    def __getitem__(self, idx):

        prompt = ""
        
        ##################################################
        # TODO: Please complete the implementation of __getitem__
        # You may use if-else statements to choose the prompt
        # transform as per the prompt type given to you.
    
        if self.prompt_type == "zero_eval":
            prompt = self.zero_shot_eval_prompt_transform(idx)
        elif self.prompt_type == "few_eval":
            prompt = self.few_shot_eval_prompt_transform(idx)
        elif self.prompt_type == "zero_evidence":
            prompt = self.zero_shot_evidence_prompt_transform(idx)
        elif self.prompt_type == "zero_evidence_eval":
            prompt = self.zero_shot_evidence_eval_prompt_transform(idx)
        
        # End of TODO.
        ##################################################
        
        return prompt
    