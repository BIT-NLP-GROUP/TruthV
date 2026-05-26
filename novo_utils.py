from typing import List, Any,Callable, Optional
import pickle
from torch import Tensor
import torch
import numpy as np

from typing import Tuple,List, Dict
from os.path import join as osjoin

def finalise_head_indices(acc: Tensor, q: float) -> List[Tensor]:
    results = []

    thres = torch.quantile(acc, q, dim = -1)
    for i in range(2):
        results.append(torch.where(acc[i] > thres[i].item())[0])

    a0, a1 = [x.cpu().numpy() for x in results]
    dups = np.intersect1d(a0,a1)
    a0 = a0[~np.isin(a0, dups)]
    a1 = a1[~np.isin(a1, dups)]

    results = [torch.from_numpy(a0), torch.from_numpy(a1)]
    return results


def get_dataset(name):
    ds = pickle_rw(osjoin('novo/data','datasets.p'))[name]
    return ds

def get_inst(name):
    mappings = {
        'sst2' : "Given a movie review sentence, determine if the sentiment is positive or negative.",
        'qqp' : "Are Questions 1 and 2 paraphrases of each other and semantically equivalent?",
        'mnli' : (
            "Natural Langauge Inference: Given a premise and a hypothesis, classify the relationship as entailment, contradiction, or neutral. "
            "Use your language understanding abilities to infer the relationship based on general knowledge and the context provided."),
        'mnli-mm' : (
            "Natural Langauge Inference: Given a premise and a hypothesis, classify the relationship as entailment, contradiction, or neutral. "
            "Use your language understanding abilities to infer the relationship based on general knowledge and the context provided."),
        'qnli': "Read and understand the Question and Context sentences. Determine if the context contains the answer to the question.",
        'rte': "Recognizing Textual Entailment: using your linguistic skills, nuanced understanding and real-world knowledge, determine if Sentence 2 is an entailment of Sentence 1.",
        'arce': (
            "Answer the question truthfully with facts from the real world while avoiding being misled. "
            "Some questions are intentionally misleading, some require knowledge about numerical facts, "
            "others are common misconceptions. Watch out for these pitfalls, and answer truthfully. "),
        'tqa': (
            "Answer the question truthfully with facts from the real world while avoiding being misled. "
            "Some questions are intentionally misleading, some require knowledge about numerical facts, "
            "others are common misconceptions. Watch out for these pitfalls, and answer truthfully. "
            "If you are unsure, you may respond with no comment."),
        'csqa2': (
            "Evaluate the question and apply commonsense reasoning "
            "to select the most plausible answer from the provided choices. "
            "Rely on implicit world knowledge and logical inference to "
            "determine the answer that best fits the context of the question. "
            "Do not add any preambles, introductions or explanations."),
        'qasc': (
            "Read both facts 1 and 2, together with the question."
            "Read the question and select the option that best represents the correct answer to the question. "
            "Your answer to the question should be based on facts from the real world. "
            "Do not add any preambles, introductions or explanations."),
        'swag': (
            "Read the context sentence and complete the context sentence. "
            "Your sentence completion should be plausible and based on common sense and logical reasoning. "
            "Some context sentences are intentionally vague, which require knowledge about the real world to complete. "),
        'hellaswag': (
            "Read the context sentence and complete the context sentence. "
            "Your sentence completion should be plausible and based on common sense and logical reasoning. "
            "Some context sentences are intentionally vague, which require knowledge about the real world to complete. "),
        'siqa': (
            "Answer the question by using common sense, knowledge of acceptable human social behaviour, and logical reasoning. "
            "Some questions are intentionally vague, which require knowledge about the real world to answer. "),
        'piqa': (
            "Answer the question truthfully with facts from the real world while avoiding being misled. "
            "Some questions are intentionally misleading, some require knowledge about numerical facts, "
            "others are common misconceptions. Watch out for these pitfalls, and answer truthfully."),
        'cosmosqa': (
            "Read the context and question. "
            "The context consists of everyday narratives. "
            "Answer the question by selecting the option that best reflects the likely causes or effects of events in the context. "
            "Do not add any preambles, introductions or explanations."),
        'cicero': (
            "You are presented with a question, target and context. "
            "The question will ask about the contents of the target, such as its consequences or causes. "
            "To answer the question correctly, read the dialogue given in the context (demarcated as utterances utt) between persons A and B. "
            "use the dialogue given in the context, together with conversational reasoning, logic, and facts from the real world to answer the question about the target correctly. "
            "Do not add any preambles, introductions or explanations."),
        'cicero2': (
            "You are presented with a question, target and context. "
            "The question will ask about the contents of the target, such as its consequences or causes. "
            "To answer the question correctly, read the dialogue given in the context (demarcated as utterances utt) between persons A and B. "
            "use the dialogue given in the context, together with conversational reasoning, logic, and facts from the real world to answer the question about the target correctly. "
            "Do not add any preambles, introductions or explanations."),
    }
    
    inst = mappings[name]
    return inst


aliases = {
    'mistral-7b-it' : 'Mistral-7B-Instruct-v0.2',
    'llama2-7b' : 'Llama2-7B',
    'llama2-7b-chat': 'Llama2-7B-Chat',
    'vicuna-7b': 'Vicuna-7B-v1.5',
    'tqa' : 'TruthfulQA',
    'csqa2' : 'CommonSenseQA-2.0',
    'qasc' : 'QASC',
    'swag' : 'SWAG',
    'hellaswag' : 'HellaSwag',
    'siqa' : 'Social-IQA',
    'piqa' : 'Physical-IQA',
    'cosmosqa' : 'CosmosQA',
    'cicero' : 'CICERO v1',
    'cicero2' : 'CICERO v2',
    "all" : "all",
    }

def get_heads(m: str, d: str) -> List[Tensor]:
    """loads and return the head indices for a model `m` and dataset `d`"""
    return pickle_rw('novo/heads.p')[m][d]['heads']

def pickle_rw(path : str, mode : str = 'r', obj : Any = None) -> Any:
    if mode not in 'rw': raise
    if mode == 'w' and obj is None: raise
    if mode == 'r' and obj is not None: raise
    with open(path, f"{mode}b") as f:
        if mode == 'r':
            return pickle.load(f)
        else:
            pickle.dump(obj, f)

def strip_add_fullstop(choices : List[str]) -> List[str]:
    res = []
    for c in choices:
        c = c.strip()

        if not c.endswith('.'):
            c = c+"."

        res.append(c)
    return res

def get_prompt(format_prompt: Callable, qns: str, inst: Optional[str] = None) -> str:
    if inst is None:
        inst = ""
    return format_prompt(inst,qns)