from novo_utils import get_dataset, strip_add_fullstop, aliases, get_inst
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import json

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1).to(logits.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.sum()


def model_forward(model, tokenizer, inst, question, choices):
    def get_prompt(choice, with_ans=True):
        if 'gemma-2' not in args.model_path:
            messages = [
                {'role': 'system', 'content': inst},
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': choice}
            ]
        else:
            # gemma
            messages = [
                {'role': 'user', 'content': f'{inst}\n\n{question}'},
                {'role': 'assistant', 'content': choice}
            ]
        if with_ans:
            return tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt')[:, :-2]
        else:
            return tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_tensors='pt')

    log_likelihood = []
    only_q = get_prompt(choice='', with_ans=False)
    for choice in choices:
        prompt = get_prompt(choice)
        with torch.no_grad():
            logits = model(prompt.to(model.device)).logits
        logits = logits[:, only_q.size(-1):-1, :]
        log_likelihood.append(get_likelihood(logits=logits, labels=prompt[:, only_q.size(-1)+1:]))
    log_likelihood = torch.stack(log_likelihood)
    pred = log_likelihood.argmax(0).item()
    return pred
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model.eval()
    
    dataset = get_dataset(args.dataset)
    inst = get_inst(args.dataset)

    save_path = f"{args.save_path}/{args.model_path.split('/')[-1]}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = f"{save_path}/{args.dataset}"

    acc = 0
    for d in tqdm(dataset):
        choices = strip_add_fullstop(d['choices'])
        pred = model_forward(model, tokenizer, inst, d['question'], choices)
        acc += int(pred == d['label'])

    acc /= len(dataset)
    print(f"{args.model_path.split('/')[-1]} | {aliases[args.dataset]} | Accuracy {acc:.2%}")

    results = {'acc': acc}

    with open(f'{save_path}_result.json', 'w') as fout:
        json.dump(results, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', choices=['tqa','csqa2','qasc','swag','hellaswag','siqa','piqa','cosmosqa','cicero','cicero2'])
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
    
