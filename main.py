from novo_utils import get_dataset, strip_add_fullstop, pickle_rw, aliases, get_inst
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json

mlp_activations = {}

def load_hook(model, args):
    
    if 'gemma-3' in args.model_path:
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
    for i, block in enumerate(layers):
        mlp = block.mlp  # LlamaMLP 层
        original_forward = mlp.forward
        def wrapped_forward(self, x, layer_index=i):
            activations = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            # activations[0, -1, :].detach()
            mlp_activations[f"layer_{layer_index}"] = activations[:, -1, :]
            down_proj = self.down_proj(activations)
            return down_proj
        mlp.forward = wrapped_forward.__get__(mlp, mlp.__class__)

def model_forward(model, tokenizer, inst, question, choices, mlp_activations=None, args=None):

    def get_prompt(choice):
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
            
        return tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt')[:, :-2]
    activations = []
    for choice in choices:
        choice_act = []
        prompt = get_prompt(choice)
        # content = inst+'\n'+question+'\n'+choice
        # prompt = tokenizer(content, return_tensors='pt').input_ids
        model(prompt.to(model.device))
        for key, value in mlp_activations.items():
            choice_act.append(value[0])
        choice_act = torch.stack(choice_act, dim=0)
        activations.append(choice_act)
    activations = torch.stack(activations, dim=0)
    activations = activations.flatten(1)

    return activations

def stat_activations(samples, model, tokenizer, args):
    if 'gemma-3' in args.model_path:
        num_hidden_layers = model.config.text_config.num_hidden_layers
        intermediate_size = model.config.text_config.intermediate_size
    else:
        num_hidden_layers = model.config.num_hidden_layers
        intermediate_size = model.config.intermediate_size
    acc_arr = torch.zeros((num_hidden_layers*intermediate_size), device=args.device)
    for d in tqdm(samples):
        choices = strip_add_fullstop(d['choices'])
        with torch.no_grad():
            activations = model_forward(model, tokenizer, d['inst'], d['question'], choices, mlp_activations=mlp_activations, args=args)
        if args.method == 'max':
            activations = activations.argmax(0)
        elif args.method == 'min':
            activations = activations.argmin(0)
        else:
            raise NotImplementedError()
        # [choice, num_layer * inter_size]
        acc_arr += (activations == d['label']).int()
    return acc_arr


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    load_hook(model, args)
        
    if args.save_path is not None:
        save_path = f"{args.save_path}/{args.model_path.split('/')[-1]}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = f"{save_path}/{args.dataset}"
    # use llama2-7b-chat as defalut
    tot_samples = []
    def prepare_dataset(inst, samples):
        for item in samples:
            tot_samples.append({
                'inst': inst,
                'question': item['question'],
                'choices': item['choices'],
                'label': item['label'],
            })
    samples = pickle_rw('novo/heads.p')['llama2-7b-chat'][args.dataset]['discovery_samples']
    inst = get_inst(args.dataset)
    prepare_dataset(inst, samples)
    stat_result = stat_activations(tot_samples, model, tokenizer, args)
    stat_result /= len(tot_samples)
    ranked_acc, indx = torch.sort(stat_result, descending=True)
    print(ranked_acc[:args.num_print])
    print(indx[:args.num_print])

    if args.save_path is not None:
        torch.save(indx, f'{save_path}_indx.pt')
    indx = indx.cpu().numpy()

    acc = {}
    dataset = get_dataset(args.dataset)
    inst = get_inst(args.dataset)
    for d in tqdm(dataset):
        choices = strip_add_fullstop(d['choices'])
        with torch.no_grad():
            activations = model_forward(model, tokenizer, inst, d['question'], choices, mlp_activations=mlp_activations, args=args)
        if args.method == 'max':
            activations = activations.argmax(0)
        elif args.method == 'min':
            activations = activations.argmin(0)
        else:
            raise NotImplementedError()

        pred = {}
        maxp = 300
        if args.save_path is None:
            maxp = 100

        for k in range(1, maxp, 1):
            p = k / 10000
            intp = int(p * activations.size(-1))
            a0 = indx[:intp]
            individual_preds = activations[a0]
            pred[f'p_{p}'] = torch.mode(individual_preds).values.item()
        
        for key, value in pred.items():
            if key not in acc:
                acc[key] = 0.0
            acc[key] += int(value == d['label'])
    for key, value in acc.items():
        acc[key] /= len(dataset)
    for key, value in acc.items():
        print(f"{args.model_path.split('/')[-1]} | {aliases[args.dataset]} | mode: {key} | Accuracy {value:.2%}")
    if args.save_path is not None:
        with open(f'{save_path}_result.json', 'w') as fout:
            json.dump(acc, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', choices=['tqa','csqa2','qasc','swag','hellaswag','siqa','piqa','cosmosqa','cicero','cicero2','arc'])
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--num_print', type=int, default=20)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)