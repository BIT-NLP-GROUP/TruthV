from novo_utils import get_dataset, strip_add_fullstop, pickle_rw, aliases, finalise_head_indices, get_inst
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import Callable
import os
import json

novo_norms = {}

def load_novo_hook(model, args):
    def norm_store(attn_output, layer_index):
        #attn_output: torch.Size([bsz, seqlen, headnum, sub_hidden_size)
        head_norms = torch.linalg.norm(attn_output,dim=-1)
        head_norms = head_norms[0, -1, :].detach()
        novo_norms[f"layer_{layer_index}"] = head_norms
    if 'gemma-3' in args.model_path:
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
        
    for i, block in enumerate(layers):
        self_attn = block.self_attn
        original_forward = self_attn.forward
        if "gemma-2" in args.model_path:
            import transformers.models.gemma2.modeling_gemma2 as modeling_gemma2
            def gemma2_forward(
                self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position=None, layer_index=i, **kwargs,):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = modeling_gemma2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "sliding_window": self.sliding_window,}
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                    # Here we need to slice as we use a static cache by default, but FA2 does not support it
                    if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
                        seq_len = attention_mask.shape[-1]
                        key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

                attention_interface: Callable = modeling_gemma2.eager_attention_forward
                if self.config._attn_implementation != "eager":
                    if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                        modeling_gemma2.logger.warning_once(
                            "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                            'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                        )
                    else:
                        attention_interface = modeling_gemma2.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=self.attention_dropout if self.training else 0.0,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    softcap=self.attn_logit_softcapping,
                    **kwargs,
                )
                
                norm_store(attn_output=attn_output, layer_index=layer_index)

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights
        
            self_attn.forward = gemma2_forward.__get__(self_attn, self_attn.__class__)
        
        elif 'gemma-3' in args.model_path:
            import transformers.models.gemma3.modeling_gemma3 as modeling_gemma3
            def gemma3_forward(
                self, hidden_states, position_embeddings, attention_mask, past_key_value = None, cache_position = None, layer_index=i, **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)

                cos, sin = position_embeddings
                query_states, key_states = modeling_gemma3.apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                        "sliding_window": self.sliding_window,
                    }
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                    # Here we need to slice as we use a static cache by default, but FA2 does not support it
                    if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
                        seq_len = attention_mask.shape[-1]
                        key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

                attention_interface: Callable = modeling_gemma3.eager_attention_forward
                if self.config._attn_implementation != "eager":
                    if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                        modeling_gemma3.logger.warning_once(
                            "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                            "Falling back to eager attention. This warning can be removed using the argument "
                            '`attn_implementation="eager"` when loading the model.'
                        )
                    else:
                        attention_interface = modeling_gemma3.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                if attention_mask is not None:
                    # backwards compatibility
                    attention_mask = attention_mask.to(query_states)
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=self.attention_dropout if self.training else 0.0,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )

                norm_store(attn_output=attn_output, layer_index=layer_index)

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights
            self_attn.forward = gemma3_forward.__get__(self_attn, self_attn.__class__)
        elif 'Llama' in args.model_path:
            import transformers.models.llama.modeling_llama as modeling_llama
            def llama_forward(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position=None, layer_index=i, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                
                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                attention_interface: Callable = modeling_llama.eager_attention_forward
                if self.config._attn_implementation != "eager":
                    if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                        modeling_llama.logger.warning_once(
                            "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                            'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                        )
                    else:
                        attention_interface = modeling_llama.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
                )

                norm_store(attn_output=attn_output, layer_index=layer_index)

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights

            self_attn.forward = llama_forward.__get__(self_attn, self_attn.__class__)
        elif 'Qwen3' in args.model_path:
            from transformers.models.qwen3 import modeling_qwen3

            def qwen3_forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, layer_index=i, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                attention_interface: Callable = modeling_qwen3.eager_attention_forward
                if self.config._attn_implementation != "eager":
                    if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                        modeling_qwen3.logger.warning_once(
                            "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                            'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                        )
                    else:
                        attention_interface = modeling_qwen3.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,  # diff with Llama
                    **kwargs,
                )

                norm_store(attn_output=attn_output, layer_index=layer_index)

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights

            self_attn.forward = qwen3_forward.__get__(self_attn, self_attn.__class__)

        else:
            raise ValueError('No implementation.')
        


def model_forward(model, tokenizer, inst, question, choices, indices, return_scores=False, novo_norms=None, args=None):
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
    

    head_norms = []
    for choice in choices:
        choice_act = []
        prompt = get_prompt(choice)
        # content = inst+'\n'+question+'\n'+choice
        # prompt = tokenizer(content, return_tensors='pt').input_ids
        model(prompt.to(model.device))
        for key, value in novo_norms.items():
            choice_act.append(value)
        choice_act = torch.stack(choice_act, dim=0)
        head_norms.append(choice_act)
    head_norms = torch.stack(head_norms, dim=0)
    head_norms = head_norms.flatten(1)
    
    if return_scores:
        return head_norms

    individual_preds = torch.cat([
        head_norms[:,indices[0]].argmax(0),
        head_norms[:,indices[1]].argmin(0)])
        
    pred = torch.mode(individual_preds).values.item()
    return pred

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    load_novo_hook(model, args)
    
    dataset = get_dataset(args.dataset)
    save_path = f"{args.save_path}/{args.model_path.split('/')[-1]}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = f"{save_path}/{args.dataset}"

    # defalut: llama2-7b-chat
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

    acc_arr = torch.zeros((2,model.config.num_hidden_layers*model.config.num_attention_heads), device=args.device)
    for d in tqdm(tot_samples):
        choices = strip_add_fullstop(d['choices'])
        with torch.no_grad():
            head_norms = model_forward(model, tokenizer, d['inst'], d['question'], choices, indices=None, return_scores=True, novo_norms=novo_norms, args=args)
        acc_arr[0] += (head_norms.argmax(0) == d['label']).int()
        acc_arr[1] += (head_norms.argmin(0) == d['label']).int()
    acc_arr = (acc_arr/len(samples))*100
    heads = finalise_head_indices(acc_arr, args.quantile_threshold)
    
    acc = 0
    for d in tqdm(dataset):
        choices = strip_add_fullstop(d['choices'])
        pred = model_forward(model, tokenizer, inst, d['question'], choices, indices=heads, novo_norms=novo_norms, args=args)
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
    parser.add_argument('--quantile_threshold', type=float, default=0.85)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)
    
