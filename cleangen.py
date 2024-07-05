import torch
import numpy as np
import copy
import logging
from colorlog import ColoredFormatter
import time


class CleanGen:
    def __init__(self, model_target, model_ref, tokenizer, alpha = 20, k = 4, back_len = 0, forward_len = 1, max_length = 1024, verbose=False):
        self.model_target = model_target
        self.model_ref = model_ref
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.k = k
        self.back_len = back_len
        self.forward_len = forward_len
        self.max_length = max_length
        self.verbose = verbose
        self.m = 2048
        logging.info("CleanGen initialized.")
    


    def decode(self, inputs, gen_config=None):

        # prepare inputs
        inputs = {k:v.cuda(self.model_target.device) for k,v in inputs.items()}
        if len(inputs['input_ids'][0].unsqueeze(0).shape) == 2:
            input_ids = inputs['input_ids'][0].unsqueeze(0)
        elif len(inputs['input_ids'][0].unsqueeze(0).shape) == 1:
            input_ids = inputs['input_ids'].unsqueeze(0)
        generated_text_ids = input_ids

        #Initialize
        count = 0
        temp_probs = []
        temp_logits = []
        reference_count = 0 
        self.model_target.eval()
        self.model_ref.eval()
        

        start_time = time.time()
        with torch.no_grad():
            for i in range(self.m):
                #check by reference model every k tokens
                if (count != 0) & (count % self.k == 0):
                    temp_probs_stack = torch.stack(temp_probs)
                    previous_logits = temp_logits
                    temp_probs = []
                    temp_logits = []
                    count = 0
       

                    outputs_ref = self.model_ref(generated_text_ids)
                    logits_ref = outputs_ref.logits
                    nexttoken_logits_ref = []
                    for guess in range(self.k):
                        # calculate suspicous score for each draft token
                        nexttoken_logits_ref.append(logits_ref[0, -self.k-1+guess, :])
                        probs_ref = torch.softmax(nexttoken_logits_ref[guess], dim=-1)
                        guess_token_indice = generated_text_ids[0][-self.k+guess]
                        suspicous_score = temp_probs_stack[guess] / probs_ref[guess_token_indice]
                        previous_probs = torch.softmax(previous_logits[guess], dim=-1) 
                        
                        # if a large suspicous score is detected, replace that token
                        if suspicous_score >= self.alpha:
                            if self.verbose:
                                logging.info("\n-----------------------------------------------")
                                logging.info("\n----------The ref model's corresponding probability--------------")
                                logging.info(f"Generation Step {generated_text_ids.shape[1] - input_ids.shape[1]}")
                                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                                logging.info("|----|----------|---------|----------|---------|")
                                for idx, (prob, token_id) in enumerate(zip(probs_ref[topk_indices_target], topk_indices_target)):
                                    token = self.tokenizer.decode(token_id.item())
                                    score = torch.log(prob)
                                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                            print(f"The suspicous score is {suspicous_score}")
                            generated_text_ids = generated_text_ids[:, 0:np.max([generated_text_ids.shape[1] - len(temp_probs_stack) + guess - self.back_len, input_ids.shape[1]])]
                            print(f"The replaced token position is {generated_text_ids.shape[1] - input_ids.shape[1]}")
                            reference_count += 1

                            # replace that drat token
                            topk_token_ref = torch.topk(probs_ref, 5)
                            topk_values_ref = topk_token_ref.values
                            topk_indices_ref = topk_token_ref.indices
                            top_tokens_indices = topk_indices_ref
                            probs_ref_softmax = torch.softmax(probs_ref[top_tokens_indices], dim=-1)
                            topk_token = torch.topk(probs_ref_softmax, len(probs_ref_softmax))
                            topk_values = topk_token.values
                            topk_indices = topk_token.indices
                            next_token = top_tokens_indices[topk_indices[0]].unsqueeze(0)
                            generated_text_ids = torch.cat([generated_text_ids, next_token.unsqueeze(0)], dim=-1)


                            if self.verbose:
                                logging("\n-----------------------------------------------")
                                logging.info("\n----------The probabilities of candidate tokens in the target model--------------")
                                logging.info(f"Generation Step {generated_text_ids.shape[1] - input_ids.shape[1]}")
                                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                                logging.info("|----|----------|---------|----------|---------|")
                                for idx, (prob, token_id) in enumerate(zip(previous_probs[top_tokens_indices], top_tokens_indices)):
                                    token = self.tokenizer.decode(token_id.item())
                                    score = torch.log(prob)
                                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                                logging.info("\n-----------------------------------------------")
                                logging.info("\n----------The probabilities of candidate tokens in the reference model--------------")
                                logging.info(f"Generation Step {generated_text_ids.shape[1] - input_ids.shape[1]}")
                                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                                logging.info("|----|----------|---------|----------|---------|")
                                for idx, (prob, token_id) in enumerate(zip(probs_ref[top_tokens_indices], top_tokens_indices)):
                                    token = self.tokenizer.decode(token_id.item())
                                    score = torch.log(prob)
                                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                                logging.info("\n-----------------------------------------------")
                                logging.info("\n----------The CleanGen decoding--------------")
                                logging.info(f"Generation Step {generated_text_ids.shape[1] - input_ids.shape[1]}")
                                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                                logging.info("|----|----------|---------|----------|---------|")
                                for idx, (prob, token_id) in enumerate(zip(topk_values, topk_indices)):
                                    token = self.tokenizer.decode(top_tokens_indices[token_id].item())
                                    score = torch.log(prob)
                                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")
                                generated_sequence = generated_text_ids[0][input_ids.shape[1]:]
                                generated_text = self.tokenizer.decode(generated_sequence)
                                logging.info(f"{generated_text}")
                            
                            break
                
                # target model decode
                if i >= 1:
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                outputs_target = self.model_target(generated_text_ids)
                logits_target = outputs_target.logits
                nexttoken_logits_target = logits_target[0, -1, :]
                temp_logits.append(nexttoken_logits_target)
                probs_target = torch.softmax(nexttoken_logits_target, dim=-1)
                topk_token_target = torch.topk(probs_target, 10)
                topk_values_target = topk_token_target.values
                topk_indices_target = topk_token_target.indices
                next_token = topk_indices_target[0].unsqueeze(0)
                count = count + 1
                temp_probs.append(topk_values_target[0])
                generated_text_ids = torch.cat([generated_text_ids, next_token.unsqueeze(0)], dim=-1)

                if self.verbose:

                    logging.info("\n-----------------------------------------------")
                    logging.info("\n----------The main model decoding--------------")
                    logging.info(f"Generation Step {generated_text_ids.shape[1] - input_ids.shape[1]}")
                    logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                    logging.info("|----|----------|---------|----------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(topk_values_1 , topk_indices_1)):
                        token = self.tokenizer.decode(token_id.item())
                        score = torch.log(prob)
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                    generated_sequence = generated_text_ids[0][input_ids.shape[1]:]
                    generated_text = self.tokenizer.decode(generated_sequence)
                    logging.info(f"{generated_text}")
                    
                
                if (generated_text_ids.shape[1] - input_ids.shape[1]) > self.max_length:
                    break 
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # print decoding results
        end_time = time.time()
        generated_sequence = generated_text_ids[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_sequence)
        print(generated_text)
        ratio = reference_count / (generated_text_ids.shape[1] - input_ids.shape[1])
        print(f"The reference model is used {reference_count} times, the ratio is {ratio}")
        average_time = (end_time - start_time) / (generated_text_ids.shape[1] - input_ids.shape[1])
        print(f"The average time for each token is {average_time}")
        logging.info(f"{generated_text}")
        

        return generated_text, ratio, average_time

    



    
    def no_defense_baseline(self, inputs):
        # if self.verbose:
        #     logging.info(f"Generation config: {gen_config}")
        inputs = {k:v.cuda(self.model_target.device) for k,v in inputs.items()}
        if len(inputs['input_ids'][0].unsqueeze(0).shape) == 2:
            input_ids = inputs['input_ids'][0].unsqueeze(0)
        elif len(inputs['input_ids'][0].unsqueeze(0).shape) == 1:
            input_ids = inputs['input_ids'].unsqueeze(0)
        generated_text_ids = input_ids
        self.model_target.cuda()
        token_probs = []
        start_time = time.time()
        with torch.no_grad():
            for _ in range(self.max_length):
                outputs = self.model_target(generated_text_ids)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                max_probs = torch.max(next_token_probs)
                token_probs.append(max_probs.item())
                generated_text_ids = torch.cat([generated_text_ids, next_token], dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        end_time = time.time()
        generated_sequence = generated_text_ids[0]
        generated_text = self.tokenizer.decode(generated_sequence[-(generated_text_ids.shape[1] - input_ids.shape[1]):])
        average_time = (end_time - start_time) / (generated_text_ids.shape[1] - input_ids.shape[1])
        print(f"The average time for each token is {average_time}")
        logging.info(f"{generated_text}")
 
        
        return generated_text, 0, average_time