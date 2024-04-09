import numpy as np
import pandas as pd
import re
import time
import gc
import torch.nn as nn
import torch
from typing import Tuple, List


from ocr.custom_ocr.model_block import PositionalEncoding, ImageEncoder, TextDecoder
from ocr.custom_ocr.utils import get_mask_seq_cat

class OCR_Model(nn.Module):
    def __init__(self, char_model, config, device, batch_size=32):
        super(OCR_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device
        self.batch_size = batch_size

        self.random_masking = self.config.random_masking
        self.distillation = self.config.distillation
        self.aux_ctc = self.config.aux_ctc
        self.beam_size = 4
        self.w = 0.5
        self.w2 = 5 
        
        reformer_n_chars = self.char_model.n_chars
        out_vocab_size = self.char_model.vocab_size
        char_embedding = nn.Embedding(reformer_n_chars, config.char_embedding_dim)
        pos_encoding_char_dim = PositionalEncoding(config.char_embedding_dim)
        self.image_encoder = ImageEncoder(pos_encoding_char_dim, config, device, out_vocab_size)
        self.text_decoder = TextDecoder(out_vocab_size, char_embedding, pos_encoding_char_dim, config, device)

    def get_null_indices_and_probabilities(self, class_data, null_class_index=0):
        class_probabilities = torch.softmax(class_data, dim=1)
        highest_prob_classes = torch.argmax(class_probabilities, dim=1)

        null_class_indices = (highest_prob_classes == null_class_index).nonzero(as_tuple=True)[0]
        null_class_probabilities = class_probabilities[null_class_indices]
        return null_class_indices, null_class_probabilities
    
    def generate(self, image_data, search_type="greedy"):
        bs = image_data.shape[0]
        
        cls_classification, cls_embedding, img2txt_enc_char = self.image_encoder(torch.from_numpy(image_data).to(self.device))
        null_indices, null_probabilites = self.get_null_indices_and_probabilities(cls_classification)
        null_indices = np.array([])
        non_null_idx = [i for i in range(bs) if i not in null_indices]
        
        # img2txt_dec_txt, confidence = self.generate_decoder_batch(img2txt_enc_char[non_null_idx],
        #                                         self.char_model.char2index["TSOS"],
        #                                         self.char_model.char2index["TEOS"],
        #                                         self.config.max_char_len, 
        #                                         cls_embedding=cls_embedding[non_null_idx])
        img2txt_dec_txt, confidence = self.generate_decoder_batch_beam_torch(img2txt_enc_char[non_null_idx],
                                                self.char_model.char2index["TSOS"],
                                                self.char_model.char2index["TEOS"],
                                                self.config.max_char_len, 
                                                cls_embedding=cls_embedding[non_null_idx])
        if len(null_indices) == bs:
            char_seq_len = 3
        else:
            char_seq_len = max(item.shape[-1] for item in img2txt_dec_txt)

        null_fields = torch.ones((len(null_indices), char_seq_len)).long()

        null_fields[:, 0] = self.char_model.char2index["TSOS"]
        null_fields[:, 1] = self.char_model.char2index[" "]
        null_fields[:, 2] = self.char_model.char2index["TEOS"]
        null_fields[:, 3:char_seq_len] = self.char_model.char2index["PAD"]
        
        final_result = [0 for _ in range(bs)]
        final_confidence = [0 for _ in range(bs)]
        
        for index in null_indices:
            final_result[index] = null_fields[index]
            
        if not len(null_indices) == bs:
            j = 0
            for i in range(bs):
                if i not in null_indices:
                    final_result[i] = img2txt_dec_txt[j]
                    final_confidence[i] = torch.round(torch.exp(torch.min(confidence[j])), decimals=2).item()
                    j += 1
                else:
                    final_confidence[i] = null_probabilites[i]
        
        return final_result, final_confidence

    def generate_decoder_batch(self, memory, sos_idx, eos_idx, max_char_len, cls_embedding=None):

        first_seq_len = memory.shape[1]
        bs = memory.shape[0]
        result = torch.ones(bs, 1).long().to(self.device) * sos_idx
        batch_probs = [[1.0] for _ in range(bs)]

        if cls_embedding is not None:
            first_seq_len = first_seq_len + 1

        for i in range(max_char_len):
            has_eos = torch.tensor([eos_idx in result[j][:i+1] for j in range(bs)])
            
            if torch.all(has_eos):
                break
            
            active_indices = torch.where(has_eos,torch.zeros(bs, dtype=torch.bool), torch.ones(bs, dtype=torch.bool))
            active_indices = active_indices.nonzero(as_tuple=True)[0].long()
            result_ = result[active_indices]
            
            tgt = result_
            tgt_mask = get_mask_seq_cat(first_seq_len, i+1).to(self.device)
            
        
            output = self.text_decoder(memory[active_indices], tgt, tgt_mask, cls_embedding[active_indices].unsqueeze(1), first_seq_len+i)
            
            values, indices = output.max(dim=-1)
            indices = indices.unsqueeze(-1)
            result_ = torch.cat([result_, indices],dim=-1)
            
            padding = torch.ones(bs, 1, dtype=torch.int64).to(self.device) * eos_idx
            result = torch.cat((result, padding), dim=1).to(self.device)

            values = values.detach().cpu().numpy()
            for j, p in zip(active_indices, values):
                batch_probs[j].append(p)
                
            result[active_indices] = result_
            
        return result, np.around([np.min(probs) for probs in batch_probs], decimals=2).tolist()

    def generate_decoder_batch_beam_torch(self, memory, sos_idx, eos_idx, max_char_len, cls_embedding=None):
        bs = memory.shape[0]
        # print("No of non-null fields:", bs)

        first_seq_len = memory.shape[1] + (1 if cls_embedding is not None else 0)

        #  Run model once to get initial beam of (bs,beam_size)
        current_beam, current_logits = self._initialize_beam(memory, cls_embedding, first_seq_len, sos_idx)
        current_beam_confidence = current_logits * self._length_penalty(2)

        # track total no of active beams for each input in batch.
        # Initial value is beam_size for each input
        active_no_of_beam_in_input = torch.ones(bs, dtype=torch.int) * self.beam_size

        completed_beams = {i: [] for i in range(bs)}
        completed_beams_logits = {i: [] for i in range(bs)}
        completed_beams_confidence = {i: [] for i in range(bs)}
        
        beam_padding, logits_padding, confidence_padding, completed_beam_start_idx = self._initialize_beam_padding(self.beam_size, max_char_len, self.device)
        
        for t in range(1,max_char_len+1):

            if torch.all(active_no_of_beam_in_input == 0):
                break
            # Get inputs from batch that has active beams and flatten them. (bs,no of active beams, seq_len) -> (bs*no of active beams, seq_len)
            active_input_idx = (active_no_of_beam_in_input > 0).nonzero(as_tuple=True)[0]
            active_input = current_beam[active_input_idx]
            active_input = active_input.clone().reshape(-1, active_input.shape[-1])
            
            active_confidence = current_beam_confidence[active_input_idx]
            active_confidence = active_confidence.reshape(-1, active_confidence.shape[-1])

            # Completed beams are stored in completed_beams and completed_beams_logits
            # Then they are replaced with zeros. Remove these inputs
            completed_beam_idx = torch.all(active_input == 0, dim=1)
            # print(active_input.shape, completed_beam_idx.shape)
            active_input = active_input[~completed_beam_idx]
            active_confidence = active_confidence[~completed_beam_idx]

            # Get image encoder input memory in same way as active_input
            image_encoder_input_memory = memory[active_input_idx]
            # image_encoder_input_memory = image_encoder_input_memory.repeat_interleave(beam_size, dim=0)
            image_encoder_input_memory = image_encoder_input_memory[:,None,:,:].expand(-1, self.beam_size, -1, -1).contiguous().view(-1, image_encoder_input_memory.shape[-2], image_encoder_input_memory.shape[-1])
            image_encoder_input_memory = image_encoder_input_memory[~completed_beam_idx]

            # Get cls_embedding input memory in same way as active_input
            input_cls_embedding = cls_embedding[active_input_idx]
            # input_cls_embedding = input_cls_embedding.repeat_interleave(beam_size, dim=0)
            input_cls_embedding = input_cls_embedding[:,None,:].expand(-1, self.beam_size, -1).contiguous().view(-1, input_cls_embedding.shape[-1])
            input_cls_embedding = input_cls_embedding[~completed_beam_idx]
            
            logits, penalty = self._model_step(image_encoder_input_memory, active_input, input_cls_embedding, first_seq_len, t)
            
            beam_start_idx = 0 #To track the start index of each input in batch since they are flattened and some inputs are removed or has active beams less than beam_size
            next_batch_inputs = []
            next_batch_logits = []
            next_batch_confidence = []
            
            completed_beam_start_idx *= 0

            for i in range(bs):
                if active_no_of_beam_in_input[i] == 0:
                    # Added to make sure that final output of the loop is of size (bs, beam_size, seq_len)
                    # If current input has all beams completed, then add zeros to next_batch_inputs and -inf to next_batch_logits 
                    # so that we can remove them on start of next loop
                    next_batch_inputs.append(beam_padding[:,:current_beam.shape[-1]+1].clone())
                    next_batch_logits.append(logits_padding.clone())
                    next_batch_confidence.append(confidence_padding[:,:current_beam_confidence.shape[-1]+1].clone())
                    continue
                
                topk_logits, topk_indices = torch.topk(logits[beam_start_idx:beam_start_idx+active_no_of_beam_in_input[i]], self. beam_size, dim=1)
                # select active beams from current_beam and add topk_indices to them
                current_batch_active_input = active_input[beam_start_idx: beam_start_idx + active_no_of_beam_in_input[i]]

                current_batch_active_input = current_batch_active_input[:,None,:].expand(-1, self. beam_size, -1).contiguous().view(-1, current_batch_active_input.shape[-1])
                current_batch_active_input = torch.cat((current_batch_active_input, topk_indices.view(-1, 1)), dim=1)
                current_batch_active_input = current_batch_active_input.reshape(-1, current_batch_active_input.shape[-1])
                
                current_batch_active_logits = current_logits[i]
                completed_beam_idx_slice = completed_beam_idx[completed_beam_start_idx:completed_beam_start_idx+4]
                completed_beam_idx_slice = torch.nonzero(~completed_beam_idx_slice, as_tuple=True)[0]

                current_batch_active_logits = current_batch_active_logits[completed_beam_idx_slice]

                current_batch_active_logits = current_batch_active_logits[:,None,:].expand(-1, self. beam_size, 1).contiguous().view(-1, current_batch_active_logits.shape[-1])
                current_batch_active_logits.add_(topk_logits.view(-1,1))
                current_batch_active_logits = current_batch_active_logits.reshape(-1)

                current_batch_active_confidence = active_confidence[beam_start_idx: beam_start_idx + active_no_of_beam_in_input[i]]
                current_batch_active_confidence = current_batch_active_confidence[:,None,:].expand(-1, self. beam_size, -1).contiguous().view(-1, current_batch_active_confidence.shape[-1])
                current_batch_active_confidence = torch.cat((current_batch_active_confidence, (topk_logits.view(-1, 1) * penalty)), dim=1)
                current_batch_active_confidence = current_batch_active_confidence.reshape(-1, current_batch_active_confidence.shape[-1])
                
                topk_logits, topk_indices = torch.topk(current_batch_active_logits, self. beam_size)
                
                next_beams = current_batch_active_input[topk_indices]
                next_beams_logits = topk_logits
                next_beam_confidence = current_batch_active_confidence[topk_indices]
                
                # check if any beam is completed
                completed_beam_count = 0
                for beam_idx, beam in enumerate(next_beams):
                    if len(completed_beams[i]) > 4:
                        minimum_logits = torch.min(torch.stack(completed_beams_logits[i]))
                        if next_beams_logits[beam_idx] < minimum_logits:
                            next_beams[beam_idx] = torch.zeros_like(beam)
                            next_beams_logits[beam_idx] = -torch.inf
                            next_beam_confidence[beam_idx] = torch.ones_like(next_beam_confidence[beam_idx]) * torch.inf
                            completed_beam_count += 1
                            
                            continue
                    if beam[-1] == eos_idx or t == max_char_len:
                        completed_beams[i].append(beam.clone())
                        completed_beams_logits[i].append(next_beams_logits[beam_idx].clone())
                        completed_beams_confidence[i].append(next_beam_confidence[beam_idx].clone())
                        completed_beam_count += 1
                        # replace the completed beam with zeros and logits with -inf 
                        # so that we can remove them on start of next loop
                        next_beams[beam_idx] = torch.zeros_like(beam)
                        # next_beams_logits[beam_idx] = -torch.inf
                        next_beams_logits[beam_idx] = next_beams_logits[beam_idx].clone()
                        next_beam_confidence[beam_idx] = torch.ones_like(next_beam_confidence[beam_idx]) * torch.inf

                next_batch_inputs.append(next_beams)
                next_batch_logits.append(next_beams_logits)
                next_batch_confidence.append(next_beam_confidence)

                beam_start_idx = beam_start_idx + active_no_of_beam_in_input[i]
                completed_beam_start_idx += 4
                active_no_of_beam_in_input[i] = self. beam_size - completed_beam_count

            current_beam = torch.vstack(next_batch_inputs)
            current_beam = current_beam.reshape(bs, -1, current_beam.shape[-1]) # (bs, beam_size, seq_len)

            current_logits = torch.hstack(next_batch_logits)
            current_logits = current_logits.reshape(bs, -1, 1) # (bs, beam_size)
    
            current_beam_confidence = torch.vstack(next_batch_confidence)
            current_beam_confidence = current_beam_confidence.reshape(bs, -1, current_beam_confidence.shape[-1]) # (bs, beam_size, seq_len)

        return self._process_beam_search_output(completed_beams, completed_beams_logits, completed_beams_confidence)

    def _initialize_beam(self, memory, cls_embedding, first_seq_len, sos_idx):
        batch_size = memory.shape[0]
        initial_beam_tokens = torch.ones((batch_size, 1), dtype=torch.int64, device=self.device) * sos_idx
        initial_beam_logits = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        logits, _ = self._model_step(memory, initial_beam_tokens, cls_embedding, first_seq_len, 0)
        
        topk_logits, topk_indices = torch.topk(logits, self.beam_size, dim=1)

        initial_beam_logits = initial_beam_logits.repeat_interleave(self.beam_size)
        initial_beam_logits += topk_logits.view(-1)

        initial_beam_tokens = initial_beam_tokens[:, None, :].expand(-1, self.beam_size, -1).contiguous().view(-1, initial_beam_tokens.shape[-1])
        initial_beam_tokens = torch.cat((initial_beam_tokens, topk_indices.view(-1, 1)), dim=1)

        initial_beams = initial_beam_tokens.reshape(batch_size, self.beam_size, -1)
        initial_beam_logits = initial_beam_logits.reshape(batch_size, self.beam_size, -1)

        return initial_beams, initial_beam_logits
    
    def _initialize_beam_padding(self, beam_size, max_char_len, device):
        beam_padding = torch.zeros((beam_size, max_char_len + 2), dtype=torch.int64).to(device)
        logits_padding = torch.ones(beam_size, dtype=torch.float32).to(device) * float('-inf')
        confidence_padding = torch.ones((beam_size, max_char_len + 1), dtype=torch.float32).to(device) * float('inf')
        completed_beam_start_idx = torch.tensor(0, device=device)
        return beam_padding, logits_padding, confidence_padding, completed_beam_start_idx
    
    def _model_step(self, memory, tgt, cls_embedding, first_seq_len, step):
        tgt_mask = get_mask_seq_cat(first_seq_len, step+1).to(self.device)

        output = self.text_decoder(
            memory, 
            tgt,
            tgt_mask, 
            cls_embedding.unsqueeze(1), 
            first_seq_len+step
        )            
        penalty = self._length_penalty(tgt.shape[1] + 1) # len of total sequence
        logits = torch.log(output) / penalty

        return logits, penalty

    def _process_beam_search_output(self, completed_beams, completed_beams_logits, completed_beams_confidence):
        batch_results = []
        batch_confidence = []
        for i, batch in enumerate(completed_beams.values()):
            sorted_indices =  torch.argsort(torch.tensor(completed_beams_logits[i]), descending=True)   # Get indices in descending order

            sorted_beam_list = [batch[j].cpu() for j in sorted_indices]
            sorted_logits_list = [completed_beams_logits[i][j].cpu() for j in sorted_indices]
            sorted_beam_confidence_list = [completed_beams_confidence[i][j].cpu() for j in sorted_indices]

            # Get the top k beams and their corresponding logits
            # top_k_results = [(sorted_beam_list[j], sorted_logits_list[j], sorted_beam_confidence_list[j]) for j in range(self.beam_size)]
            # batch_results.append(top_k_results)
            batch_results.append(sorted_beam_list[0])
            batch_confidence.append(sorted_beam_confidence_list[0])

        return batch_results, batch_confidence
        
    def _length_penalty(self, length, alpha=0.2):
        return ((5.0 + length)/5.0) ** alpha