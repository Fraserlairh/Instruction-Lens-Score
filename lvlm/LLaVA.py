"""LLaVA-1.5 wrapper used by the GL_sim hallucination detector.

Loads a ``llava-hf/llava-1.5-*`` checkpoint and exposes :meth:`generate`, which
runs sampling and returns the hidden states / attentions / token bookkeeping
consumed by ``detector.detector.compute_scores``.
"""

import time
import warnings

import torch
import nltk
import inflect
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from lvlm.base import BaseLVLM

warnings.filterwarnings("ignore")

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

_inflect_engine = inflect.engine()


def singularize(word):
    return _inflect_engine.singular_noun(word) or word


class LLaVA(BaseLVLM):
    """Wrapper around ``LlavaForConditionalGeneration`` for LLaVA-1.5."""

    def build_model(self, args):
        model_name = f"llava-hf/{self.version}"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = self.processor.tokenizer
        self.image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.im_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        self.im_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)

    def _build_inputs(self, image, question):
        """Apply the chat template and return model-ready tensors on GPU."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(0, torch.float16)
        return inputs

    def _image_token_span(self, input_ids):
        """Return (start, end) indices of the image token block."""
        positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
        return positions[0].item(), positions[-1].item()

    def generate(self, image, question, img_id, args):
        inputs = self._build_inputs(image, question)
        input_ids = inputs["input_ids"][0]
        image_start, image_end = self._image_token_span(input_ids)

        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.inference_temp,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=True,
            )
            torch.cuda.synchronize()

            answer_attentions = output.attentions[1:]
            answer_attentions = [torch.cat(att, dim=0) for att in answer_attentions]

        output_ids = output.sequences[0]
        final_ans = (
            self.processor.decode(output_ids, skip_special_tokens=True)
            .split("ASSISTANT: ")[-1]
            .strip()
        )

        answer_start = input_ids.shape[0]
        hidden_states = output.hidden_states

        tokens = nltk.word_tokenize(final_ans.lower())
        if len(tokens) != len([singularize(w) for w in tokens]):
            print("warning, the token number unmatched")

        answer_hidden_states = []
        for hidden_index in range(1, len(hidden_states)):
            step_hidden = torch.stack(hidden_states[hidden_index]).squeeze(1).squeeze(1)
            answer_hidden_states.append(step_hidden.unsqueeze(0))
        answer_hidden_states = torch.cat(answer_hidden_states, dim=0)

        instruction_ids = output.sequences[:, image_end:inputs.data["input_ids"].shape[1]][0]
        instruction_tokens = [
            self.processor.decode([instruction_ids[idx]])
            for idx in range(instruction_ids.shape[-1])
        ]

        return {
            "instruction_tokens": instruction_tokens,
            "hidden_states": hidden_states,
            "output_ids": output_ids,
            "image_start": image_start,
            "image_end": image_end,
            "answer_start": answer_start,
            "tokens": tokens,
            "final_ans": final_ans,
            "img_id": img_id,
            "vq_hidden_states": torch.stack(hidden_states[0]),
            "answer_hidden_states": answer_hidden_states,
            "answer_attentions": answer_attentions,
        }

    def easy_generate(self, image, question, args, mask_idx=None):
        inputs = self._build_inputs(image, question)
        input_ids = inputs["input_ids"][0]
        image_start, _ = self._image_token_span(input_ids)
        if mask_idx is not None:
            inputs["attention_mask"][0, mask_idx + image_start] = 0

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.inference_temp,
            )
        return output
