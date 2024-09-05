from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import copy

class Qwen2VLForRetrieval(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.processor = AutoProcessor.from_pretrained("/mnt/workspace/liulf/pretrained_models/Qwen2-VL-7B-Instruct")
        self.message_image = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ]
        self.message_text = [
            {"role": "system", "content": "You are a helpful assistant to choose match the image with the text."},
            {"role": "user", "content": None},
        ]

    def encode_image(self,images, device):
        # images: batch of iamges: [str: image_root]
        inputs = self.process_images(images).to(device)
        outputs = super().forward(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        mask = (inputs.input_ids == self.config.image_token_id).to(device)
        image_embeds = self.mean_mask(last_hidden_states, mask)
        #print("the shape of last_hidden_states is: ", last_hidden_states.shape)
        #image_embeds = last_hidden_states[:, -5, :]
        return image_embeds

    def encode_text(self,texts, device):
        """
        "user":872, "<|im_end|>":151645
        """
        inputs = self.process_texts(texts).to(device)
        outputs = super().forward(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        start_mask = (inputs.input_ids == 872).to(device)
        end_mask = (inputs.input_ids == 151645).to(device)
        text_embeds = self.mean_range_mask(last_hidden_states, start_mask, end_mask)
        #text_embeds = last_hidden_states[:, -5, :]
        return text_embeds

    
    def mean_range_mask(self, hidden_states, start_mask, end_mask):
        b,s,h = hidden_states.shape
        assert start_mask.shape == (b, s)
        assert end_mask.shape == (b, s)
        mean_hidden_states = torch.zeros(b, h)
        for i in range(b):
            left = int(torch.nonzero(start_mask[i])[-1])
            right = int(torch.nonzero(end_mask[i])[-1])
            text_indices = torch.arange(left+2, right)
            mean_hidden_states[i] = hidden_states[i, text_indices].mean(dim=0)
        return mean_hidden_states


    def mean_mask(self, hidden_states, mask):
        b,s,h = hidden_states.shape
        assert mask.shape == (b, s)
        masked_hidden_states = hidden_states[mask]  # Flattened tensor where mask is True

        # Find the number of True values per batch
        num_true_per_batch = mask.sum(dim=1).float()

        # To avoid division by zero, handle cases where num_true_per_batch is zero
        num_true_per_batch = num_true_per_batch.clamp(min=1)  # Avoid division by zero
        mean_hidden_states = torch.zeros(b, h)
        start, end = 0, 0
        for i in range(b):
            if num_true_per_batch[i] > 0:
                end = start + int(num_true_per_batch[i])
                # Reshape the masked hidden states for this batch
                batch_masked_states = masked_hidden_states[start:end]
                mean_hidden_states[i] = batch_masked_states.mean(dim=0)
                start = end
        assert mean_hidden_states.shape == (b, h)
        return mean_hidden_states


    def process_images(self, images):
        """
        images: list of str, str is the path to the image
        """
        messages = []
        for image in images:
            msg = copy.deepcopy(self.message_image)
            msg[0]["content"][0]["image"] = image
            messages.append(msg)
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs
    
    def process_texts(self, texts):
        """
        texts: list of str
        """
        messages = []
        for text in texts:
            msg = copy.deepcopy(self.message_text)
            msg[1]["content"] = text
            messages.append(msg)
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        inputs = self.processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        return inputs