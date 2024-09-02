from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import copy

class Qwen2VLForRetrieval(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.processor = AutoProcessor.from_pretrained("../../pretrained_models/Qwen2-VL-7B-Instruct")
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
        print("the shape of last_hidden_states is: ", last_hidden_states.shape)
        print("the last 5 token of input_ids is : ", inputs["input_ids"][:,-5:])
        image_embeds = last_hidden_states[:, -6, :]
        return image_embeds
    def encode_text(self,texts, device):
        inputs = self.process_texts(texts).to(device)
        outputs = super().forward(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        text_embeds = last_hidden_states[:, -6, :]
        return text_embeds

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