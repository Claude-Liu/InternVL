import argparse
import itertools
import json
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional

import torch
from internvl.model.internvl_chat import InternVLChatModel
#from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from textvqa_eval import TextVQAAccuracyEvaluator
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor
import copy

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

ds_collections = {
    'vqav2_val': {
        'train': '../../data/vqav2/vqav2_train.jsonl',
        'test': '../../data/vqav2/vqav2_val.jsonl',
        'question': '../../data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': '../../data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': '../../data/okvqa/okvqa_train.jsonl',
        'test': '../../data/okvqa/okvqa_val.jsonl',
        'question': '../../data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': '../../data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': '../../data/textvqa/textvqa_train.jsonl',
        'test': '../../data/textvqa/textvqa_val.jsonl',
        'question': '../../data/textvqa/textvqa_val_questions.json',
        'annotation': '../../data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val_ocr': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val_llava.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': '../../data/vizwiz/vizwiz_train.jsonl',
        'test': '../../data/vizwiz/vizwiz_val.jsonl',
        'question': '../../data/vizwiz/vizwiz_val_questions.json',
        'annotation': '../../data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/val.jsonl',
        'annotation': 'data/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': 'data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/test_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'gqa_testdev_llava': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/llava_gqa_testdev_balanced_qwen_format.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/train.jsonl',
        'test': 'data/ai2diagram/test_vlmevalkit.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'infographicsvqa_val': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/val.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_val_v1.0_withQT.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'infographicsvqa_test': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/test.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_test_v1.0.json',
        'metric': None,
        'max_new_tokens': 100,
    }
}


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches, tokenizer):
    num_patches_list = []
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    if batches[0].get('num_patches_list', None) is not None:
        for _ in batches:
            num_patches_list += _['num_patches_list'] 
    else:
        num_patches_list = None

    return pixel_values, questions, question_ids, annotations, num_patches_list

from torch.nn.utils.rnn import pad_sequence

def preprocess_image(image, max_size=2048, target_size=1024):
    """
    预处理图像：
    1. 如果图像的任何边长超过 max_size，则将其缩小
    2. 然后将图像调整为 target_size x target_size
    3. 最后进行标准化
    """
    # 检查图像类型，如果是文件路径，则打开图像
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image or a file path")

    # 获取原始图像尺寸
    original_width, original_height = image.size

    # 定义预处理步骤
    preprocess_steps = []

    # 如果图像尺寸超过 max_size，先进行缩放
    if original_width > max_size or original_height > max_size:
        preprocess_steps.append(T.Resize(max_size))

    # 添加其他预处理步骤
    preprocess_steps.extend([
        T.Resize(target_size),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 组合所有预处理步骤
    transform = T.Compose(preprocess_steps)

    # 应用预处理
    return transform(image)

    
    return image

def collate_fn_qwen2vl(batch, processor, verbose=False):
    assert len(batch) == 1
    chat_messages = []
    images = []
    annotations = []
    questions = []
    question_ids = []
    
    for item in batch:
        chat_messages.append(item['chat_message'])
        images.append(item['image'])
        annotations.append(item['annotation'])
        questions.append(item['question'])
        question_ids.append(item['question_id'])

    texts = [processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chat_messages]
    images_inputs, video_inputs = process_vision_info(chat_messages)
    #images_inputs = [process_image(image, 2048) for image in images_inputs]
    inputs = processor(text=texts, images=images_inputs, videos=video_inputs, 
                       padding=True, return_tensors="pt")

    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    if verbose:
        print("text: {}".format(texts[0]))
        print("decoded text: {}".format(processor.batch_decode(input_ids)))
        print(f"input_ids: {input_ids.size()}")
        print(f"attention_mask: {attention_mask.size()}")
        print(f"pixel_values: {pixel_values.size()}")
        print(f"image_grid_thw: {image_grid_thw.size()}")
        print(f"annotations: {len(annotations)}")
    assert input_ids.size(0) == attention_mask.size(0) == len(annotations)
    
    return input_ids, attention_mask, pixel_values, image_grid_thw, annotations, questions, question_ids

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class VQADatasetForQwen2vl(torch.utils.data.Dataset):
    def __init__(self, train, test, prompt, few_shot,
                 use_thumbnail=False, max_num=6, mode="single-round"):
        """
        qwen2vl use naive resolution.
        """
        self.root = "/mnt/workspace/liulf/" # to be changed
        self.test = open(test).readlines()
        self.prompt = prompt
        self.few_shot = few_shot
        if few_shot > 0:
            self.train = open(train).readlines()
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.mode = mode # or "single-round"
        #self.processor = AutoProcessor.from_pretrained("/mnt/workspace/liulf/pretrained_models/Qwen2-VL-7B-Instruct")
        self.min_pixels = 1280*28*28
        self.max_pixels = 5120*28*28
        self.message_template =  [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels,},
                    {"type": "text", "text": None},
                ],
            },
            {
                "role": "assistant",
                "content": None
            },
        ]

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)
        #if len(self.prompt) > 0:
        question = question + ' ' + self.prompt
        image = os.path.join(self.root, image)
        chat_message = []
        if self.few_shot > 0 :
            few_shot_samples = random.sample(self.train, self.few_shot)
            if self.mode == "multi-round":
                for sample in few_shot_samples:
                    sample = json.loads(sample.strip())
                    image_, question_, question_id_, annotation_ = sample['image'], sample['question'], sample['question_id'], sample.get('answer', None)
                    image_ = os.path.join(self.root, image_)
                    question_ = question_ + ' ' + self.prompt
                    message_user = copy.deepcopy(self.message_template[0])
                    message_user['content'][0]['image'] = image_
                    message_user['content'][1]['text'] = question_
                    message_system = copy.deepcopy(self.message_template[1])
                    message_system['content'] = annotation_
                    chat_message += [message_user, message_system]
            else:
                assert self.mode == "single-round"
                message = {"role": "user",  "content": [],}
                chat_message.append(message) # single round
                for sample in few_shot_samples:
                    sample = json.loads(sample.strip())
                    image_, question_, question_id_, annotation_ = sample['image'], sample['question'], sample['question_id'], sample.get('answer', None)
                    question_ = question_ + ' ' + self.prompt
                    image_ = os.path.join(self.root, image_)
                    chat_message[0]['content'].append({
                        "type": "image",
                        "image": image_,
                        "min_pixels": self.min_pixels, "max_pixels": self.max_pixels,
                    })
                    chat_message[0]['content'].append({
                        "type": "text",
                        "text": question_ + " The answer is \n" + annotation_,
                    })

        if self.mode == "multi-round" or self.few_shot == 0:
            message_user = copy.deepcopy(self.message_template[0])
            message_user['content'][0]['image'] = image
            message_user['content'][1]['text'] = question
            chat_message.append(message_user)
        else:
            assert self.mode == "single-round"
            chat_message[0]['content'].append({
                "type": "image",
                "image": image,
                "min_pixels": self.min_pixels, "max_pixels": self.max_pixels,
            })
            chat_message[0]['content'].append({
                "type": "text",
                "text": question,
            })

        return {
        'chat_message': chat_message,
        'image': image,
        'annotation': annotation,
        'question_id': question_id,
        'question': question,
    }


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.test = open(test).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.few_shot = few_shot
        self.max_num = max_num
        self.image_pad = "<image>"
        if few_shot > 0:
            self.train = open(train).readlines()
        #self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)
        image = os.path.join('/mnt/workspace/liulf/', image)

        few_shot_prompt = ""
        if self.few_shot > 0:
            pixel_values_list = []
            num_patches_list = []
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += self.image_pad + "\n" + sample["question"] + ' ' + self.prompt +"\n"+ sample['answer'] + "\n"
                pixel_values_ = load_image(os.path.join('/mnt/workspace/liulf/', sample['image']), input_size=self.input_size, max_num=self.max_num)
                pixel_values_list.append(pixel_values_)
                num_patches_list.append(pixel_values_.size(0))

        if self.few_shot == 0:
            pixel_values = load_image(image, input_size=self.input_size, max_num=self.max_num)
        else:
            pixel_values_ = load_image(image, input_size=self.input_size, max_num=self.max_num)
            pixel_values_list.append(pixel_values_)
            num_patches_list.append(pixel_values_.size(0))
            pixel_values = torch.cat(pixel_values_list, dim=0)
        if len(self.prompt) != 0:
            question = self.image_pad + "\n" + question + ' ' + self.prompt
        if len(few_shot_prompt) != 0:
            question = few_shot_prompt + ' ' + question
        return {
            'question_id': question_id,
            'question': question,
            'pixel_values': pixel_values,
            'annotation': annotation,
            "num_patches_list": num_patches_list if self.few_shot > 0 else None
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(response):
    response = response.strip().split('.')[0].split(
        ',')[0].split('!')[0].lower()
    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]
    response = response.strip()
    return response

def get_gpu_info():
    try:
        output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        return output
    except subprocess.CalledProcessError:
        return "nvidia-smi command failed"

def evaluate_chat_model():
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    # infovqa_prompt = 'Answer the question directly.'
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    ai2d_prompt = ''
    random.seed(args.seed)
    summaries = []

    print(args.few_shot)

    for ds_name in args.datasets:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'ai2d' in ds_name:
            input_prompt = ai2d_prompt
        elif 'infographicsvqa' in ds_name:
            input_prompt = infovqa_prompt
        else:
            input_prompt = base_prompt
        
        if args.qwen2vl:
            dataset = VQADatasetForQwen2vl(
                train=ds_collections[ds_name]['train'],
                test=ds_collections[ds_name]['test'],
                prompt=input_prompt,
                few_shot=args.few_shot,
                max_num=args.max_num,
                mode=args.mode
            )
        else:
            dataset = VQADataset(
                train=ds_collections[ds_name]['train'],
                test=ds_collections[ds_name]['test'],
                prompt=input_prompt,
                few_shot=args.few_shot,
                input_size=image_size,
                dynamic_image_size=args.dynamic,
                use_thumbnail=use_thumbnail,
                max_num=args.max_num
            )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer) if not args.qwen2vl else partial(collate_fn_qwen2vl, processor=processor),
        )

        outputs = []
        for i, batch in tqdm(enumerate(dataloader)):
            generation_config = dict(
                    num_beams=args.num_beams,
                    max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                    min_new_tokens=1,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                )
            if args.qwen2vl:
                with torch.no_grad():
                    try:
                        input_ids, attention_mask, pixel_values, image_grid_thw, annotations, questions, question_ids = batch
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        pixel_values = pixel_values.cuda()
                        image_grid_thw = image_grid_thw.cuda()
                        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                        pixel_values=pixel_values, image_grid_thw=image_grid_thw, max_new_tokens=2048,
                                                        do_sample=True,
                                                        top_p=0.001,
                                                        top_k=1,
                                                        temperature=0.01,
                                                        repetition_penalty=1.0,)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
                        ]
                        pred = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                    except:
                        #print(get_gpu_info())
                        print(question_ids)
                        raise
                if i<5:
                    print("pred: {}".format(pred))
                answers = pred
            else:
                pixel_values, questions, question_ids, annotations, num_patches_list = batch
                print(f"pixel_values: {pixel_values.size()}")
                print(f"questions: {len(questions)}")
                print(questions[0])
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                generation_config = dict(
                    num_beams=args.num_beams,
                    max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                    min_new_tokens=1,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                )
                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config,
                    num_patches_list = num_patches_list,
                    verbose=False,
                )
                if i<5:
                    print("question: {}".format(questions[0]))
                    print("pred: {}".format(pred))
                answers = [pred]
            if i>=20 and args.debug:
                break

            for question, question_id, answer, annotation in zip(questions, question_ids, answers, annotations):
                if ds_name in ['vqav2_val', 'vqav2_testdev', 'okvqa_val', 'textvqa_val',
                               'vizwiz_val', 'textvqa_val_ocr']:
                    outputs.append({
                        'question': question,
                        'question_id': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['docvqa_val', 'infographicsvqa_val', 'gqa_testdev', 'ocrvqa_val',
                                 'ocrvqa_test', 'gqa_testdev_llava', 'infographicsvqa_test',]:
                    outputs.append({
                        'question': question,
                        'questionId': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['ai2diagram_test']:
                    outputs.append({
                        'question': question,
                        'image': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['chartqa_test_human', 'chartqa_test_augmented']:
                    outputs.append({
                        'question': question,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['docvqa_test']:
                    outputs.append({
                        'questionId': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['vizwiz_test']:
                    outputs.append({
                        'image': question_id.replace('data/vizwiz/test/', ''),
                        'answer': answer,
                    })
                else:
                    raise NotImplementedError

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            if args.debug:
                for item in merged_outputs:
                    print(item)
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

            if ds_collections[ds_name]['metric'] == 'vqa_score':
                evaluator = TextVQAAccuracyEvaluator()
                annotation = json.load(open(ds_collections[ds_name]['annotation'], 'r'))['annotations']
                question_id2answers = {}
                for item in annotation:
                    question_id = item['question_id']
                    answers = [answer['answer'] for answer in item['answers']]
                    question_id2answers[question_id] = answers
                for item in merged_outputs:
                    item['pred_answer'] = item['answer']
                    item['gt_answers'] = question_id2answers[item['question_id']]
                accuracy = evaluator.eval_pred_list(merged_outputs)
                print(ds_name, accuracy)
                summaries.append([args.checkpoint, ds_name, accuracy])

            elif ds_collections[ds_name]['metric'] == 'anls':
                json.dump(merged_outputs,
                          open(results_file, 'w'),
                          ensure_ascii=False)
                print('python eval/vqa/infographicsvqa_eval.py -g ' +
                      ds_collections[ds_name]['annotation'] + ' -s ' +
                      results_file)
                os.system('python eval/vqa/infographicsvqa_eval.py -g ' +
                          ds_collections[ds_name]['annotation'] + ' -s ' +
                          results_file)
            elif ds_collections[ds_name]['metric'] == 'relaxed_accuracy':
                relaxed_accuracy = evaluate_relaxed_accuracy(merged_outputs)
                print(ds_name, {'relaxed_accuracy': relaxed_accuracy})
                summaries.append([ds_name, {'relaxed_accuracy': relaxed_accuracy}])
            elif ds_collections[ds_name]['metric'] == 'accuracy':
                if 'gqa' in ds_name:
                    dst_file = './data/gqa/testdev_balanced_predictions.json'
                    print('python eval/vqa/convert_gqa_for_eval.py --src ' +
                          results_file + ' --dst ' + dst_file)
                    python_path = 'python'
                    os.system(python_path + ' eval/vqa/convert_gqa_for_eval.py --src ' +
                              results_file + ' --dst ' + dst_file)
                    command = f'cd ./data/gqa/ && {python_path} eval.py --tier testdev_balanced && cd ../../'
                    print(command)
                    accuracy = subprocess.check_output(command, shell=True, universal_newlines=True)
                else:
                    accuracy = {'accuracy': evaluate_exact_match_accuracy(merged_outputs)}
                print(ds_name, accuracy)
                summaries.append([args.checkpoint, ds_name, accuracy])

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str,
                        default='okvqa_val,textvqa_val,vizwiz_val,ai2diagram_test,gqa_testdev_llava')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')

    parser.add_argument("--qwen2vl", action='store_true', help="Use Qwen2VL model")
    parser.add_argument("--few_shot", type=int, default=0, help="Few shot samples")
    parser.add_argument("--debug", action='store_true', help="Debug mode")
    parser.add_argument("--mode", type=str, default="single-round", help="multi-round or single-round")
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    if not args.qwen2vl:
        assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if torch.distributed.get_rank() == 0 and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.auto:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    kwargs = {'device_map': 'auto'} if args.auto else {}
    if args.qwen2vl:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",).eval()
        min_pixels = 1280 * 28 * 28
        max_pixels = 5120 * 28 * 28
        processor = AutoProcessor.from_pretrained("/mnt/workspace/liulf/pretrained_models/Qwen2-VL-7B-Instruct", )
        torch.cuda.empty_cache() # release the cached memory
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
        model = InternVLChatModel.from_pretrained(
            args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
            load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()
        use_thumbnail = model.config.use_thumbnail
        image_size = model.config.force_image_size or model.config.vision_config.image_size
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    if not args.qwen2vl:
        print(f'[test] image_size: {image_size}')
        print(f'[test] template: {model.config.template}')
        print(f'[test] dynamic_image_size: {args.dynamic}')
        print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_chat_model()
