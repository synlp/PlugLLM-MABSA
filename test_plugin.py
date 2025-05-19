import argparse
import torch
import json
import cv2
import re

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from tqdm import tqdm
from PIL import Image
import os
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from sklearn.metrics import accuracy_score, f1_score
import nltk
from transformers import ViTModel, ViTFeatureExtractor, BertModel, BertTokenizer
from plugins import Hub, VisualPlugin, TextPlugin
from safetensors.torch import load_file


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def process_visual_plugin(image_folder, instance, visual_plugin, visual_extractor, device):
    def crop_bbox(image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return image[y1:y2, x1:x2]

    image_path = os.path.join(image_folder, instance['image_path'])
    image = cv2.imread(image_path)
    visual_features = instance['visual_features']
    # 遍历每个边界框，裁剪并提取特征
    image_list = []
    connections = []
    for i, feature in enumerate(visual_features[:min(10, len(visual_features))]):
        box = feature['bbox']
        # 裁剪边界框区域
        cropped_img = crop_bbox(image, box)
        # 将裁剪后的图像转换为 PIL 格式（ViT 需要 PIL 格式的图像）
        cropped_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        cropped_img = visual_extractor.preprocess(cropped_img, return_tensors='pt')['pixel_values'][0]
        image_list.append(cropped_img)

    size = len(image_list)
    if size == 0:
        image_list.append(torch.zeros((3, 224, 224)))
    size = len(image_list)
    visual_plugin_adj_matrix = torch.zeros(1, size, size, dtype=torch.float16)

    for i in range(size):
        for j in range(size):
            visual_plugin_adj_matrix[0][i][j] = 1
            visual_plugin_adj_matrix[0][j][i] = 1

    image_list = torch.stack(image_list)
    visual_plugin_inputs = torch.stack([image_list]).to(device)
    visual_plugin_valid = torch.tensor([size]).to(device)
    visual_plugin_adj_matrix = visual_plugin_adj_matrix.to(device)

    output = visual_plugin(visual_plugin_inputs, visual_plugin_valid, visual_plugin_adj_matrix)
    return output


def process_text_plugin(instance, text_plugin, text_tokenizer, device):
    text_features = instance['text_features']

    textlist = text_features['sentence']
    tokens = []
    valid = []
    input_text_list = ["[CLS]"] + textlist + ["[SEP]"]

    for i, word in enumerate(input_text_list):
        token = text_tokenizer.encode(word, add_special_tokens=False)
        tokens.extend(token)
        for m in range(len(token)):
            if m == 0 and word not in ["[CLS]", "[SEP]"]:
                valid.append(1)
            else:
                valid.append(0)

    assert len(valid) == len(tokens)

    chunks = text_features['chunks']
    connections = []
    for i in range(len(textlist)):
        for j in range(len(textlist)):
            if i - 1 <= j <= i + 1:
                connections.append((i, j))

    aspect_index = text_features['aspect_term']['index']
    # for i in range(max(0, aspect_index-3), min(aspect_index+3, len(textlist))):
    #     for j in range(max(0, aspect_index - 3), min(aspect_index + 3, len(textlist))):
    #         connections.append((i, j))
    if 'm_connection' in text_features:
        for (a, b) in text_features['m_connection']:
            connections.append((a, b))
            connections.append((b, a))

    connections = [list(set(connections))]
    text_plugin_inputs = torch.tensor([tokens], dtype=torch.long).to(device)
    text_plugin_attention_mask = text_plugin_inputs.ne(text_tokenizer.pad_token_id).to(device)
    text_plugin_valid_ids = torch.tensor([valid], dtype=torch.long).to(device)
    text_aspect_index = torch.tensor([aspect_index], dtype=torch.long).to(device)
    # text
    batch_size, max_length = text_plugin_inputs.shape[0], text_plugin_inputs.shape[1]
    text_adj_matrix = torch.zeros(batch_size, max_length, max_length, dtype=torch.float16)
    try:
        for i, text_conn in enumerate(connections):
            for pair in text_conn:
                text_adj_matrix[i][pair[0]][pair[1]] = 1
                text_adj_matrix[i][pair[1]][pair[0]] = 1
    except IndexError:
        pass
    text_adj_matrix = text_adj_matrix.to(device)
    output = text_plugin(text_plugin_inputs, text_plugin_attention_mask, text_plugin_valid_ids,
                         text_adj_matrix, text_aspect_index)
    return output

def remove_rt_prefix(text):
    return re.sub(r'^RT @\S+ : ', '', text)


def main(args):
    # Model
    disable_torch_init()

    para = {
        'gcn_layer_num': 3,
        'hub_memory_size': 20,
        'hub_hidden_size': 768,
        'hub_output_size': 8
    }

    # para = {
    #     'gcn_layer_num': 3,
    #     'hub_memory_size': 40,
    #     'hub_hidden_size': 768,
    #     'hub_output_size': 16
    # }

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           args.load_8bit, args.load_4bit,
                                                                           device=args.device)
    visual_plugin_feature_extractor = ViTFeatureExtractor.from_pretrained(args.visual_plugin_model_path)
    visual_plugin_encoder = ViTModel.from_pretrained(args.visual_plugin_model_path)
    visual_plugin = VisualPlugin(visual_plugin_encoder, para['gcn_layer_num'])

    text_plugin_tokenizer = BertTokenizer.from_pretrained(args.text_plugin_model_path)
    text_plugin_encoder = BertModel.from_pretrained(args.text_plugin_model_path)
    text_plugin = TextPlugin(text_plugin_encoder, para['gcn_layer_num'])

    hub = Hub(para['hub_memory_size'], para['hub_hidden_size'], para['hub_output_size'])

    model.add_plugins_and_hub(visual_plugin, text_plugin, hub)

    for index in range(6):
        weights = load_file(f"{args.model_path}/model-0000%d-of-00006.safetensors" % (index+1))
        model.load_state_dict(weights, strict=False)
    model.to(args.device)
    model.to(torch.float16)
    model.eval()
    print('parameters are loaded')

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv_mode = 'llava_v1'

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                              args.conv_mode,
                                                                                                              args.conv_mode))
    else:
        args.conv_mode = conv_mode

    with open(args.test_data, 'r') as f:
        test_set = json.load(f)
    all_output = []
    output_content = []
    gold_content = []
    with torch.inference_mode():
        for instance in tqdm(test_set, desc='processing test set'):
            conv = conv_templates[args.conv_mode].copy()

            visual_output = process_visual_plugin(args.data_home, instance, model.get_visual_plugin(),
                                                  visual_plugin_feature_extractor, model.device)
            text_output = process_text_plugin(instance, model.get_text_plugin(), text_plugin_tokenizer, model.device)
            hub_output = model.get_hub()(visual_output, text_output)
            hub_output = [model.projection(hub_output)]

            image_path = instance['image_path']
            image_file = os.path.join(args.data_home, image_path)
            image = Image.open(image_file).convert('RGB')
            sentence = instance['sentence']
            aspect_term = instance['aspect term']
            # text = sentence.replace('$T$', aspect_term)
            text = remove_rt_prefix(sentence.replace('$T$', '<Aspect> ' + aspect_term + ' </Aspect>'))
            image_size = image.size
            # Similar operation in model_worker.py
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            hint_words = ' '.join(instance['text_features']['key_phrase'])

            inp = f"Sentence: {text}\nAspect term: {aspect_term}\n" \
                  f"Hint Words: {hint_words}\n"
                  # f"What is the sentiment polarity towards the aspect term " \
                  # f"given the above image and text?\n"

            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
                model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                hub_output=hub_output,
            )

            outputs = tokenizer.decode(output_ids[0]).strip().strip('</s>').strip('<s>').strip()
            conv.messages[-1][-1] = outputs
            gold = instance['label']

            instance = {
                'input': prompt,
                'image': image_path,
                'sentence': sentence,
                'aspect_term': aspect_term,
                'pred': outputs,
                'gold': gold
            }

            print('pred: ', outputs)
            print('gold: ', gold)

            all_output.append(instance)

            output_content.append(outputs)
            gold_content.append(gold)

    accuracy = accuracy_score(gold_content, output_content)
    macro_f1 = f1_score(gold_content, output_content, average='macro')
    results = {
        'acc': accuracy,
        'macro_f1': macro_f1
    }
    print(results)

    final_results = {
        'results': results,
        'all_output': all_output
    }

    # visualize all the results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, args.outfile), 'w', encoding='utf8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_home", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--outfile", type=str, default=None)

    parser.add_argument("--visual_plugin_model_path", type=str, default=None)
    parser.add_argument("--text_plugin_model_path", type=str, default=None)

    args = parser.parse_args()
    main(args)
