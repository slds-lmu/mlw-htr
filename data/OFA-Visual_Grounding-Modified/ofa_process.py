import argparse as ap
import cv2
import json
import os
import numpy as np
import pandas as pd
import re
import torch

from datetime import datetime as dt
from models.ofa import OFAModel
from PIL import Image
from tasks.mm_tasks.refcoco import RefcocoTask
from torchvision import transforms
from tqdm import tqdm
from utils.eval_utils import eval_step

try:
    from fairseq import checkpoint_utils
    from fairseq import utils, tasks
except:
    os.system('cd fairseq;'
        'pip install .; cd ..')
    os.system('ls -l')
    from fairseq import checkpoint_util
    from fairseq import utils, tasks


# Register refcoco task
tasks.register_task('refcoco', RefcocoTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

# Download model
if not os.path.exists('./checkpoints/refcocog.pt'):
    print("Create path and download model.")
    os.system('wget https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/refcocog_large_best.pt; '
        'mkdir -p checkpoints; mv refcocog_large_best.pt checkpoints/refcocog.pt')

# Load pretrained ckpt & config
overrides = {"bpe_dir": "utils/BPE", "eval_cider": False, "beam": 5,
             "max_len_b": 16, "no_repeat_ngram_size": 3, "seed": 7}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths('checkpoints/refcocog.pt'),
    arg_overrides=overrides
)

cfg.common.seed = 7
cfg.generation.beam = 5
cfg.generation.min_len = 4
cfg.generation.max_len_a = 0
cfg.generation.max_len_b = 4
cfg.generation.no_repeat_ngram_size = 3

# Fix seed for stochastic decoding
if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


patch_image_size = cfg.task.patch_image_size


def construct_sample(image: Image, text: str):
    w, h = image.size
    w_resize_ratio = torch.tensor(patch_image_size / w).unsqueeze(0)
    h_resize_ratio = torch.tensor(patch_image_size / h).unsqueeze(0)
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True,
                           append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
        "w_resize_ratios": w_resize_ratio,
        "h_resize_ratios": h_resize_ratio,
        "region_coords": torch.randn(1, 4)
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


# Function for visual grounding
def visual_grounding(Image, Text):
    sample = construct_sample(Image, Text.lower())
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    with torch.no_grad():
        result, _ = eval_step(task, generator, models, sample)
    return result


def check_path_exists(path: str, throw_error: bool) -> bool:
    if '~' in path:
        raise Exception(f"ERROR:\n\t'-> Absolute path must be provided. Current path is '{path}'.")
    exists: bool = os.path.exists(path)
    if exists:
        return True
    else:
        if throw_error:
            raise Exception(f"ERROR:\n\t'-> Path: '{path}' does not exist.")
        else:
            return False


def read_prompt(path: str) -> tuple:
    """Process Prompt File.
    
    Prompt file is supposed to have the following format:
        `
        {
            "all": "prompt_all",
            "except": [
                {
                    "prompt": "prompt_exc",
                    "enum": [1,2,3,4,5]
                    },
                {
                    "prompt": "prompt_exc_2",
                    "enum": [10]
                }
            ]
        }
        `
    In which a general prompt ('all') is defined and a list
    of prompts for specific lengths.
    :param path: Path to prompt.json file.
    :return: Tuple where the first element is the processed list, optimized for fast
        lookup and the second element is the original data.json so that the prompts
        can be written to the target directory.
    """
    check_path_exists(path, throw_error=True)
    f = open(path, 'r')
    prompt_dict: dict = json.load(f)
    f.close()

    assert isinstance(prompt_dict['except'], list)
    assert isinstance(prompt_dict['all'], str)

    zipped: list = list(
        map(
            lambda e: list(zip(e['enum'], ([e['prompt']] * len(e['enum'])))),
            prompt_dict['except']))
    combined: list = [inner for outer in zipped for inner in outer]
    new_dict: dict = dict(combined)
    new_dict['all'] = prompt_dict['all']
    return new_dict, prompt_dict


def get_data_json(path: str) -> dict:
    path_data_json: str = os.path.join(path, 'data.json')
    check_path_exists(path_data_json, throw_error=True)
    f = open(path_data_json)
    data_tmp = json.load(f)
    f.close()
    return data_tmp


def create_target_folder(path: str) -> None:
    """Call before `get_df`!"""
    if not check_path_exists(path, throw_error=False):
        os.mkdir(path)


def get_df(data_json: dict, path_source: str, path_target: str) -> pd.DataFrame:
    """Call before `create_target_folder`!"""
    exist: bool = check_path_exists(path_target, throw_error=False)
    files: list = []
    if exist:
        f = open(os.path.join(path_target, 'output'), 'r')
        for line in f.readlines():
            files.append(int(re.search(r"([0-9]+).jpg", line).group(1)))
    df: pd.DataFrame = pd.DataFrame(data_json)
    df = df[~df['id'].isin(files)]
    df['abs-path'] = list(map(lambda e: os.path.join(path_source, 'zettel', (str(e) + '.jpg')), df['id'].values))
    df["length_lemma"] = [len(i) for i in df["lemma"]]
    return df


def get_appropriate_prompt(lemma_len: int, prompt_rules_transformed: dict) -> str:
    if lemma_len in prompt_rules_transformed.keys():
        return prompt_rules_transformed[lemma_len]
    else:
        return prompt_rules_transformed['all']


def main():
    parser: ap.ArgumentParser = ap.ArgumentParser(
        prog = 'OFA-Preprocessing',
        description = 'Locate the lemmata on the cards')
    parser.add_argument(
        '-d',
        '--data',
        help="Path to the 'MLW' folder.",
        type=str,
        required=True)
    parser.add_argument(
        '-p',
        '--prompt',
        type=str,
        required=True)
    parser.add_argument(
        '-n',
        '--n-total',
        type=int,
        default=1000)
    parser.add_argument(
        '-tf',
        '--target-folder',
        default="data_" + str(dt.now()).replace(' ','-'),
        type=str)

    args = parser.parse_args()

    path: str = args.data
    ret_tuple: tuple = read_prompt(args.prompt)
    prompt_dict, prompt = ret_tuple
    n: int = args.n_total
    target_path: str = args.target_folder

    print(f"Prompt: {prompt}")

    check_path_exists(path, throw_error=True)
    data_info: dict = get_data_json(path)
    df: pd.DataFrame = get_df(data_info, path, target_path)
    create_target_folder(target_path)
    files: list = np.random.choice(df['abs-path'].values, n)
    df = df[df['abs-path'].isin(files)]
    output_data = open(os.path.join(target_path, "output"), "a")

    for _, e in tqdm(df.iterrows()):
        # Define local vars
        file: str = e['abs-path']
        lem_len: int = e['length_lemma']

        # Compute result
        img = Image.open(file, mode="r")
        prompt_tmp: str = get_appropriate_prompt(lem_len, prompt_dict)

        result = visual_grounding(img, prompt_tmp)

        instance: str = re.search(r"([0-9]+)\.jpg", file).group()
        output_data.write(json.dumps({'file': instance, 'result': result}) + "\n")
    output_data.close()
    
    prompt_file = open(os.path.join(target_path, "prompt.json"), "w")
    prompt_file.write(json.dumps(prompt))
    prompt_file.close()

    print(f"DONE! Files written to {target_path}")


if __name__ == "__main__":
    main()
