import os

os.system('cd fairseq;'
          'pip install --use-feature=in-tree-build ./; cd ..')
os.system('ls -l')

import torch
import numpy as np
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from PIL import Image
from torchvision import transforms
import cv2
import gradio as gr

# Register refcoco task
tasks.register_task('refcoco', RefcocoTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

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
        result, scores = eval_step(task, generator, models, sample)
    img = np.asarray(Image)
    cv2.rectangle(
        img,
        (int(result[0]["box"][0]), int(result[0]["box"][1])),
        (int(result[0]["box"][2]), int(result[0]["box"][3])),
        (0, 255, 0),
        3
    )
    return img


title = "OFA-Visual_Grounding"
description = "Gradio Demo for OFA-Visual_Grounding. Upload your own image or click any one of the examples, " \
              "and write a description about a certain object.  " \
              "Then click \"Submit\" and wait for the result of grounding. "
article = "<p style='text-align: center'><a href='https://github.com/OFA-Sys/OFA' target='_blank'>OFA Github " \
          "Repo</a></p> "
examples = [['pokemons.jpg', 'a blue turtle-like pokemon with round head'],
            ['one_piece.jpeg', 'a man in a straw hat and a red dress'],
            ['flowers.jpg', 'a white vase and pink flowers']]
io = gr.Interface(fn=visual_grounding, inputs=[gr.inputs.Image(type='pil'), "textbox"],
                  outputs=gr.outputs.Image(type='numpy'),
                  title=title, description=description, article=article, examples=examples,
                  allow_flagging=False, allow_screenshot=False)
io.launch(cache_examples=True)
