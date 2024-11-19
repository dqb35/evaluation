from abc import ABC, abstractmethod
from contextlib import nullcontext
import time
import os
from functools import partial
from copy import deepcopy
from multiprocessing import Pool
from threading import Lock
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import einops
from transformers import LlamaForCausalLM

from .vqvae_muse import VQGANModel, get_tokenizer_muse
from .torch_vqvae_model import get_tokenizer


def get_torch_float_dtype(dtype):
    """
    用于根据输入的类型字符串返回相应的 PyTorch 数据类型
    """
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return dtype
    return {
        'float16': torch.float16,
        'fp16': torch.float16,
        'f16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
        'f32': torch.float32,
    }[dtype]


def get_pid():
    """
    在延迟一秒后获取当前Python进程的进程ID(PID)
    """
    time.sleep(1)
    return os.getpid()

# 抽象基类
class InferenceModel(ABC):

    @abstractmethod
    def __call__(input_images, n_new_frames, n_candidates, temperature=1.0, top_p=1.0):
        raise NotImplementedError()


class LocalInferenceModel(InferenceModel):

    def __init__(self, checkpoint, dtype='float16', torch_device='cuda',
                 context_frames=16, use_lock=False):
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.torch_device = torch_device
        self.context_frames = context_frames

        # old version of the tokenizer
        # self.tokenizer = get_tokenizer()
        # self.tokenizer.to(self.torch_device)

        # new tokenizer
        self.tokenizer = get_tokenizer_muse()
        self.tokenizer.to(self.torch_device)

        self.model = LlamaForCausalLM.from_pretrained(
            self.checkpoint, torch_dtype=get_torch_float_dtype(self.dtype)
        ).to(self.torch_device)

        if use_lock:
            self.lock = Lock()
        else:
            self.lock = nullcontext()

    @torch.no_grad()
    def compute_perplexity(self, input_images, target_images):
        input_images = np.array(input_images)
        target_images = np.array(target_images)
        assert len(input_images.shape) == 5 and len(target_images.shape) == 5  # [B, S, H, W, C]
        assert input_images.shape[0] == target_images.shape[0]
        batch_size = input_images.shape[0]
        with self.lock:
            input_images = torch.tensor(
                einops.rearrange(input_images, 'b s h w c -> b s c h w')
            ).to(self.torch_device)
            target_images = torch.tensor(
                einops.rearrange(target_images, 'b s h w c -> b s c h w')
            ).to(self.torch_device)
            input_ids = self.tokenizer.tokenize(input_images).view(batch_size, -1)
            target_ids = self.tokenizer.tokenize(target_images).view(batch_size, -1)
            all_ids = torch.cat([input_ids, target_ids], dim=1)
            logits = self.model(all_ids).logits
            log_probs = F.log_softmax(logits, dim=-1)
            target_ids_onehot = F.one_hot(target_ids, num_classes=logits.shape[-1])
            target_log_probs = log_probs[:, input_ids.shape[1] - 1 : -1]
            perplexity = torch.exp(
                -torch.mean(
                    torch.sum(target_log_probs * target_ids_onehot, dim=-1),
                    dim=-1
                )
            )
            return perplexity.detach().cpu().numpy()

    @torch.no_grad()
    def generate_once(self, input_images, n_new_frames, temperature=1.0, top_p=1.0):
        assert type(input_images) == np.ndarray
        with self.lock:
            input_images = np.array(input_images, dtype=np.float32) # (4, 256, 256, 3)即(一个样本中包含图片的数量, 图片高, 图片宽, RGB通道) 
            input_images = torch.tensor(
                einops.rearrange(input_images, 'b h w c -> b c h w')
            ).to(self.torch_device) # torch.Size([4, 3, 256, 256])

            print('here:', type(input_images))

            # old tokenizer
            # input_ids = self.tokenizer.tokenize(input_images).view(1, -1)

            # new tokenizer
            _, input_ids = self.tokenizer.encode(input_images) # input_ids:torch.Size([4, 256])
            input_ids = input_ids.view(1, -1) # torch.Size([1, 1024])


            input_ids = input_ids[:, -(self.context_frames - 1) * 256:] # torch.Size([1, 1024])

            new_tokens = []
            current_context_frames = input_ids.shape[1] // 256 # 4
            fisrt_generation_left = self.context_frames - current_context_frames # 16-4=12
            first_new_frames = min(fisrt_generation_left, n_new_frames) # 1
            input_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=8192,
                max_new_tokens=256 * first_new_frames,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                suppress_tokens=list(range(8192, self.model.vocab_size)),
            ) # torch.Size([1, 1280])
            new_tokens.append(input_ids[:, -256 * first_new_frames:]) # [torch.Size([1, 256])]
            input_ids = input_ids[:, -(self.context_frames - 1) * 256:] # torch.Size([1, 1280])

            for _ in range(max(0, n_new_frames - first_new_frames)):
                input_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    pad_token_id=8192,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    suppress_tokens=list(range(8192, self.model.vocab_size)),
                )
                new_tokens.append(input_ids[:, -256:])
                input_ids = input_ids[:, -(self.context_frames - 1) * 256:]

            new_tokens = torch.cat(new_tokens, dim=1).view(-1, 256) # torch.Size([1, 256])
            new_images = einops.rearrange(
                torch.clamp(self.tokenizer.decode_code(new_tokens), 0.0, 1.0),
                'b c h w -> b h w c'
            ).detach().cpu().numpy() # (1, 256, 256, 3)
        return new_images

    def __call__(self, input_images, n_new_frames, n_candidates, temperature=1.0, top_p=1.0):
        output = []
        for seq in input_images:
            output.append(
                [self.generate_once(seq, n_new_frames, temperature, top_p)
                 for _ in range(n_candidates)]
            )
        return output


class MultiProcessInferenceModel(InferenceModel):

    def __init__(self, checkpoint, torch_devices=None, dtype='float16',
                 context_frames=16, use_lock=False, perplexity_batch_size=2):
        if torch_devices is None or torch_devices == '':
            torch_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] 

        self.torch_devices = torch_devices # 本人笔记本:['cuda:0']
        self.n_processes = len(torch_devices) # 1
        print(f'Using {self.n_processes} processes for inference')
        self.worker_pool = Pool(self.n_processes)
        # starmap 方法会将 get_pid 函数应用到列表中的每个元素.这里传入的列表包含 self.n_processes 个空元组,因此每个进程都会调用一次 get_pid()
        self.worker_pids = self.worker_pool.starmap(get_pid, [tuple() for _ in range(self.n_processes)]) # 最终,self.worker_pids 将保存所有工作进程的 PID
        self.device_map = {
            pid: torch_device
            for pid, torch_device in zip(self.worker_pids, self.torch_devices)
        } # 将每个工作进程的 PID 映射到相应的 PyTorch 设备
        self.worker_pool.starmap(
            self.initialize_worker,
            [(self.device_map, checkpoint, dtype, context_frames) for _ in range(self.n_processes)]
        )
        self.perplexity_batch_size = perplexity_batch_size
        if use_lock:
            self.lock = Lock()
        else:
            self.lock = nullcontext()

    @staticmethod
    def initialize_worker(device_map, checkpoint, dtype, context_frames):
        """
        这个函数通常在多进程工作流中调用,以确保每个工作进程都能根据其 PID 正确配置自己的计算设备和模型
        """
        global _current_process_backend # 全局变量
        torch_device = device_map[os.getpid()]
        _current_process_backend = LocalInferenceModel(
            checkpoint, dtype, torch_device, context_frames
        )

    @staticmethod
    def generate_once(input_images, n_new_frames, temperature=1.0, top_p=1.0):
        return _current_process_backend.generate_once(input_images, n_new_frames, temperature, top_p)

    @staticmethod
    def compute_perplexity_once(input_images, target_images):
        return _current_process_backend.compute_perplexity(input_images, target_images)

    def compute_perplexity(self, input_images, target_images):
        with self.lock:
            map_args = []
            for i in range(0, len(input_images), self.perplexity_batch_size):
                map_args.append((
                    input_images[i : i + self.perplexity_batch_size],
                    target_images[i : i + self.perplexity_batch_size]
                ))
            outputs = self.worker_pool.starmap(self.compute_perplexity_once, map_args)
            return np.concatenate(outputs, axis=0)

    def __call__(self, input_images, n_new_frames, n_candidates, temperature=1.0, top_p=1.0):
        """
        input_images:形状为(batch大小, 一个样本中包含图片的数量, 图片高, 图片宽, RGB通道).
        n_new_frames:生成的新帧数量
        n_candidates:生成的候选结果数量
        temperature and top_p:用于控制生成的随机性和多样性
        """
        # 使用锁来确保多线程或多进程环境下对共享资源的安全访问,防止数据竞争
        with self.lock:
            map_args = []
            # 遍历 input_images 中的每个序列(即每个样本的所有图像),为每个序列生成 n_candidates 个候选参数,构造 map_args 列表
            for seq in input_images: 
                #  seq是每个样本:shape(一个样本中包含图片的数量, 图片高, 图片宽, RGB通道)  
                for _ in range(n_candidates):
                    map_args.append((seq, n_new_frames, temperature, top_p))

            outputs = self.worker_pool.starmap(self.generate_once, map_args) # [(1, 256, 256, 3),(1, 256, 256, 3),(1, 256, 256, 3),(1, 256, 256, 3)]
            reshaped_output = []
            index = 0
            for _ in range(len(input_images)):
                candidates = []
                for _ in range(n_candidates):
                    candidates.append(outputs[index])
                    index += 1
                reshaped_output.append(candidates)
        return reshaped_output

