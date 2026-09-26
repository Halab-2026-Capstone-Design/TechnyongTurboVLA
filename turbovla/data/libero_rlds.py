from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoImageProcessor
import tensorflow as tf
import tensorflow_datasets as tfds

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


class LiberoRLDSDataset(IterableDataset):
    def __init__(
        self,
        dataset_dir,
        LOCAL_DINOV3_PATH,
        rank=0,
        world_size=1,
        chunk_size=8,
        split="train",
        shuffle_buffer=512,
        shuffle_steps_within_episode=False,
        step_mix_buffer_size=0,
        seed=42,
        local_files_only=True,
        expected_image_size=256,
    ):
        self.dataset_dir = dataset_dir
        self.dino_processor = AutoImageProcessor.from_pretrained(
            LOCAL_DINOV3_PATH,
            local_files_only=bool(local_files_only),
        )

        self._disable_spatial_resize(self.dino_processor)

        self.chunk_size = int(chunk_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.split = split
        self.shuffle_buffer = int(shuffle_buffer)
        self.shuffle_steps_within_episode = bool(shuffle_steps_within_episode)
        self.step_mix_buffer_size = int(step_mix_buffer_size)
        self.seed = int(seed)
        self.expected_image_size = int(expected_image_size)

        self.proprio_mean = torch.tensor(
            [
                -0.04190646484494209,
                0.03539437800645828,
                0.8257066607475281,
                2.908315658569336,
                -0.5562158823013306,
                -0.16649103164672852,
                0.02831534668803215,
                -0.028561558574438095,
            ],
            dtype=torch.float32,
        )

        self.proprio_std = torch.tensor(
            [
                0.10743443667888641,
                0.14424759149551392,
                0.25723373889923096,
                0.34413808584213257,
                1.234430193901062,
                0.35798805952072144,
                0.013308786787092686,
                0.013174591585993767,
            ],
            dtype=torch.float32,
        )

        self.action_min = torch.tensor(
            [
                -0.9375,
                -0.9375,
                -0.9375,
                -0.23642857372760773,
                -0.3053571283817291,
                -0.3675000071525574,
                -1.0,
            ],
            dtype=torch.float32,
        )
        self.action_max = torch.tensor(
            [
                0.9375,
                0.9375,
                0.9375,
                0.30000001192092896,
                0.29357144236564636,
                0.375,
                1.0,
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _disable_spatial_resize(processor):
        if hasattr(processor, "do_resize"):
            processor.do_resize = False
        if hasattr(processor, "do_center_crop"):
            processor.do_center_crop = False

    def _normalize_state(self, state):
        return (state - self.proprio_mean) / (self.proprio_std + 1e-6)

    def _normalize_action_chunk(self, action_chunk):
        action_chunk = action_chunk.clone()
        action_chunk[:, :6] = (
            2.0
            * (action_chunk[:, :6] - self.action_min[:6])
            / (self.action_max[:6] - self.action_min[:6] + 1e-6)
            - 1.0
        )
        action_chunk[:, :6] = action_chunk[:, :6].clamp(-1.0, 1.0)
        return action_chunk

    def _ensure_expected_size(self, img_np):
        height, width = img_np.shape[:2]
        if height != self.expected_image_size or width != self.expected_image_size:
            raise ValueError(
                f"Expected raw RLDS image to already be {self.expected_image_size}x{self.expected_image_size}, "
                f"but got {height}x{width}. This dataset loader intentionally does not resize."
            )

    @staticmethod
    def _ensure_processor_preserved_resolution(pixel_values, img_np, backbone_name):
        height, width = img_np.shape[:2]
        if tuple(pixel_values.shape[-2:]) != (height, width):
            raise RuntimeError(
                f"{backbone_name} preprocessor changed image size from {(height, width)} to "
                f"{tuple(pixel_values.shape[-2:])}. Resize/crop should stay disabled."
            )

    def _process_image_pair(self, rgb_np, depth_np):
        """
        [수정됨] RGB 이미지와 Depth 이미지를 받아 4채널(RGB-D) 텐서로 변환한 뒤 DINOv3 프로세서를 거칩니다.
        """
        self._ensure_expected_size(rgb_np)
        
        # 1. RGB 정규화 및 [0, 1] 범위로 변환 후 [C, H, W] 텐서화
        if rgb_np.max() > 1.0:
            rgb_np = rgb_np.astype(np.float32) / 255.0
        else:
            rgb_np = rgb_np.astype(np.float32)
            
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() # [3, H, W]

        # 2. Depth 전처리 및 [1, H, W] 텐서화
        if depth_np.ndim == 2:
            depth_np = np.expand_dims(depth_np, axis=-1)
        depth_tensor = torch.from_numpy(depth_np).permute(2, 0, 1).float() # [1, H, W]

        # 3. 채널 방향 결합 -> [4, H, W] (RGB-D)
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)

        # 4. DINOv3 프로세서는 기본적으로 3채널을 기대하므로, 
        # 우리가 앞서 수정한 4채널 패치 임베딩 레이어에 맞게 픽셀 값을 직접 전달하거나 가공합니다.
        # 기존 dino_processor는 3채널용이므로 4채널용 텐서는 프로세서를 우회하거나 직접 텐서 형태로 둡니다.
        # 만약 프로세서가 3채널을 강제한다면 아래와 같이 분리 후 합치는 방식을 쓸 수도 있습니다.
        
        # 여기서는 4채널 텐서 자체를 모델에 바로 넘기기 위해 형태를 유지합니다.
        # (만약 정규화나 mean/std 처리가 필요하다면 여기서 추가 수행)
        return {"dinov3": rgbd_tensor}

    @staticmethod
    def _decode_instruction(raw_instruction):
        if isinstance(raw_instruction, bytes):
            return raw_instruction.decode("utf-8")
        if isinstance(raw_instruction, np.ndarray) and raw_instruction.dtype.type is np.bytes_:
            return raw_instruction.item().decode("utf-8")
        return str(raw_instruction)

    def _build_step_sample(self, steps, t, episode_len):
        current_step = steps[t]
        
        # [수정됨] observation에서 RGB와 Depth를 각각 가져와서 전달합니다.
        # (데이터셋 내의 실제 Depth 키 이름이 다를 경우 'depth', 'wrist_depth' 부분을 수정하세요)
        rgb1 = current_step["observation"]["image"]
        depth1 = current_step["observation"]["depth"]
        img1 = self._process_image_pair(rgb1, depth1)

        rgb2 = current_step["observation"]["wrist_image"]
        depth2 = current_step["observation"]["wrist_depth"]
        img2 = self._process_image_pair(rgb2, depth2)

        instruction = self._decode_instruction(current_step["language_instruction"])

        state = torch.tensor(current_step["observation"]["state"], dtype=torch.float32)
        state = self._normalize_state(state)

        future_actions = [steps[i]["action"] for i in range(t, t + min(self.chunk_size, episode_len - t))]
        valid_len = len(future_actions)
        padding_len = self.chunk_size - valid_len

        if padding_len > 0:
            last_action = future_actions[-1]
            future_actions.extend([last_action] * padding_len)

        action_chunk = torch.tensor(np.stack(future_actions), dtype=torch.float32)
        action_chunk = self._normalize_action_chunk(action_chunk)

        action_chunk_mask = torch.zeros(self.chunk_size, dtype=torch.float32)
        action_chunk_mask[:valid_len] = 1.0

        return (img1, img2), instruction, state, action_chunk, action_chunk_mask

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        base_seed = self.seed + 1009 * self.rank + 9176 * worker_id
        builder = tfds.builder_from_directory(builder_dir=self.dataset_dir)

        epoch = 0
        while True:
            epoch_seed = base_seed + epoch
            rng = np.random.default_rng(epoch_seed)
            step_buffer = []

            dataset = builder.as_dataset(split=self.split)

            if self.world_size > 1:
                dataset = dataset.shard(num_shards=self.world_size, index=self.rank)

            if worker_info is not None:
                dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer,
                seed=epoch_seed,
                reshuffle_each_iteration=False,
            )

            for episode in tfds.as_numpy(dataset):
                steps = list(episode["steps"])
                episode_len = len(steps)
                step_indices = list(range(episode_len))

                if self.shuffle_steps_within_episode:
                    rng.shuffle(step_indices)

                if self.step_mix_buffer_size > 0:
                    for t in step_indices:
                        step_buffer.append(self._build_step_sample(steps, t, episode_len))
                        if len(step_buffer) >= self.step_mix_buffer_size:
                            out_idx = int(rng.integers(0, len(step_buffer)))
                            yield step_buffer.pop(out_idx)
                else:
                    for t in step_indices:
                        yield self._build_step_sample(steps, t, episode_len)

            while len(step_buffer) > 0:
                out_idx = int(rng.integers(0, len(step_buffer)))
                yield step_buffer.pop(out_idx)

            epoch += 1


def vla_collate_fn(batch):
    if len(batch) == 0:
        raise ValueError("empty batch cannot be collated")

    dino_img1_list = []
    dino_img2_list = []
    instructions = []
    action_chunks = []
    action_chunk_masks = []
    states = []

    for images, instruction, state, action_chunk, action_chunk_mask in batch:
        if not isinstance(images, (list, tuple)) or len(images) != 2:
            raise ValueError("Each sample must contain two camera views: (img1, img2)")

        img1, img2 = images
        if "dinov3" not in img1 or "dinov3" not in img2:
            raise ValueError("Each view must contain preprocessed tensor for key 'dinov3'")

        dino_img1_list.append(img1["dinov3"])
        dino_img2_list.append(img2["dinov3"])
        instructions.append(instruction)
        action_chunks.append(action_chunk)
        action_chunk_masks.append(action_chunk_mask)
        states.append(state)

    dino_img1 = torch.stack(dino_img1_list, dim=0)
    dino_img2 = torch.stack(dino_img2_list, dim=0)

    samples = {"dinov3": torch.stack([dino_img1, dino_img2], dim=1)}

    action_chunks = torch.stack(action_chunks, dim=0)
    action_chunk_masks = torch.stack(action_chunk_masks, dim=0)
    states = torch.stack(states, dim=0)

    return samples, instructions, states, action_chunks, action_chunk_masks