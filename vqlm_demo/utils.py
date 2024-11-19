import os
from multiprocessing import Pool
import numpy as np
import random
from PIL import Image
import re
import cv2
import glob
from natsort import natsorted


class MultiProcessImageSaver(object):
    """
    MultiProcessImageSaver类提供了一种有效的方式来并行保存多张图像。
    通过使用进程池,它能够处理多个图像保存任务,从而加快处理速度,适用于大规模图像处理的场景
    """

    def __init__(self, n_workers=1):
        self.pool = Pool(n_workers) #  创建了一个进程池,以便能够并行执行图像保存任务

    def __call__(self, images, output_files, resizes=None):
        if resizes is None:
            resizes = [None for _ in range(len(images))] # 如果resizes为None,则创建一个与图像数量相同的包含None的列表
        return self.pool.imap(
            self.save_image,
            zip(images, output_files, resizes),
        ) # 用于并行处理图像保存操作,它将图像、文件名和尺寸打包为元组并传递给save_image方法

    def close(self):
        # 关闭进程池,确保没有新的任务被提交,并等待所有进程完成当前任务
        self.pool.close()
        self.pool.join()

    @staticmethod
    def save_image(args):
        image, filename, resize = args
        image = Image.fromarray(image) # 将numpy数组转换为PIL图像对象
        if resize is not None:
            image = image.resize(tuple(resize)) # 如果提供了resize,则使用image.resize方法调整图像大小
        image.save(filename) # 通过image.save(filename)将处理后的图像保存到指定的文件中


def list_dir_with_full_path(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def find_all_files_in_dir(path):
    files = []
    for root, _, files in os.walk(path):
        for file in files:
            files.append(os.path.join(root, file))
    return files


def is_image(path):
    return (
        path.endswith('.jpg')
        or path.endswith('.png')
        or path.endswith('.jpeg')
        or path.endswith('.JPG')
        or path.endswith('.PNG')
        or path.endswith('.JPEG')
    )


def is_video(path):
    return (
        path.endswith('.mp4')
        or path.endswith('.avi')
        or path.endswith('.MP4')
        or path.endswith('.AVI')
        or path.endswith('.webm')
        or path.endswith('.WEBM')
        or path.endswith('.mkv')
        or path.endswith('.MVK')
    )


def random_square_crop(img, random_generator=None):
    # If no random generator is provided, use numpy's default
    if random_generator is None:
        random_generator = np.random.default_rng()

    # Get the width and height of the image
    width, height = img.size

    # Determine the shorter side
    min_size = min(width, height)

    # Randomly determine the starting x and y coordinates for the crop
    if width > height:
        left = random_generator.integers(0, width - min_size)
        upper = 0
    else:
        left = 0
        upper = random_generator.integers(0, height - min_size)

    # Calculate the ending x and y coordinates for the crop
    right = left + min_size
    lower = upper + min_size

    # Crop the image
    return img.crop((left, upper, right, lower))


def read_image_to_tensor(path, center_crop=1.0):
    """
    用于读取图像文件并将其转换为一个归一化的张量(numpy 数组)
    path:图像文件的路径
    center_crop:用于控制中心裁剪的比例,默认为1.0(表示不裁剪)
    """
    pil_im = Image.open(path).convert('RGB') # 打开图像文件,并将其转换为RGB格式
    if center_crop < 1.0:
        # 如果center_crop小于1.0，函数会计算裁剪区域并将图像进行裁剪.裁剪区域是根据给定的center_crop比例计算的,确保裁剪后的图像居中
        width, height = pil_im.size
        pil_im = pil_im.crop((
            int((1 - center_crop) * height / 2), int((1 + center_crop) * height / 2),
            int((1 - center_crop) * width / 2), int((1 + center_crop) * width / 2),
        ))
    input_img = pil_im.resize((256, 256)) # 将裁剪或原始的图像调整为256x256像素的大小
    input_img = np.array(input_img) / 255.0 # 将图像转换为numpy数组,并进行归一化处理(将像素值从0-255范围转换到0-1范围)
    input_img = input_img.astype(np.float32) # 将数据类型转换为 float32
    return input_img


def match_mulitple_path(root_dir, regex):
    videos = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            videos.append(os.path.join(root, file))

    videos = [v for v in videos if not v.split('/')[-1].startswith('.')]

    grouped_path = {}
    for r in regex:
        r = re.compile(r)
        for v in videos:
            matched = r.findall(v)
            if len(matched) > 0:
                groups = matched[0]
                if groups not in grouped_path:
                    grouped_path[groups] = []
                grouped_path[groups].append(v)

    grouped_path = {
        k: tuple(v) for k, v in grouped_path.items()
        if len(v) == len(regex)
    }
    return list(grouped_path.values())


def randomly_subsample_frame_indices(length, n_frames, max_stride=30, random_start=True):
    assert length >= n_frames
    max_stride = min(
        (length - 1) // (n_frames - 1),
        max_stride
    )
    stride = np.random.randint(1, max_stride + 1)
    if random_start:
        start = np.random.randint(0, length - (n_frames - 1) * stride)
    else:
        start = 0
    return np.arange(n_frames) * stride + start


def read_frames_from_dir(dir_path, n_frames, stride, random_start=True, center_crop=1.0):
    files = [os.path.join(dir_path, x) for x in os.listdir(dir_path)]
    files = natsorted([x for x in files if is_image(x)])

    total_frames = len(files)

    if total_frames < n_frames:
        return None

    max_stride = (total_frames - 1) // (n_frames - 1)
    stride = min(max_stride, stride)

    if random_start:
        start = np.random.randint(0, total_frames - (n_frames - 1) * stride)
    else:
        start = 0
    frame_indices = np.arange(n_frames) * stride + start

    frames = []
    for frame_index in sorted(frame_indices):
        # Check if the frame_index is valid
        frames.append(read_image_to_tensor(files[frame_index], center_crop=center_crop))
    if len(frames) < n_frames:
        return None
    frames = np.stack(frames, axis=0)
    return frames


def read_frames_from_video(video_path, n_frames, stride, random_start=True, center_crop=1.0):

    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < n_frames:
        cap.release()
        return None

    max_stride = (total_frames - 1) // (n_frames - 1)
    stride = min(max_stride, stride)

    if random_start:
        start = np.random.randint(0, total_frames - (n_frames - 1) * stride)
    else:
        start = 0
    frame_indices = np.arange(n_frames) * stride + start

    for frame_index in sorted(frame_indices):
        # Check if the frame_index is valid
        if 0 <= frame_index < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                if center_crop < 1.0:
                    height, width, _ = frame.shape
                    frame = frame[
                        int((1 - center_crop) * height / 2):int((1 + center_crop) * height / 2),
                        int((1 - center_crop) * width / 2):int((1 + center_crop) * width / 2),
                        :
                    ]
                frame = cv2.resize(frame, (256, 256))

                frames.append(frame)

        else:
            print(f"Frame index {frame_index} is out of bounds. Skipping...")

    cap.release()
    if len(frames) < n_frames:
        return None
    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0

    # From BGR to RGB
    return np.stack(
        [frames[..., 2], frames[..., 1], frames[..., 0]], axis=-1
    )


def read_all_frames_from_video(video_path, center_crop=1.0):

    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    for frame_index in range(total_frames):
        # Check if the frame_index is valid
        if 0 <= frame_index < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                if center_crop < 1.0:
                    height, width, _ = frame.shape
                    frame = frame[
                        int((1 - center_crop) * height / 2):int((1 + center_crop) * height / 2),
                        int((1 - center_crop) * width / 2):int((1 + center_crop) * width / 2),
                        :
                    ]
                frames.append(cv2.resize(frame, (256, 256)))
        else:
            print(f"Frame index {frame_index} is out of bounds. Skipping...")

    cap.release()
    if len(frames) == 0:
        return None
    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
    # From BGR to RGB
    return np.stack(
        [frames[..., 2], frames[..., 1], frames[..., 0]], axis=-1
    )


def read_max_span_frames_from_video(video_path, n_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < n_frames:
        cap.release()
        return None
    stride = (total_frames - 1) // (n_frames - 1)
    frame_indices = np.arange(n_frames) * stride

    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(frame, (256, 256)))

    cap.release()
    if len(frames) < n_frames:
        return None

    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
    # From BGR to RGB
    return np.stack(
        [frames[..., 2], frames[..., 1], frames[..., 0]], axis=-1
    )

