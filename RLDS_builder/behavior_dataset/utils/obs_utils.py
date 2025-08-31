import av
import cv2
import numpy as np
import open3d as o3d
import torch as th
import torch.nn.functional as F
from av.container import Container
from av.stream import Stream
from tqdm import trange
from torch_cluster import fps
from typing import Dict, Optional, Tuple, Generator, List


#==============================================
# Depth
#==============================================

MIN_DEPTH = 0.0
MAX_DEPTH = 10.0
DEPTH_SHIFT = 3.5

def quantize_depth(
    depth: np.ndarray, 
    min_depth: float = MIN_DEPTH, 
    max_depth: float = MAX_DEPTH, 
    shift: float=DEPTH_SHIFT
) -> np.ndarray:
    """
    Quantizes depth values to a 14-bit range (0 to 16383) based on the specified min and max depth.
    
    Args:
        depth (np.ndarray): Depth tensor.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        shift (float): Small value to shift depth to avoid log(0).
    Returns:
        np.ndarray: Quantized depth tensor.
    """
    qmax = (1 << 14) - 1
    log_min = np.log(min_depth + shift)
    log_max = np.log(max_depth + shift)

    log_depth = np.log(depth + shift)
    log_norm = (log_depth - log_min) / (log_max - log_min)
    quantized_depth = np.clip((log_norm * qmax).round(), 0, qmax).astype(np.uint16)

    return quantized_depth


def dequantize_depth(
    quantized_depth: np.ndarray, 
    min_depth: float = MIN_DEPTH, 
    max_depth: float = MAX_DEPTH, 
    shift: float=DEPTH_SHIFT
) -> np.ndarray:
    """
    Dequantizes a 14-bit depth tensor back to the original depth values.
    
    Args:
        quantized_depth (np.ndarray): Quantized depth tensor.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        shift (float): Small value to shift depth to avoid log(0).
    Returns:
        np.ndarray: Dequantized depth tensor.
    """
    qmax = (1 << 14) - 1
    log_min = np.log(min_depth + shift)
    log_max = np.log(max_depth + shift)

    log_norm = quantized_depth / qmax
    log_depth = log_norm * (log_max - log_min) + log_min
    depth = np.clip(np.exp(log_depth) - shift, min_depth, max_depth)

    return depth


#==============================================
# Video I/O
#==============================================

def create_video_writer(
    fpath, 
    resolution, 
    codec_name="libx264", 
    rate=30, 
    pix_fmt="yuv420p",
    stream_options=None,
    context_options=None, 
) -> Tuple[Container, Stream]:
    """
    Creates a video writer to write video frames to when playing back the dataset using PyAV

    Args:
        fpath (str): Absolute path that the generated video writer will write to. Should end in .mp4 or .mkv
        resolution (tuple): Resolution of the video frames to write (height, width)
        codec_name (str): Codec to use for the video writer. Default is "libx264"
        rate (int): Frame rate of the video writer. Default is 30
        pix_fmt (str): Pixel format to use for the video writer. Default is "yuv420p"
        stream_options (dict): Additional stream options to pass to the video writer. Default is None
        context_options (dict): Additional context options to pass to the video writer. Default is None
    Returns:
        av.Container: PyAV container object that can be used to write video frames
        av.Stream: PyAV stream object that can be used to write video frames
    """
    assert fpath.endswith(".mp4") or fpath.endswith(".mkv"), f"Video writer fpath must end with .mp4 or .mkv! Got: {fpath}"
    container = av.open(fpath, mode='w')
    stream = container.add_stream(codec_name, rate=rate)
    stream.height = resolution[0]
    stream.width = resolution[1]
    stream.pix_fmt = pix_fmt
    if stream_options is not None:
        stream.options = stream_options
    if context_options is not None:
        stream.codec_context.options = context_options
    return container, stream

def write_video(obs, video_writer, mode="rgb", batch_size=None, **kwargs) -> None:
    """
    Writes videos to the specified video writers using the current trajectory history

    Args:
        obs (torch.Tensor): Observation tensor
        video_writer (container, stream): PyAV container and stream objects to write video frames to
        mode (str): Mode to write video frames to. Only "rgb", "depth" and "seg" are supported.
        batch_size (int): Batch size to write video frames to. If None, write video frames to the entire video.
        kwargs (dict): Additional keyword arguments to pass to the video writer.
    """
    container, stream = video_writer
    batch_size = batch_size or obs.shape[0]
    if mode == "rgb":
        for i in range(0, obs.shape[0], batch_size):
            for frame in obs[i:i+batch_size]:
                frame = av.VideoFrame.from_ndarray(frame[..., :3], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
    elif mode == "depth":
        for i in range(0, obs.shape[0], batch_size):
            quantized_depth = quantize_depth(obs[i:i+batch_size])
            for frame in quantized_depth:
                frame = av.VideoFrame.from_ndarray(frame, format='gray16le')
                for packet in stream.encode(frame):
                    container.mux(packet)
    elif mode == "seg":
        seg_ids = kwargs["seg_ids"]
        palette = th.from_numpy(generate_yuv_palette(len(seg_ids)))
        # Vectorized mapping - much faster than loop
        max_id = seg_ids.max().item() + 1
        instance_id_to_idx = th.full((max_id,), -1, dtype=th.long)
        instance_id_to_idx[seg_ids] = th.arange(len(seg_ids))
        for i in range(0, obs.shape[0], batch_size):
            seg_colored = palette[instance_id_to_idx[obs[i:i+batch_size]]].numpy()
            for frame in seg_colored:
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
    elif mode == "bbox":
        bbox_2d_data = kwargs["bbox"]
        for i in range(0, obs.shape[0], batch_size):
            for j, frame in enumerate(obs[i:i+batch_size].numpy()):
                # overlay bboxes with names
                frame = overlay_bboxes_with_names(
                    frame, 
                    bbox_2d_data=bbox_2d_data[i+j], 
                    instance_mapping=kwargs["instance_mapping"], 
                    task_relevant_objects=kwargs["task_relevant_objects"]
                )
                frame = av.VideoFrame.from_ndarray(frame[..., :3], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
    else:
        raise ValueError(f"Unsupported video mode: {mode}.")


class VideoLoader:
    def __init__(
        self, 
        path: str,
        batch_size: Optional[int]=None, 
        stride: int=1, 
        output_size: Tuple[int, int]=(128, 128),
        *args, 
        **kwargs
    ):
        """
        Sequentially load RGB video with robust frame extraction.

        Args:
            path (str): Path to the video file
            batch_size (int): Batch size to load the video into memory. If None, load the entire video into memory.
            stride (int): Stride to load the video into memory.
                i.e. if batch_size=3 and stride=1, __iter__ will return [0, 1, 2], [1, 2, 3], [2, 3, 4], ...
        Returns:
            th.Tensor: (T, H, W, 3) RGB video tensor
        """
        self.container = av.open(path.replace(":", "+"))
        self.stream = self.container.streams.video[0]
        self._frames = []
        self.batch_size = batch_size
        self.stride = stride
        self._frame_iter = None
        self._done = False
        self.output_size = output_size
        
    def __iter__(self) -> Generator[th.Tensor, None, None]:
        self.container.seek(0)
        self._frame_iter = self.container.decode(self.stream)
        self._frames = []
        self._done = False
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        try:
            while True:
                frame = next(self._frame_iter)  # may raise StopIteration
                processed_frame = self._process_single_frame(frame)
                self._frames.append(processed_frame)
                if self.batch_size and len(self._frames) == self.batch_size:
                    batch = th.cat(self._frames, dim=0)
                    self._frames = self._frames[self.stride:]
                    return batch
        except StopIteration:
            self._done = True
            if len(self._frames) > 0:
                batch = th.cat(self._frames, dim=0)
                self._frames = []
                return batch
            else:
                raise
        except Exception as e:
            self._done = True
            raise e

    def _process_single_frame(self, frame: av.VideoFrame) -> th.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def reset(self):
        self.container.seek(0)

    @property
    def frames(self) -> th.Tensor:
        """
        Return all frames at once.
        """
        assert not self.batch_size, "Cannot get all frames at once when batch_size is set"
        return next(iter(self))

    def close(self):
        self.container.close()


class RGBVideoLoader(VideoLoader):
    def __init__(
        self, 
        data_path: str, 
        task_id: int,
        camera_id: str,
        demo_id: str,
        *args, 
        **kwargs
        ):
        super().__init__(
            path=f"{data_path}/videos/task-{task_id:04d}/observation.images.rgb.{camera_id}/episode_{task_id:04d}{demo_id:04d}.mp4",
            *args, 
            **kwargs
        )

    def _process_single_frame(self, frame: av.VideoFrame) -> th.Tensor:
        rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3)
        rgb = F.interpolate(
            th.from_numpy(rgb).to(th.uint8).movedim(-1, -3).unsqueeze(0), 
            size=self.output_size, 
            mode='nearest-exact'
        )
        return rgb  # (1, H, W, 3)


class DepthVideoLoader(VideoLoader):
    def __init__(
        self, 
        data_path: str, 
        task_id: int,
        camera_id: str,
        demo_id: str,
        *args, 
        **kwargs
    ):
        self.min_depth = kwargs.get("min_depth", MIN_DEPTH)
        self.max_depth = kwargs.get("max_depth", MAX_DEPTH)
        self.shift = kwargs.get("shift", DEPTH_SHIFT)
        super().__init__(
            path=f"{data_path}/videos/task-{task_id:04d}/observation.images.depth.{camera_id}/episode_{task_id:04d}{demo_id:04d}.mp4",
            *args, 
            **kwargs
        )

    def _process_single_frame(self, frame: av.VideoFrame) -> th.Tensor:
        # Decode Y (luma) channel only; YUV420 → grayscale image
        frame_gray16 = frame.reformat(format='gray16le').to_ndarray()  # (H, W)
        depth = dequantize_depth(
            frame_gray16,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            shift=self.shift
        )
        depth = th.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        depth = F.interpolate(depth, size=self.output_size, mode='nearest-exact')
        return depth.squeeze(0)

class SegVideoLoader(VideoLoader):
    def __init__(
        self, 
        data_path: str, 
        task_id: int,
        camera_id: str,
        demo_id: str,
        *args, 
        **kwargs
    ):
        self.id_list = kwargs.get("id_list", None)
        assert self.id_list is not None, "id_list must be provided for SegVideoLoader"
        self.id_list = self.id_list.to(device="cuda")  # (N_ids,)
        self.palette = th.from_numpy(generate_yuv_palette(len(self.id_list))).float().to(device="cuda")  # (N_ids, 3)
        super().__init__(
            path=f"{data_path}/videos/task-{task_id:04d}/observation.images.seg_instance_id.{camera_id}/episode_{task_id:04d}{demo_id:04d}.mp4",
            *args, 
            **kwargs
        )

    def _process_single_frame(self, frame: av.VideoFrame) -> th.Tensor:
        rgb = th.from_numpy(frame.to_ndarray(format="rgb24")).float().to(device="cuda")  # (H, W, 3)
        rgb_flat = rgb.reshape(-1, 3)  # (H*W, 3)
        # For each rgb pixel, find the index of the nearest color in the equidistant bins
        distances = th.cdist(rgb_flat[None, :, :], self.palette[None, :, :], p=2)[0]  # (H*W, N_ids)
        ids = th.argmin(distances, dim=-1)  # (H*W,)
        ids = self.id_list[ids].reshape((rgb.shape[0], rgb.shape[1])).unsqueeze(0)  # (1, H, W)
        ids = F.interpolate(ids.unsqueeze(0), size=self.output_size, mode='nearest-exact')
        return ids.squeeze(0).cpu().to(th.long)  # (1, H, W)


OBS_LOADER_MAP = {
    "rgb": RGBVideoLoader,
    "depth_linear": DepthVideoLoader,
    "seg_instance_id": SegVideoLoader,
}

# ==============================================
# Segmentation
# ==============================================

def generate_yuv_palette(num_ids: int) -> np.ndarray:
    """
    Generate `num_ids` equidistant YUV colors in the valid YUV space.
    """
    # Y in [16, 235], U, V in [16, 240] for 8-bit YUV standards (BT.601)
    Y_vals = np.linspace(16, 235, int(np.ceil(num_ids ** (1/3))))
    U_vals = np.linspace(16, 240, int(np.ceil(num_ids ** (1/3))))
    V_vals = np.linspace(16, 240, int(np.ceil(num_ids ** (1/3))))

    palette = []
    for y in Y_vals:
        for u in U_vals:
            for v in V_vals:
                palette.append([y, u, v])
                if len(palette) >= num_ids:
                    return np.array(palette, dtype=np.uint8)
    return np.array(palette[:num_ids], dtype=np.uint8)


def instance_id_to_instance(obs: th.Tensor, instance_id_mapping: Dict[int, str], unique_ins_ids: List[int]) -> Tuple[th.Tensor, Dict[int, str]]:
    """
    Instance_id segmentation map each unique visual meshes of objects (e.g. /World/scene_name/object_name/visual_mesh_0)
    This function merges all visual meshes of the same object instance to a single instance id.
    Args:
        obs (th.Tensor): (N, H, W) instance_id segmentation
        instance_id_mapping (Dict[int, str]): Dict mapping instance_id ids to instance names
    Returns:
        instance_seg (th.Tensor): (N, H, W) instance segmentation
        instance_mapping (Dict[int, str]): Dict mapping instance ids to instance names
    """
    # trim the instance ids mapping to the valid instance ids
    instance_id_mapping = {k: v for k, v in instance_id_mapping.items() if k in unique_ins_ids}
    # extract the actual instance name, which is located at /World/scene_name/object_name
    # Note that 0, 1 are special cases for background and unlabelled, respectivelly
    instance_id_to_instance = {k: v.split("/")[3] for k, v in instance_id_mapping.items() if k not in [0, 1]}
    # get all unique instance names 
    instance_names = set(instance_id_to_instance.values())
    # construct a new instance mapping from instance names to instance ids
    instance_mapping = {0: "background", 1: "unlabelled"}
    instance_mapping.update({k+2: v for k, v in enumerate(instance_names)}) # {i: object_name}
    reversed_instance_mapping = {v: k for k, v in instance_mapping.items()} # {object_name: i}
    # put back the background and unlabelled
    instance_id_to_instance.update({0: "background", 1: "unlabelled"})
    # Now, construct the instance segmentation
    instance_seg = th.zeros_like(obs)
    # Create lookup tensor for faster indexing
    lookup = th.full((max(unique_ins_ids) + 1,), -1, dtype=th.long, device=obs.device)
    for instance_id in unique_ins_ids:
        lookup[instance_id] = reversed_instance_mapping[instance_id_to_instance[instance_id]]
    instance_seg = lookup[obs]
    # Note that now the returned instance mapping will be unique (i.e. no unused instance ids)
    return instance_seg, instance_mapping


def instance_to_bbox(obs: th.Tensor, instance_mapping: Dict[int, str], unique_ins_ids: List[int]) -> List[List[Tuple[int, int, int, int, int]]]:
    """
    Convert instance segmentation to bounding boxes.
    
    Args:
        obs (th.Tensor): (N, H, W) tensor of instance IDs
        instance_mapping (Dict[int, str]): Dict mapping instance IDs to instance names
            Note: this does not need to include all instance IDs, only the ones that we want to generate bbox for
        unique_ins_ids (List[int]): List of unique instance IDs
    Returns:
        List of N lists, each containing tuples (x_min, y_min, x_max, y_max, instance_id) for each instance
    """
    if len(obs.shape) == 2:
        obs = obs.unsqueeze(0)  # Add batch dimension if single frame
    N = obs.shape[0]
    bboxes = [[] for _ in range(N)]
    valid_ids = [id for id in instance_mapping if id in unique_ins_ids]
    for instance_id in valid_ids:
        # Create mask for this instance
        mask = (obs == instance_id)  # (N, H, W)
        # Find bounding boxes for each frame
        for n in range(N):
            frame_mask = mask[n]  # (H, W)
            if not frame_mask.any():
                continue
            # Find non-zero indices (where instance exists)
            y_coords, x_coords = th.where(frame_mask)
            if len(y_coords) == 0:
                continue
            # Calculate bounding box
            x_min = x_coords.min().item()
            x_max = x_coords.max().item()
            y_min = y_coords.min().item()
            y_max = y_coords.max().item()
            bboxes[n].append((x_min, y_min, x_max, y_max, instance_id))
    
    return bboxes


# ==============================================
# Bounding box
# ==============================================

def find_non_overlapping_text_position(x1, y1, x2, y2, text_size, occupied_regions, img_height, img_width):
    """Find a text position that doesn't overlap with existing text."""
    text_w, text_h = text_size
    padding = 5

    # Try different positions in order of preference
    positions = [
        # Above bbox
        (x1, y1 - text_h - padding),
        # Below bbox
        (x1, y2 + text_h + padding),
        # Right of bbox
        (x2 + padding, y1 + text_h),
        # Left of bbox
        (x1 - text_w - padding, y1 + text_h),
        # Inside bbox (top-left)
        (x1 + padding, y1 + text_h + padding),
        # Inside bbox (bottom-right)
        (x2 - text_w - padding, y2 - padding),
    ]

    for text_x, text_y in positions:
        # Check bounds
        if text_x < 0 or text_y < text_h or text_x + text_w > img_width or text_y > img_height:
            continue

        # Check for overlap with existing text
        text_rect = (text_x - padding, text_y - text_h - padding, text_x + text_w + padding, text_y + padding)

        overlap = False
        for occupied_rect in occupied_regions:
            if (
                text_rect[0] < occupied_rect[2]
                and text_rect[2] > occupied_rect[0]
                and text_rect[1] < occupied_rect[3]
                and text_rect[3] > occupied_rect[1]
            ):
                overlap = True
                break

        if not overlap:
            return text_x, text_y, text_rect

    # Fallback: use the first position even if it overlaps
    text_x, text_y = positions[0]
    text_rect = (text_x - padding, text_y - text_h - padding, text_x + text_w + padding, text_y + padding)
    return text_x, text_y, text_rect

def overlay_bboxes_with_names(
    img: np.ndarray, 
    bbox_2d_data: List[Tuple[int, int, int, int, int]],
    instance_mapping: Dict[int, str],
    task_relevant_objects: List[str],
) -> np.ndarray:
    """
    Overlays bounding boxes with object names on the given image.

    Args:
        img (np.ndarray): The input image (RGB) to overlay on.
        bbox_2d_data (List[Tuple[int, int, int, int, int]]): Bounding box data with format (x1, y1, x2, y2, instance_id)
        instance_mapping (Dict[int, str]): Mapping from instance ID to object name
        task_relevant_objects (List[str]): List of task relevant objects
    Returns:
        np.ndarray: The image with bounding boxes and object names overlaid.
    """
    # Create a copy of the image to draw on
    overlay_img = img.copy()
    img_height, img_width = img.shape[:2]

    # Track occupied text regions to avoid overlap
    occupied_text_regions = []

    # Process each bounding box
    for bbox in bbox_2d_data:
        x1, y1, x2, y2, instance_id = bbox
        object_name = instance_mapping[instance_id]
        # Only overlay task relevant objects
        if object_name not in task_relevant_objects:
            continue

        # Generate a consistent color based on instance_id
        color = get_consistent_color(instance_id)

        # Draw the bounding box
        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color, 2)

        # Draw the object name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(object_name, font, font_scale, font_thickness)[0]
        # Find non-overlapping position for text
        text_x, text_y, text_rect = find_non_overlapping_text_position(
            x1, y1, x2, y2, text_size, occupied_text_regions, img_height, img_width
        )
        # Add this text region to occupied regions
        occupied_text_regions.append(text_rect)

        # Draw background rectangle for text
        cv2.rectangle(
            overlay_img, (int(text_rect[0]), int(text_rect[1])), (int(text_rect[2]), int(text_rect[3])), color, -1
        )

        # Draw the text
        cv2.putText(
            overlay_img,
            object_name,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return overlay_img


def get_consistent_color(instance_id):
    import colorsys
    colors = [
        (52, 73, 94),  # Dark blue-gray
        (142, 68, 173),  # Purple
        (39, 174, 96),  # Emerald green
        (230, 126, 34),  # Orange
        (231, 76, 60),  # Red
        (41, 128, 185),  # Blue
        (155, 89, 182),  # Amethyst
        (26, 188, 156),  # Turquoise
        (241, 196, 15),  # Yellow (darker)
        (192, 57, 43),  # Dark red
        (46, 204, 113),  # Green
        (52, 152, 219),  # Light blue
        (155, 89, 182),  # Violet
        (22, 160, 133),  # Dark turquoise
        (243, 156, 18),  # Dark yellow
        (211, 84, 0),  # Dark orange
        (154, 18, 179),  # Dark purple
        (31, 81, 255),  # Royal blue
        (20, 90, 50),  # Forest green
        (120, 40, 31),  # Maroon
    ]

    # Use hash to consistently select a color from the palette
    hash_val = hash(str(instance_id))
    base_color_idx = hash_val % len(colors)
    base_color = colors[base_color_idx]

    # Add slight variation while maintaining sophistication
    # Convert to HSV for easier manipulation
    r, g, b = [c / 255.0 for c in base_color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Add small random variation to hue (±10 degrees) and saturation/value
    hue_variation = ((hash_val >> 8) % 20 - 10) / 360.0  # ±10 degrees
    sat_variation = ((hash_val >> 16) % 20 - 10) / 200.0  # ±5% saturation
    val_variation = ((hash_val >> 24) % 20 - 10) / 200.0  # ±5% value

    # Apply variations with bounds checking
    h = (h + hue_variation) % 1.0
    s = max(0.4, min(0.9, s + sat_variation))  # Keep saturation between 40-90%
    v = max(0.3, min(0.7, v + val_variation))  # Keep value between 30-70% (darker for contrast)

    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    # Convert to 0-255 range
    return (int(r * 255), int(g * 255), int(b * 255))
