import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from os.path import join, dirname
from functools import partial
from typing import Iterable, Union

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
OPENAI_CLIP_SIZE = [224, 224]
OPENAI_RESCALE = 1 / 255.0

image_dir = join(dirname(__file__), '..', '..', 'envs', 'probs', 'tile_ims')
tiles = {
        1: Image.open(join(image_dir, 'empty.png')),
        2: Image.open(join(image_dir, 'solid.png')),
        3: Image.open(join(image_dir, 'bat.png')),
        4: Image.open(join(image_dir, 'bat.png')),
        5: Image.open(join(image_dir, 'scorpion.png')),
        6: Image.open(join(image_dir, 'spider.png')),
        7: Image.open(join(image_dir, 'key.png')),
        8: Image.open(join(image_dir, 'door.png')),
    }
np_tiles = []
for tile_num in sorted(tiles.keys()):
    img = tiles[tile_num]
    img = img.resize((16, 16), resample=Image.BICUBIC)
    np_tiles.append(np.array(img, dtype=np.uint8))
    
JNP_TILES = jnp.array(np_tiles, dtype=jnp.uint8)


@partial(jax.jit, static_argnums=(1,2))
def render_level_from_arr(array: jnp.array, tile_size: int = 16, render_border: bool = True) -> Image:
    if render_border:
        array = jnp.pad(array, ((1,1),(1,1)), constant_values=2)
    idx0_array = array - 1
    
    height, width = idx0_array.shape
    T, tile_h, tile_w, C = JNP_TILES.shape
    
    one_hot = jax.nn.one_hot(idx0_array, T, dtype=JNP_TILES.dtype)
    
    # generate tile: (H, W, tile_h, tile_w, 3)
    block = jnp.tensordot(one_hot, JNP_TILES, axes=([2], [0]))
    
    # 4) transpose, reshape: (H*tile_h, W*tile_w, C)
    #    current block shape = (H, W, tile_h, tile_w, C)
    img = block.transpose(0, 2, 1, 3, 4).reshape(height * tile_h, width * tile_w, C)
    
    return img



def convert_to_rgb(
    image: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convert an image to RGB format and move channels to the last dimension.
    Args:
        image (`jnp.ndarray`):
            The image to convert.
    Returns:
        `jnp.ndarray`: The converted image.
    """
    
    if (image.ndim == 3) and image.shape[0] in (3, 4):
        image = jnp.moveaxis(image, 0, -1)

    if image.shape[-1] == 4:
        image = image[..., :3]
        
    return image

def _cubic_kernel(a: float=-0.5):
    """
    1D cubic convolution kernel 
    """
    def kernel(x):
        absx = jnp.abs(x)
        absx2 = absx ** 2
        absx3 = absx ** 3
        return jnp.where(
            absx <= 1,
            (a + 2) * absx3 - (a + 3) * absx2 + 1,
            jnp.where(
                (absx > 1) & (absx < 2),
                a * absx3 - 5 * a * absx2 + 8 * a * absx - 4 * a,
                0.0
            )
        )
    return kernel

_cubic = _cubic_kernel(-0.5)

def resize(
    image: jnp.ndarray,
    size: Union[int, Iterable[int]],
) -> jnp.ndarray:
    """
    Resize the input image to the given size.
    Args:
        image (`jnp.ndarray`):
            The image to resize.
        size (`int` or `Tuple[int, int]`):
            The target size. If an int is provided, the image will be resized to (size, size).
    Returns:
        `jnp.ndarray`: The resized image.
    """

    orig_h, orig_w, channels = image.shape
    
    scale_h = orig_h / size[0]
    scale_w = orig_w / size[1]
    
    ys = (jnp.arange(size[0]) + 0.5) * scale_h - 0.5
    xs = (jnp.arange(size[1]) + 0.5) * scale_w - 0.5
    
    ys = jnp.clip(ys, 0, orig_h - 1)
    xs = jnp.clip(xs, 0, orig_w - 1)
    
    def _interp(y, x):
        y0 = jnp.floor(y).astype(jnp.int32)
        x0 = jnp.floor(x).astype(jnp.int32)
        dy = y - y0
        dx = x - x0
        
        wy = jnp.stack([_cubic(dy + 1), _cubic(dy), _cubic(dy - 1), _cubic(dy - 2)])
        wx = jnp.stack([_cubic(dx + 1), _cubic(dx), _cubic(dx - 1), _cubic(dx - 2)])
        
        ys_idx = jnp.clip(y0 + jnp.arange(-1, 3), 0, orig_h - 1)
        xs_idx = jnp.clip(x0 + jnp.arange(-1, 3), 0, orig_w - 1)
        
        patch = image[ys_idx[:, None], xs_idx[None, :]] 
        
        tmp = jnp.tensordot(wy, patch, axes=(0, 0))
        val = jnp.tensordot(tmp, wx, axes=(0, 0))
        return val
    
    out = jax.vmap(lambda yy: jax.vmap(lambda xx: _interp(yy, xx))(xs))(ys)
    return out 
    
    



def rescale(
    image: jnp.ndarray,
    scale: float,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Rescales `image` by `scale`.

    Args:
        image (`jnp.ndarray`):
            The image to rescale.
        scale (`float`):
            The scale to use for rescaling the image.
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            The dtype of the output image. Defaults to `jnp.float32`. Used for backwards compatibility with feature
            extractors.
            
    Returns:
        `jnp.ndarray`: The rescaled image.
    """
    if not isinstance(image, jnp.ndarray):
        raise TypeError(f"Input image must be of type jnp.ndarray, got {type(image)}")

    rescaled_image = image.astype(dtype) * scale  
    return rescaled_image


def normalize(
    image: jnp.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> jnp.ndarray:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`jnp.ndarray`):
            The image to normalize.
        mean (`float` or `Iterable[float]`):
            The mean to use for normalization.
        std (`float` or `Iterable[float]`):
            The standard deviation to use for normalization.
    """
    if not isinstance(image, jnp.ndarray):
        raise ValueError("image must be a jnp array")

    num_channels = image.shape[-1]

    # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
    # We preserve the original dtype if it is a float type to prevent upcasting float16.
    if not jnp.issubdtype(image.dtype, jnp.floating):
        image = image.astype(jnp.float32)

    if isinstance(mean, Iterable):
        if len(mean) != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = jnp.array(mean, dtype=image.dtype)

    if isinstance(std, Iterable):
        if len(std) != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
    else:
        std = [std] * num_channels
    std = jnp.array(std, dtype=image.dtype)

    image = (image - mean) / std
    
    return image


def clip_single_preprocess(
    image: jnp.ndarray,
) -> jnp.ndarray:
    """
    Preprocesses an image for CLIP.

    Args:
        image (`jnp.ndarray`):
            The image to preprocess.
    """
    image = convert_to_rgb(image)
    image = resize(image, size=OPENAI_CLIP_SIZE)
    image = rescale(image, scale=OPENAI_RESCALE)
    image = normalize(image, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)

    return image

def batch_channel_validate(
    image_batch: jnp.ndarray
) -> jnp.ndarray:
    """
    Validate the channel dimension of the image batch.

    Args:
        image_batch (`jnp.ndarray`):
            The image batch to validate.
    """
    
    if (image_batch.ndim == 4) & (image_batch.shape[1] in (3, 4)):
        image_batch = jnp.moveaxis(image_batch, 1, -1)
        
    return image_batch


def clip_batch_preprocess(
    image_batch: jnp.ndarray
) -> jnp.ndarray:
    """
    Preprocesses a batch of images for CLIP.

    Args:
        images (`jnp.ndarray`):
            The images to preprocess.
    """
    image_batch = batch_channel_validate(image_batch)
    return jax.vmap(clip_single_preprocess)(image_batch)


def clip_img_transition_preprocess(
    init_image_batch: jnp.ndarray,
    goal_image_batch: jnp.ndarray
) -> jnp.ndarray:
    """
    Preprocesses 2 batch of images for CLIP.
    Args:
        init_image_batch (`jnp.ndarray`):
            The initial images to preprocess.
        goal_image_batch (`jnp.ndarray`):
            The goal images to preprocess.
    """
    batch_size = init_image_batch.shape[0]
    init_image_batch = batch_channel_validate(init_image_batch)
    goal_image_batch = batch_channel_validate(goal_image_batch)
    image_batch = jnp.concatenate([init_image_batch, goal_image_batch], axis=0)
    pre_processed_batch = clip_batch_preprocess(image_batch)
    
    init_image_batch = pre_processed_batch[:batch_size]
    goal_image_batch = pre_processed_batch[batch_size:]
    
    return jnp.concatenate([init_image_batch, goal_image_batch], axis=-1)

if __name__ == "__main__":
    
    #int image
    image = jax.random.randint(jax.random.PRNGKey(0), (3, 288, 288), minval=0, maxval=255)
    
    pre_processed_image = clip_single_preprocess(image)
    
    print("Original image shape:", image.shape)
    print("Preprocessed image shape:", pre_processed_image.shape)
    print("Preprocessed image dtype:", pre_processed_image.dtype)
    print("Preprocessed image mean:", jnp.mean(pre_processed_image))
    
    
    
    # rgb
    # resize
    # rescale
    # normalize