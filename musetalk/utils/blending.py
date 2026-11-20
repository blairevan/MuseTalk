from PIL import Image
import numpy as np
import cv2
import copy

from musetalk.utils.mask_utils import apply_side_protect_mask, confine_mask_below


def _build_lower_face_alpha(mask_image, ori_shape, face_rect, upper_boundary_ratio,
                            blur_ratio, side_protect_ratio, protect_upper_face):
    """Keep the lower part of a parsing mask, feather it, and optionally lock the upper face.

    Args:
        mask_image (PIL.Image.Image): Parsing mask covering the expanded crop.
        ori_shape (tuple[int, int]): Expanded crop size ``(width, height)``.
        face_rect (tuple[int, int, int, int]): Face box inside the crop.
        upper_boundary_ratio (float): Fraction of crop height above which the
            source frame is kept.
        blur_ratio (float): Gaussian kernel size as a fraction of crop width.
        side_protect_ratio (float): Side-edge fade ratio.
        protect_upper_face (bool): When true, the feather is moved below the
            boundary and every row above it gets zero alpha, so the upper face
            keeps the exact source pixels.

    Returns:
        np.ndarray: ``uint8`` alpha mask of the crop.
    """
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    blur_kernel_size = int(blur_ratio * ori_shape[0] // 2 * 2) + 1
    paste_top = top_boundary + blur_kernel_size // 2 if protect_upper_face else top_boundary
    paste_top = min(paste_top, height)

    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, paste_top, width, height)), (0, paste_top))
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    mask_array = apply_side_protect_mask(mask_array, face_rect, side_protect_ratio)
    if protect_upper_face:
        mask_array = confine_mask_below(mask_array, top_boundary)
    return mask_array


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw",
              fp=None, side_protect_ratio=0.0, protect_upper_face=False):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式
        side_protect_ratio: 左右边缘保护比例，用于降低耳朵和头发附近的融合强度。
        protect_upper_face: 为 True 时上半区 alpha 严格为 0，上半脸保持原帧像素。

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标
    face_position = (x, y)  # 面部在原始图像中的位置

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 裁剪后图像的原始尺寸

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码
    
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 将面部掩码粘贴到全黑图像上
    
    
    # 保留面部区域的下半部分（用于控制说话区域），并对掩码做高斯羽化与侧边保护
    face_rect = (x - x_s, y - y_s, x1 - x_s, y1 - y_s)
    mask_array = _build_lower_face_alpha(
        mask_image,
        ori_shape,
        face_rect,
        upper_boundary_ratio,
        blur_ratio=0.05,
        side_protect_ratio=side_protect_ratio,
        protect_upper_face=protect_upper_face,
    )
    mask_image = Image.fromarray(mask_array)  # 将模糊后的掩码转换回 PIL 图像
    
    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    body.paste(face_large, crop_box[:2], mask_image)
    
    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组

    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）


def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    mask_image = Image.fromarray(mask_array)
    mask_image = mask_image.convert("L")
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5,
                               fp=None, mode="raw", side_protect_ratio=0.0,
                               blur_ratio=0.1, protect_upper_face=False):
    """Prepare reusable blend mask material for realtime inference.

    Args:
        blur_ratio (float): Feather kernel as a fraction of crop width. The
            legacy realtime value is ``0.1``; the offline path uses ``0.05``
            (upstream PR #422).
        protect_upper_face (bool): Lock the upper face to source pixels.
    """
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep the lower part of the talking area, feather it and protect the sides
    face_rect = (x - x_s, y - y_s, x1 - x_s, y1 - y_s)
    mask_array = _build_lower_face_alpha(
        mask_image,
        ori_shape,
        face_rect,
        upper_boundary_ratio,
        blur_ratio=blur_ratio,
        side_protect_ratio=side_protect_ratio,
        protect_upper_face=protect_upper_face,
    )
    return mask_array, crop_box
