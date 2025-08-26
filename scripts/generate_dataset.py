import os
import cv2
import csv
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func

def fix_img_shape(img: np.ndarray) -> np.ndarray:
    """
    Ensures the input image has 3 channels and dtype uint8.
    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Image with 3 channels and dtype uint8.
    Raises:
        ValueError: If the input image is None.
    """
    if img is None:
        raise ValueError("Empty image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[-1] not in [1,3]:
        img = img[..., :3]
    if img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def prepare_image_for_psnr(orig: np.ndarray, img_sr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares two images for PSNR/SSIM calculation (same shape, contiguous, 3 channels).
    Args:
        orig (np.ndarray): Original image.
        img_sr (np.ndarray): Super-resolved or upsampled image.
    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of processed original and upsampled images.
    """
    orig = fix_img_shape(orig)
    img_sr = fix_img_shape(img_sr)
    if img_sr.shape[:2] != orig.shape[:2]:
        img_sr = cv2.resize(img_sr, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_CUBIC)
    orig = np.ascontiguousarray(orig)
    img_sr = np.ascontiguousarray(img_sr)
    return orig, img_sr

def get_quality_values(orig: np.ndarray, img_sr: np.ndarray) -> tuple[float, float]:
    """
    Calculates PSNR and SSIM between two images.
    Args:
        orig (np.ndarray): Original image.
        img_sr (np.ndarray): Super-resolved or upsampled image.
    Returns:
        tuple[float, float]: PSNR and SSIM values.
    """
    orig, img_sr = prepare_image_for_psnr(orig, img_sr)
    psnr_val = cv2.PSNR(orig, img_sr)
    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_sr, cv2.COLOR_BGR2GRAY)
    ssim_val = ssim_func(orig_gray, img_gray, data_range=img_gray.max() - img_gray.min())
    return psnr_val, ssim_val

def load_all_models(pb_models: dict) -> dict:
    """
    Loads all DNN super-resolution models from given paths.
    Args:
        pb_models (dict): Dictionary with model paths.
    Returns:
        dict: Loaded models organized by algorithm and scale.
    """
    sr_models = {}
    for algorithm in pb_models:
        sr_models[algorithm] = {}
        for scale in pb_models[algorithm]:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(pb_models[algorithm][scale])
            sr.setModel(algorithm, scale)
            sr_models[algorithm][scale] = sr
    return sr_models

model_dir = "../models"
pb_models = {
    "edsr": {2: os.path.join(model_dir, "EDSR_x2.pb"), 4: os.path.join(model_dir, "EDSR_x4.pb")},
    "espcn": {2: os.path.join(model_dir, "ESPCN_x2.pb"), 4: os.path.join(model_dir, "ESPCN_x4.pb")},
    "fsrcnn": {2: os.path.join(model_dir, "FSRCNN_x2.pb"), 4: os.path.join(model_dir, "FSRCNN_x4.pb")},
    "lapsrn": {2: os.path.join(model_dir, "LapSRN_x2.pb"), 4: os.path.join(model_dir, "LapSRN_x4.pb")}
}
algorithms = ["edsr", "espcn", "fsrcnn", "lapsrn"]
scales = [2, 4]
downscale_factors = [0.5, 0.6, 0.75, 0.9]
image_dir = "dataset_COCO_2000"
image_paths = glob(os.path.join(image_dir, "*.jpg"))
max_images = 100
if len(image_paths) > max_images:
    image_paths = image_paths[:max_images]
csv_path = "dataset_prob_estatistica_v2.csv"
sr_models = load_all_models(pb_models)

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "imagem", "algoritmo", "scale", "downscale_factor",
        "psnr_dl", "ssim_dl", "mse_dl",
        "psnr_bicubic", "ssim_bicubic", "mse_bicubic",
        "psnr_nearest", "ssim_nearest", "mse_nearest",
        "psnr_lanczos", "ssim_lanczos", "mse_lanczos"
    ])
    for path in tqdm(image_paths, desc="Processando imagens"):
        img_name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue
        h_orig, w_orig = img.shape[:2]
        for factor in downscale_factors:
            h = max(1, int(h_orig * factor))
            w = max(1, int(w_orig * factor))
            img_down = cv2.resize(img, (w, h))
            bicubic = cv2.resize(img_down, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            img_ref, bicubic_resized = prepare_image_for_psnr(img, bicubic)
            psnr_bic, ssim_bic = get_quality_values(img_ref, bicubic_resized)
            mse_bic = np.mean((img_ref.astype(np.float32) - bicubic_resized.astype(np.float32)) ** 2)
            nearest = cv2.resize(img_down, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            img_ref, nearest_resized = prepare_image_for_psnr(img, nearest)
            psnr_near, ssim_near = get_quality_values(img_ref, nearest_resized)
            mse_near = np.mean((img_ref.astype(np.float32) - nearest_resized.astype(np.float32)) ** 2)
            lanczos = cv2.resize(img_down, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
            img_ref, lanczos_resized = prepare_image_for_psnr(img, lanczos)
            psnr_lanc, ssim_lanc = get_quality_values(img_ref, lanczos_resized)
            mse_lanc = np.mean((img_ref.astype(np.float32) - lanczos_resized.astype(np.float32)) ** 2)
            for scale in scales:
                for algorithm in algorithms:
                    try:
                        sr = sr_models[algorithm][scale]
                        img_sr = sr.upsample(img_down)
                        img_ref, img_sr_resized = prepare_image_for_psnr(img, img_sr)
                        psnr_dl, ssim_dl = get_quality_values(img_ref, img_sr_resized)
                        mse_dl = np.mean((img_ref.astype(np.float32) - img_sr_resized.astype(np.float32)) ** 2)
                    except Exception as e:
                        psnr_dl, ssim_dl, mse_dl = -1, -1, -1
                    writer.writerow([
                        img_name, algorithm, scale, factor,
                        psnr_dl, ssim_dl, mse_dl,
                        psnr_bic, ssim_bic, mse_bic,
                        psnr_near, ssim_near, mse_near,
                        psnr_lanc, ssim_lanc, mse_lanc
                    ])
        csvfile.flush()