import os
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import get_file_name
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    batch_size = 20

    classes_path = os.path.join("tree","tree","VOCdevkit","VOC2012","class_names.txt")
    images_path = os.path.join("tree","tree","VOCdevkit","VOC2012","JPEGImages")
    masks_path = os.path.join("tree","tree","VOCdevkit","VOC2012","SegmentationClass")
    split_folder = os.path.join("tree","tree","VOCdevkit","VOC2012","ImageSets","Segmentation")
    images_ext = ".jpg"
    masks_ext = ".png"


    def get_unique_colors(img):
        unique_colors = []
        img = img.astype(np.int32)
        h, w = img.shape[:2]
        colhash = img[:, :, 0] * 256 * 256 + img[:, :, 1] * 256 + img[:, :, 2]
        unq, unq_inv, unq_cnt = np.unique(colhash, return_inverse=True, return_counts=True)
        indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
        col2indx = {unq[i]: indxs[i][0] for i in range(len(unq))}
        for col, indx in col2indx.items():
            if col != 0:
                unique_colors.append((col // (256**2), (col // 256) % 256, col % 256))

        return unique_colors


    def create_ann(image_path):
        labels = []

        image_name = get_file_name(image_path)
        class_name = image_name.split("_")[0]
        class_name_lower = class_name.lower()
        class_name_lower_corr = '_'.join(class_name_lower.split(' '))
        obj_class = meta.get_obj_class(class_name_lower_corr)
        tags = [sly.Tag(class_tag, value=class_name_lower_corr)]
        mask_path = os.path.join(masks_path, image_name + masks_ext)

        mask_np = sly.imaging.image.read(mask_path)
        img_height = mask_np.shape[0]
        img_wight = mask_np.shape[1]
        unique_colors = get_unique_colors(mask_np)
        for color in unique_colors:
            mask = np.all(mask_np == color, axis=2)
            ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
            for i in range(1, ret):
                obj_mask = curr_mask == i
                bitmap = sly.Bitmap(data=obj_mask)
            # if bitmap.area > 100:
                label = sly.Label(bitmap, obj_class)
                labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)


    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta()
    tag_metas = []
    with open(classes_path) as f:
        content = f.read().split("\n")

        for idx, class_name in enumerate(content):
            if idx != 0:
                if class_name == "Magnolia grandiflora L.":
                    class_name = "Magnolia grandiflora L"
                mask_path = os.path.join(masks_path, class_name + "_tree_1 (1).png")
                mask_np = sly.imaging.image.read(mask_path)
                unique_colors = get_unique_colors(mask_np)
                class_name_lower = class_name.lower()
                class_name_lower_corr = '_'.join(class_name_lower.split(' '))
                
                obj_class = sly.ObjClass(class_name_lower_corr, sly.Bitmap, color=unique_colors[0])
                meta = meta.add_obj_class(obj_class)

    class_tag = sly.TagMeta(name="classification tag",value_type=sly.TagValueType.ANY_STRING)
    tag_metas.append(class_tag)
    meta = meta.add_tag_metas(tag_metas)
    api.project.update_meta(project.id, meta.to_json())


    for split_file in os.listdir(split_folder):
        ds_name = get_file_name(split_file)
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
        split_path = os.path.join(split_folder, split_file)

        with open(split_path) as f:
            images_names = [im_name + images_ext for im_name in f.read().split("\n")]

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(images_path, image_name) for image_name in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project
