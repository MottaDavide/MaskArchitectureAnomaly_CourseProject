# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import re
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional
from typing import Tuple
import torch
from PIL import Image
from torch.utils.data import get_worker_info
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


# ---------------------------------------------------------------------------
# Thin abstraction: unifies zipfile.ZipFile and plain filesystem folder
# so the rest of the code works identically for both.
# ---------------------------------------------------------------------------

class _FolderHandle:
    """Mimics the zipfile.ZipFile interface but reads from a plain folder."""

    def __init__(self, folder_root: Path):
        self.folder_root = Path(folder_root)
        self._names = None  # lazy

    # ---- ZipFile-compatible helpers ----------------------------------------

    def namelist(self) -> list[str]:
        if self._names is None:
            self._names = [
                p.relative_to(self.folder_root).as_posix()
                for p in self.folder_root.rglob("*")
                if p.is_file()
            ]
        return self._names

    def infolist(self) -> list["_FolderInfo"]:
        return [_FolderInfo(n) for n in self.namelist()]

    def open(self, name: str, mode: str = "r"):
        return open(self.folder_root / name, "rb")

    def close(self):
        pass  # nothing to close


class _FolderInfo:
    """Mimics zipfile.ZipInfo for _FolderHandle.infolist()."""

    def __init__(self, rel_posix: str):
        self.filename = rel_posix

    def is_dir(self) -> bool:
        return False  # rglob already filters dirs out


# ---------------------------------------------------------------------------
# Factory: return either a ZipFile or a _FolderHandle
# ---------------------------------------------------------------------------

def _open_source(path: Optional[Path]) -> Optional[object]:
    """
    Given a path that can be:
      • a .zip file  → return zipfile.ZipFile
      • a directory  → return _FolderHandle
      • None         → return None
    Raises FileNotFoundError if the path exists in neither form.
    """
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        return _FolderHandle(path)
    if path.is_file() and zipfile.is_zipfile(path):
        return zipfile.ZipFile(path)
    raise FileNotFoundError(
        f"Dataset source not found as zip or directory: {path}"
    )


# ---------------------------------------------------------------------------
# Main Dataset class — identical public API, now zip/folder agnostic
# ---------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zip_path: Path,                          # kept for back-compat; can be a dir
        img_suffix: str,
        target_parser: Callable,
        check_empty_targets: bool,
        transforms: Optional[Callable] = None,
        only_annotations_json: bool = False,
        target_suffix: str = None,
        stuff_classes: Optional[list[int]] = None,
        img_stem_suffix: str = "",
        target_stem_suffix: str = "",
        target_zip_path: Optional[Path] = None,  # can be a dir
        target_zip_path_in_zip: Optional[Path] = None,
        target_instance_zip_path: Optional[Path] = None,
        img_folder_path_in_zip: Path = Path("./"),
        target_folder_path_in_zip: Path = Path("./"),
        target_instance_folder_path_in_zip: Path = Path("./"),
        annotations_json_path_in_zip: Optional[Path] = None,
    ):
        # Store raw paths (may be zip files or directories)
        self.zip_path = Path(zip_path)
        self.target_zip_path = Path(target_zip_path) if target_zip_path else None
        self.target_zip_path_in_zip = target_zip_path_in_zip
        self.target_instance_zip_path = (
            Path(target_instance_zip_path) if target_instance_zip_path else None
        )

        self.target_parser = target_parser
        self.transforms = transforms
        self.only_annotations_json = only_annotations_json
        self.stuff_classes = stuff_classes
        self.target_folder_path_in_zip = target_folder_path_in_zip
        self.target_instance_folder_path_in_zip = target_instance_folder_path_in_zip

        # Per-worker handles (populated lazily in _load_zips)
        self.zip = None
        self.target_zip = None
        self.target_instance_zip = None

        # Open once for __init__ indexing, then release
        img_src, target_src, target_instance_src = self._load_zips()

        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}

        if annotations_json_path_in_zip is not None:
            ann_src = _open_source(target_zip_path or zip_path)
            with ann_src.open(str(annotations_json_path_in_zip), "r") as file:
                annotation_data = json.load(file)

            image_id_to_file_name = {
                image["id"]: image["file_name"] for image in annotation_data["images"]
            }

            for annotation in annotation_data["annotations"]:
                img_filename = image_id_to_file_name[annotation["image_id"]]

                if "segments_info" in annotation:
                    self.labels_by_id[img_filename] = {
                        segment_info["id"]: segment_info["category_id"]
                        for segment_info in annotation["segments_info"]
                    }
                    self.is_crowd_by_id[img_filename] = {
                        segment_info["id"]: bool(segment_info["iscrowd"])
                        for segment_info in annotation["segments_info"]
                    }
                else:
                    if img_filename not in self.labels_by_id:
                        self.labels_by_id[img_filename] = {}
                    if img_filename not in self.polygons_by_id:
                        self.polygons_by_id[img_filename] = {}
                    if img_filename not in self.is_crowd_by_id:
                        self.is_crowd_by_id[img_filename] = {}

                    self.labels_by_id[img_filename][annotation["id"]] = annotation["category_id"]
                    self.polygons_by_id[img_filename][annotation["id"]] = annotation["segmentation"]
                    self.is_crowd_by_id[img_filename][annotation["id"]] = bool(annotation["iscrowd"])

        self.imgs = []
        self.targets = []
        self.targets_instance = []

        target_src_filenames = target_src.namelist()

        for img_info in sorted(img_src.infolist(), key=self._sort_key):
            if not self.valid_member(
                img_info, img_folder_path_in_zip, img_stem_suffix, img_suffix
            ):
                continue

            img_path = Path(img_info.filename)
            if not only_annotations_json:
                rel_path = img_path.relative_to(img_folder_path_in_zip)
                target_parent = target_folder_path_in_zip / rel_path.parent
                target_stem = rel_path.stem.replace(img_stem_suffix, target_stem_suffix)
                target_filename = (target_parent / f"{target_stem}{target_suffix}").as_posix()

            if self.labels_by_id:
                if img_path.name not in self.labels_by_id:
                    continue
                if not self.labels_by_id[img_path.name]:
                    continue
            else:
                if target_filename not in target_src_filenames:
                    continue

                if check_empty_targets:
                    with target_src.open(target_filename) as target_file:
                        min_val, max_val = Image.open(target_file).getextrema()
                        if min_val == max_val:
                            continue

            if target_instance_src is not None:
                target_instance_filename = (
                    target_instance_folder_path_in_zip / (target_stem + target_suffix)
                ).as_posix()

                if check_empty_targets:
                    with target_instance_src.open(target_instance_filename) as ti:
                        extrema = Image.open(ti).getextrema()
                        if all(mn == mx for mn, mx in extrema):
                            _, labels, _ = self.target_parser(
                                target=tv_tensors.Mask(Image.open(target_src.open(target_filename))),
                                target_instance=tv_tensors.Mask(Image.open(ti)),
                                stuff_classes=self.stuff_classes,
                            )
                            if not labels:
                                continue

            self.imgs.append(img_path.as_posix())

            if not only_annotations_json:
                self.targets.append(target_filename)

            if target_instance_src is not None:
                self.targets_instance.append(target_instance_filename)

    # -----------------------------------------------------------------------
    # __getitem__
    # -----------------------------------------------------------------------

    def __getitem__(self, index: int):
        img_src, target_src, target_instance_src = self._load_zips()

        with img_src.open(self.imgs[index]) as img:
            img = tv_tensors.Image(Image.open(img).convert("RGB"))

        target = None
        if not self.only_annotations_json:
            with target_src.open(self.targets[index]) as target_file:
                target = tv_tensors.Mask(Image.open(target_file), dtype=torch.long)

            if img.shape[-2:] != target.shape[-2:]:
                target = F.resize(
                    target,
                    list(img.shape[-2:]),
                    interpolation=F.InterpolationMode.NEAREST,
                )

        target_instance = None
        if self.targets_instance:
            with target_instance_src.open(self.targets_instance[index]) as ti:
                target_instance = tv_tensors.Mask(Image.open(ti), dtype=torch.long)

        masks, labels, is_crowd = self.target_parser(
            target=target,
            target_instance=target_instance,
            stuff_classes=self.stuff_classes,
            polygons_by_id=self.polygons_by_id.get(Path(self.imgs[index]).name, {}),
            labels_by_id=self.labels_by_id.get(Path(self.imgs[index]).name, {}),
            is_crowd_by_id=self.is_crowd_by_id.get(Path(self.imgs[index]).name, {}),
            width=img.shape[-1],
            height=img.shape[-2],
        )

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels),
            "is_crowd": torch.tensor(is_crowd),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    # -----------------------------------------------------------------------
    # _load_zips  (now: _load_sources — same name kept for compatibility)
    # -----------------------------------------------------------------------

    def _load_zips(self) -> Tuple[object, object, Optional[object]]:
        """
        Returns per-worker source handles (zip or folder).
        Handles are cached per worker id so each worker opens its own handle.
        """
        worker = get_worker_info()
        worker_id = worker.id if worker else None

        if self.zip is None:
            self.zip = {}
        if self.target_zip is None:
            self.target_zip = {}
        if self.target_instance_zip is None and self.target_instance_zip_path:
            self.target_instance_zip = {}

        # --- image source ---
        if worker_id not in self.zip:
            self.zip[worker_id] = _open_source(self.zip_path)

        # --- target source ---
        if worker_id not in self.target_zip:
            if self.target_zip_path:
                src = _open_source(self.target_zip_path)
                # nested zip inside zip (original behaviour, only for actual zips)
                if self.target_zip_path_in_zip and isinstance(src, zipfile.ZipFile):
                    with src.open(str(self.target_zip_path_in_zip)) as nested:
                        nested_data = BytesIO(nested.read())
                    src.close()
                    src = zipfile.ZipFile(nested_data)
                self.target_zip[worker_id] = src
            else:
                self.target_zip[worker_id] = self.zip[worker_id]

        # --- instance target source (optional) ---
        if (
            self.target_instance_zip_path is not None
            and worker_id not in self.target_instance_zip
        ):
            self.target_instance_zip[worker_id] = _open_source(self.target_instance_zip_path)

        return (
            self.zip[worker_id],
            self.target_zip[worker_id],
            self.target_instance_zip[worker_id] if self.target_instance_zip_path else None,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _sort_key(m):
        filename = m.filename if hasattr(m, "filename") else str(m)
        match = re.search(r"\d+", filename)
        return (int(match.group()) if match else float("inf"), filename)

    @staticmethod
    def valid_member(
        img_info,
        img_folder_path_in_zip: Path,
        img_stem_suffix: str,
        img_suffix: str,
    ):
        return (
            Path(img_info.filename).is_relative_to(img_folder_path_in_zip)
            and img_info.filename.endswith(img_stem_suffix + img_suffix)
            and not img_info.is_dir()
        )

    def __len__(self):
        return len(self.imgs)

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def close(self):
        for store in (self.zip, self.target_zip, self.target_instance_zip):
            if store is not None:
                for handle in store.values():
                    handle.close()

        self.zip = None
        self.target_zip = None
        self.target_instance_zip = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["zip"] = None
        state["target_zip"] = None
        state["target_instance_zip"] = None
        return state
