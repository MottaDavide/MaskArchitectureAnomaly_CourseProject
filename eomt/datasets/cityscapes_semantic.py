# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms


def _resolve_cityscapes_source(base: Path, zip_name: str, folder_name: str) -> Path:
    """
    Returns the path to use as the data source, preferring an already-extracted
    folder over the zip file.

    Search order:
      1. <base>/<folder_name>/          — extracted folder  ✅ fastest
      2. <base>/<zip_name>              — zip file          ✅ works, but slower
      3. FileNotFoundError

    Examples
    --------
    _resolve_cityscapes_source(
        Path("/content/cityscapes"),
        "leftImg8bit_trainvaltest.zip",
        "leftImg8bit",
    )
    # → Path("/content/cityscapes/leftImg8bit")   if that dir exists
    # → Path("/content/cityscapes/leftImg8bit_trainvaltest.zip")  otherwise
    """
    folder_path = base / folder_name
    zip_path = base / zip_name

    if folder_path.is_dir():
        print(f"[CityscapesSemantic] Using extracted folder: {folder_path}")
        return folder_path

    if zip_path.is_file():
        print(f"[CityscapesSemantic] Using zip file: {zip_path}")
        return zip_path

    raise FileNotFoundError(
        f"Cityscapes source not found.\n"
        f"  Expected extracted folder : {folder_path}\n"
        f"  Expected zip file         : {zip_path}\n"
        f"Please extract the dataset or place the zip under '{base}'."
    )


class CityscapesSemantic(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        masks, labels = [], []
        for label_id in target[0].unique():
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)
            if cls is None or cls.ignore_in_eval:
                continue
            masks.append(target[0] == label_id)
            labels.append(cls.train_id)
        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        base = Path(self.path)

        # ------------------------------------------------------------------
        # Auto-detect: prefer extracted folder, fall back to zip
        # ------------------------------------------------------------------
        img_source = _resolve_cityscapes_source(
            base,
            zip_name="leftImg8bit_trainvaltest.zip",
            folder_name="leftImg8bit",
        )
        target_source = _resolve_cityscapes_source(
            base,
            zip_name="gtFine_trainvaltest.zip",
            folder_name="gtFine",
        )

        # ------------------------------------------------------------------
        # Build Dataset kwargs that work for both zip and folder.
        #
        # When the source is a folder, `img_folder_path_in_zip` is used as a
        # relative prefix that must match the files found inside the folder.
        # The layout is the same whether extracted or zipped:
        #   leftImg8bit/train/<city>/<file>.png
        #   gtFine/train/<city>/<file>.png
        #
        # For a zip the prefix includes the top-level dir ("leftImg8bit/...").
        # For an extracted folder the root IS leftImg8bit/, so the prefix
        # should NOT include it — just "train/..." would be needed.
        #
        # We handle this by checking what source type we got:
        # ------------------------------------------------------------------
        using_folder = img_source.is_dir()

        if using_folder:
            # Folder root IS leftImg8bit/ and gtFine/ respectively.
            img_folder_train    = Path("train")
            img_folder_val      = Path("val")
            target_folder_train = Path("train")
            target_folder_val   = Path("val")
        else:
            # Inside the zip, paths start with leftImg8bit/ or gtFine/
            img_folder_train    = Path("leftImg8bit/train")
            img_folder_val      = Path("leftImg8bit/val")
            target_folder_train = Path("gtFine/train")
            target_folder_val   = Path("gtFine/val")

        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            # zip_path / target_zip_path now accept both zip files and dirs
            "zip_path": img_source,
            "target_zip_path": target_source,
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }

        self.cityscapes_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=img_folder_train,
            target_folder_path_in_zip=target_folder_train,
            **cityscapes_dataset_kwargs,
        )
        self.cityscapes_val_dataset = Dataset(
            img_folder_path_in_zip=img_folder_val,
            target_folder_path_in_zip=target_folder_val,
            **cityscapes_dataset_kwargs,
        )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.cityscapes_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cityscapes_val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
