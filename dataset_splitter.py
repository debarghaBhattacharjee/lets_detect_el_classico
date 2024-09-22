import random
from pathlib import Path
from shutil import copy2
from tqdm import tqdm

class DatasetSplitter:
    def __init__(self, 
                 source_dirs, 
                 output_dir, 
                 train_ratio, 
                 val_ratio, 
                 test_ratio):
        """
        This class initializes the DatasetSplitter with 
        user-defined ratios and directories.

        INPUTS:
            - source_dirs: List of directories containing images.
            - output_dir: The directory where the split image sets will be stored.
            - train_ratio: Ratio of images to be used for training.
            - val_ratio: Ratio of images to be used for validation.
            - test_ratio: Ratio of images to be used for testing.
        """
        self.source_dirs = [Path(dir) for dir in source_dirs]
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_image_paths(self):
        """
        This method collects all image paths from the 
        source directories.
        """
        image_paths = []
        for directory in self.source_dirs:
            # We assume input images are in JPEG format.
            image_paths.extend(directory.glob("*.jpg"))
        return image_paths

    def split_dataset(self, image_paths):
        """
         This method splits the dataset into train, 
         validation, and test sets.
        """
        # Shuffle the image list.
        random.shuffle(image_paths)

        # Determine split indices.
        total_images = len(image_paths)
        train_idx = int(total_images * self.train_ratio)
        val_idx = train_idx + int(total_images * self.val_ratio)

        # Split the list.
        train_images = image_paths[:train_idx]
        val_images = image_paths[train_idx:val_idx]
        test_images = image_paths[val_idx:]

        return (
            train_images, 
            val_images, 
            test_images
        )

    def copy_images(self, 
                    image_paths, 
                    subset):
        """
        This method copies the images to the appropriate 
        directory (train, val, test).
        """
        subset_dir = self.output_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm(image_paths, unit=" image"):
            destination = subset_dir / image_path.name
            copy2(image_path, destination)

        return

    def execute(self):
        """
        Tis method executes the image splitting and 
        copying process.
        """
        # Gather all image paths.
        image_paths = self.get_image_paths()

        # Split the images into train, val, and test.
        train_images, \
        val_images, \
        test_images = self.split_dataset(image_paths)

        # Copy images to their respective directories.
        self.copy_images(train_images, "train")
        self.copy_images(val_images, "val")
        self.copy_images(test_images, "test")

        return


if __name__ == "__main__":
    source_directories = [
        "./dataset/images/src/bvr-2006_07", 
        "./dataset/images/src/bvr-2010_11",
        "./dataset/images/src/bvr-2015_16", 
        "./dataset/images/src/bvr-2016_17"
    ]
    output_directory = "./dataset/images/dest"
    train_val_test_ratio = (0.6, 0.2, 0.2)  # 60% train, 20% val, 20% test

    splitter = DatasetSplitter(
        source_directories, 
        output_directory, 
        *train_val_test_ratio
    )
    splitter.execute()
