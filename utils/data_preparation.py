from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from torchvision.datasets import ImageFolder  # type: ignore[import-untyped]

def get_dataset(folder: str) -> ImageFolder:
    '''
    Gets the dataset images and labels map from the specified folder.
    Args: 
        folder (str): Path to the dataset folder.
    Returns:
        images_list (list[str]): List of image file paths.
        labels_map (dict): Mapping from label indices to label names.
    '''
    dataset = ImageFolder(root=folder)

    return dataset

def train_test_val_split(dataset: ImageFolder, test_size: float=0.2, val_size: float=0.1, random_state: int=42, stratify: bool=True) -> tuple[list[str], list[str], list[str], list[int], list[int], list[int]]:
    '''
    Splits the dataset into training, validation, and test sets.
    Args:
        dataset (ImageFolder): The dataset to split.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
    '''
    images_list = [path for path, _ in dataset.imgs]
    label_list = [label for _, label in dataset.imgs]
    train_images, test_images, train_labels, test_labels = train_test_split(images_list, label_list, test_size=test_size, random_state=random_state, stratify=label_list if stratify else None)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=val_size, random_state=random_state, stratify=train_labels if stratify else None)
    print(f"After split - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    print('After split percentages - Train: {:.2f}%, Val: {:.2f}%, Test: {:.2f}%'.format(
        len(train_images) / len(images_list) * 100,
        len(val_images) / len(images_list) * 100,
        len(test_images) / len(images_list) * 100
    ))
    return train_images, val_images, test_images, train_labels, val_labels, test_labels
