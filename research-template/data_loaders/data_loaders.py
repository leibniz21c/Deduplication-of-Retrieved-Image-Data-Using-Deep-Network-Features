from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import ImageDataset, ProcessedImageDataset, ProcessedPairDataset

class ImageDataLoader(DataLoader):
    def __init__(
        self,
        root,
        batch_size=1,
        num_workers=0,
    ):
        trsfm = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
        self.__dataset = ImageDataset(root=root, transform=trsfm)
        super().__init__(
            self.__dataset, batch_size, shuffle=False, num_workers=num_workers
        )

class ProcessedPairDataLoader(DataLoader):
    def __init__(
        self,
        root,
        prep_name,
        batch_size=1,
        num_workers=0,
    ):
        self.__dataset = ProcessedPairDataset(root=root, prep_name=prep_name)
        super().__init__(
            self.__dataset, batch_size, shuffle=False, num_workers=num_workers
        )