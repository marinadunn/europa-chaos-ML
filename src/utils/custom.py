import transforms as T
import torchvision.transforms as PT

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        pass
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        # transforms.append(T.RandomBrightnessAdjustment(0.5))
    return T.Compose(transforms)

def get_transform_torch(train):
    transforms = []
    transforms.append(PT.ToTensor())
    # transforms.append(PT.Resize((1024, 1024)))
    # transforms.append(PT.Normalize(mean=[0.5], std=[0.5])) # does these valus even make sense
    # transforms.append(PT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])) # does these valus even make sense
    if train:
        transforms.append(PT.RandomHorizontalFlip(0.5))
        transforms.append(PT.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)))
        # transforms.append(PT.RandomResizedCrop(512))
        # transforms.append(PT.ColorJitter(brightness=(0.5,1.5)))

        transforms.append(PT.GaussianBlur(kernel_size=3))
    return PT.Compose(transforms)