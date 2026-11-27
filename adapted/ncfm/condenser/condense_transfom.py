from data.transform import (
    transform_imagenet,
    transform_cifar,
    transform_svhn,
    transform_mnist,
    transform_fashion,
    transform_tiny,
)
import torchvision.transforms as transforms
from data.dataset_statistics import MEANS, STDS


def get_train_transform(
    dataset,
    augment=True,
    from_tensor=True,
    size=0,
    rrc=False,
    rrc_size=None,
    device="cpu",
    medical_normalize=None,
):
    if dataset in [
        "imagenette",
        "imagewoof",
        "imagemeow",
        "imagesquawk",
        "imagefruit",
        "imageyellow",
        "imagenet",
    ]:
        train_transform, _ = transform_imagenet(
            augment=augment,
            from_tensor=from_tensor,
            size=size,
            rrc=rrc,
            rrc_size=rrc_size,
            device=device,
        )
    elif dataset[:5] == "cifar":
        train_transform, _ = transform_cifar(augment=augment, from_tensor=from_tensor)
    elif dataset == "svhn":
        train_transform, _ = transform_svhn(augment=augment, from_tensor=from_tensor)
    elif dataset == "mnist":
        train_transform, _ = transform_mnist(augment=augment, from_tensor=from_tensor)
    elif dataset == "fashion":
        train_transform, _ = transform_fashion(augment=augment, from_tensor=from_tensor)
    elif dataset == "tinyimagenet":
        train_transform, _ = transform_tiny(augment=augment, from_tensor=from_tensor)
    elif dataset in ["pathmnist", "covid", "kvasir"]:
        medical_sizes = {"pathmnist": 32, "covid": 112, "kvasir": 128}
        cast = [] if from_tensor else [transforms.ToTensor()]
        resize = [] if from_tensor else [
            transforms.Resize((size or medical_sizes[dataset], size or medical_sizes[dataset]))
        ]
        # Other callers retain their existing tensor preprocessing contract.
        if medical_normalize is None:
            medical_normalize = not from_tensor
        normalize = [
            transforms.Normalize(MEANS[dataset], STDS[dataset])
        ] if medical_normalize else []
        aug = [transforms.RandomHorizontalFlip()] if augment else []
        train_transform = transforms.Compose(resize + cast + aug + normalize)
        test_transform = transforms.Compose(resize + cast + normalize)
        _ = test_transform
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_transform, _
