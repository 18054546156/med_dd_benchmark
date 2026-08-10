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
        # 医疗合成张量已经由 data.utils 按合同完成 resize/normalize；
        # from_tensor=True 时只做可微增强，避免在 evaluation 阶段重复归一化。
        medical_sizes = {"pathmnist": 32, "covid": 112, "kvasir": 128}
        medical_mean = {
            "pathmnist": MEANS["pathmnist"],
            "covid": MEANS["covid"],
            "kvasir": MEANS["kvasir"],
        }
        medical_std = {
            "pathmnist": STDS["pathmnist"],
            "covid": STDS["covid"],
            "kvasir": STDS["kvasir"],
        }
        cast = [] if from_tensor else [transforms.ToTensor()]
        resize = [] if from_tensor else [
            transforms.Resize((size or medical_sizes[dataset], size or medical_sizes[dataset]))
        ]
        normalize = [] if from_tensor else [
            transforms.Normalize(medical_mean[dataset], medical_std[dataset])
        ]
        aug = [transforms.RandomHorizontalFlip()] if augment else []
        train_transform = transforms.Compose(resize + cast + aug + normalize)
        test_transform = transforms.Compose(resize + cast + normalize)
        _ = test_transform
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_transform, _
