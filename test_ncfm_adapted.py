"""
测试 NCFM 适配的医学数据集加载功能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'adapted', 'ncfm'))

from utils.utils import load_resized_data

# NCFM 与其他算法共享预处理后的 ImageFolder 根目录。
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data', 'prepared')

def test_pathmnist():
    print("=" * 60)
    print("测试 PathMNIST 数据加载")
    print("=" * 60)

    try:
        train_dataset, val_dataset = load_resized_data(
            dataset='pathmnist',
            data_dir=DATA_ROOT,
            size=32,
            nclass=9,
            load_memory=False,
            seed=0
        )

        print(f"[OK] PathMNIST 加载成功")
        print(f"  训练集大小: {len(train_dataset)}")
        print(f"  测试集大小: {len(val_dataset)}")
        print(f"  类别数: {train_dataset.nclass}")

        # 测试单个样本
        img, label = train_dataset[0]
        print(f"  样本形状: {img.shape}")
        print(f"  标签类型: {type(label)}, 值: {label}")

        # 验证标签是标量整数
        assert isinstance(label, (int, type(0))), f"标签应该是 int, 但是是 {type(label)}"
        print(f"  [OK] 标签类型正确 (标量整数)")

        return True
    except Exception as e:
        print(f"[FAIL] PathMNIST 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_covid():
    print("\n" + "=" * 60)
    print("测试 COVID 数据加载")
    print("=" * 60)

    covid_dir = os.path.join(DATA_ROOT, 'COVID', 'train')
    if not os.path.exists(covid_dir):
        print(f"[SKIP] COVID 数据集不存在: {covid_dir}")
        print("  跳过测试 (数据集未准备)")
        return None

    try:
        train_dataset, val_dataset = load_resized_data(
            dataset='covid',
            data_dir=DATA_ROOT,
            size=112,
            nclass=4,
            load_memory=False,
            seed=0
        )

        print(f"[OK] COVID 加载成功")
        print(f"  训练集大小: {len(train_dataset)}")
        print(f"  测试集大小: {len(val_dataset)}")
        print(f"  类别数: {train_dataset.nclass}")

        # 测试单个样本
        img, label = train_dataset[0]
        print(f"  样本形状: {img.shape}")
        print(f"  标签类型: {type(label)}, 值: {label}")

        assert isinstance(label, (int, type(0))), f"标签应该是 int, 但是是 {type(label)}"
        print(f"  [OK] 标签类型正确 (标量整数)")

        return True
    except Exception as e:
        print(f"[FAIL] COVID 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kvasir():
    print("\n" + "=" * 60)
    print("测试 Kvasir 数据加载")
    print("=" * 60)

    kvasir_dir = os.path.join(DATA_ROOT, 'Kvasir', 'train')
    if not os.path.exists(kvasir_dir):
        print(f"[SKIP] Kvasir 数据集不存在: {kvasir_dir}")
        print("  跳过测试 (数据集未准备)")
        return None

    try:
        train_dataset, val_dataset = load_resized_data(
            dataset='kvasir',
            data_dir=DATA_ROOT,
            size=128,
            nclass=8,
            load_memory=False,
            seed=0
        )

        print(f"[OK] Kvasir 加载成功")
        print(f"  训练集大小: {len(train_dataset)}")
        print(f"  测试集大小: {len(val_dataset)}")
        print(f"  类别数: {train_dataset.nclass}")

        # 测试单个样本
        img, label = train_dataset[0]
        print(f"  样本形状: {img.shape}")
        print(f"  标签类型: {type(label)}, 值: {label}")

        assert isinstance(label, (int, type(0))), f"标签应该是 int, 但是是 {type(label)}"
        print(f"  [OK] 标签类型正确 (标量整数)")

        return True
    except Exception as e:
        print(f"[FAIL] Kvasir 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("NCFM 医学数据集适配测试\n")

    results = {
        'PathMNIST': test_pathmnist(),
        'COVID': test_covid(),
        'Kvasir': test_kvasir()
    }

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for dataset, result in results.items():
        if result is True:
            status = "[OK] 通过"
        elif result is False:
            status = "[FAIL] 失败"
        else:
            status = "[SKIP] 跳过 (数据未准备)"
        print(f"{dataset:15s}: {status}")
