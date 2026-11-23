import os


def safe_path(path: str) -> str:
    """增强型路径处理"""
    try:
        # 基本验证
        if not isinstance(path, str):
            raise TypeError("路径必须是字符串类型")
        if len(path.strip()) == 0:
            raise ValueError("路径不能为空")

        # 替换非法字符
        path = path.strip()
        path = path.replace('“', '"').replace('”', '"')

        # 标准化路径
        normalized = os.path.normpath(path).replace('\\', '/')
        return normalized
    except Exception as e:
        raise RuntimeError(f"路径处理失败: {str(e)}") from e


def create_directories():
    """安全创建目录"""
    dirs = [
        'data/processed_features',
        'data/train_val_splits',
        'checkpoints',
        'results/figures',
        'logs'
    ]
    for d in dirs:
        try:
            os.makedirs(safe_path(d), exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"目录创建失败 {d}: {str(e)}")