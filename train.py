import os
import argparse
import logging
import sys
import locale
from modules.path_utils import safe_path
from modules import preprocess, features, train_utils


class ASCIIFilter(logging.Filter):
    def filter(self, record):
        replacements = {
            '✅': '[OK]',
            '❌': '[ERROR]',
            '🔄': '[PROCESS]',
            '🏋️': '[TRAIN]',
            '🎉': '[SUCCESS]'
        }
        for uni, asc in replacements.items():
            record.msg = record.msg.replace(uni, asc)
        return True


def configure_logging():
    """配置支持多平台的日志系统"""
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        pass

    if sys.platform.startswith('win'):
        if sys.version_info >= (3, 7):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        os.system('chcp 65001 > nul')

    handlers = [
        logging.FileHandler(
            safe_path("logs/training.log"),
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )

    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.addFilter(ASCIIFilter())


def main():
    try:
        configure_logging()
        logging.info("=== 系统启动 ===")

        # 参数解析
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", required=True, help="原始数据路径")
        parser.add_argument("--full_training", action="store_true", help="启用全量数据集训练模式")
        parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
        parser.add_argument("--tune", action="store_true", help="在全量模式下搜索 lr/batch_size（仅用验证集）")
        parser.add_argument("--lr", type=float, default=1e-4, help="学习率（未启用 --tune 时生效）")
        parser.add_argument("--batch_size", type=int, default=32, help="批大小")
        parser.add_argument("--epochs", type=int, default=150, help="最大训练轮数")
        parser.add_argument("--patience", type=int, default=30, help="早停耐心值")
        args = parser.parse_args()

        # 固定随机种子：特征提取的增强、划分、初始化、shuffle 都受它影响
        train_utils.set_global_seed(args.seed)
        logging.info("[OK] 随机种子固定为 %d", args.seed)

        # 路径处理
        input_path = safe_path(args.data_path)
        logging.info("[OK] 验证路径通过: %s", input_path)

        # 特征提取
        logging.info("[PROCESS] 开始特征提取...")
        output_csv_train = safe_path("data/processed_features/train_data.csv")
        output_csv_val = safe_path("data/processed_features/val_data.csv")
        output_csv_test = safe_path("data/processed_features/test_data.csv")
        preprocess.convert_wav_to_csv(input_path, output_csv_train, output_csv_val, output_csv_test)
        logging.info("[OK] 训练特征已保存至: %s", output_csv_train)
        logging.info("[OK] 验证特征已保存至: %s", output_csv_val)
        logging.info("[OK] 测试特征已保存至: %s", output_csv_test)

        # 训练模式选择
        if args.full_training:
            logging.info("[PROCESS] 启动全量训练模式...")
            lr = args.lr
            batch_size = args.batch_size
            if args.tune:
                logging.info("[TUNE] 开始超参数搜索（仅使用训练/验证集）...")
                best = train_utils.tune_full_hyperparams(
                    train_csv=output_csv_train,
                    test_csv=output_csv_test,
                    seed=args.seed,
                    epochs=min(args.epochs, 80),
                    patience=min(args.patience, 15),
                )
                lr = best["lr"]
                batch_size = best["batch_size"]
                logging.info("[TUNE] 采用最优配置: lr=%g, batch_size=%d", lr, batch_size)

            train_utils.train_full(
                train_csv=output_csv_train,
                test_csv=output_csv_test,
                lr=lr,
                batch_size=batch_size,
                seed=args.seed,
                epochs=args.epochs,
                patience=args.patience,
            )
            logging.info("[SUCCESS] 全量训练完成！")
        else:
            if args.tune:
                logging.warning("[TUNE] 超参搜索仅支持 --full_training，五折模式已跳过")
            logging.info("[PROCESS] 生成五折交叉验证...")
            features.create_stratified_folds(
                output_csv_train, safe_path("data/train_val_splits"), seed=args.seed
            )
            logging.info("[OK] 五折数据生成完成")

            logging.info("[TRAIN] 开始交叉验证训练...")
            for fold in range(5):
                logging.info("\n=== 训练第 %d 折 ===", fold)
                train_csv = safe_path(f"data/train_val_splits/train_fold{fold}.csv")
                test_csv = safe_path(f"data/train_val_splits/test_fold{fold}.csv")
                train_utils.train_fold(
                    fold, train_csv, test_csv,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    epochs=min(args.epochs, 100),
                    patience=min(args.patience, 20),
                )
                logging.info("[OK] 第 %d 折训练完成", fold)
            logging.info("[SUCCESS] 交叉验证训练完成！")

    except Exception as e:
        logging.error("[ERROR] 系统错误: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
    # 4.2
