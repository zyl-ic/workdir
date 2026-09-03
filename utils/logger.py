import csv
import logging
import sys
from pathlib import Path


class Logger:
    """训练统计日志：控制台输出 + 可选 CSV 落盘。

    用法：
        logger = Logger(log_dir="logs/3m_s1")
        logger.log_stat("loss", 0.5, t_env=1000)
        logger.info("Updated target network")
        logger.save()   # 训练结束时把累积统计写入 logs/3m_s1/stats.csv
    """

    def __init__(self, log_dir=None):
        self.console_logger = self._make_console_logger()
        self.records = []  # [(t_env, name, value), ...]
        self.log_dir = Path(log_dir) if log_dir else None

    @staticmethod
    def _make_console_logger():
        logger = logging.getLogger("marl")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # 避免重复输出（不往 root logger 传播）
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            ))
            logger.addHandler(handler)
        return logger

    def log_stat(self, name, value, t_env):
        value = float(value)
        t_env = int(t_env)
        self.records.append((t_env, name, value))
        self.console_logger.info("t_env=%8d  %-22s %.5f" % (t_env, name, value))

    def info(self, msg):
        self.console_logger.info(msg)

    def save(self):
        """把累积的统计写入 CSV（若配置了 log_dir）。"""
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "stats.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_env", "name", "value"])
            writer.writerows(self.records)
        self.info(f"Saved stats to {path}")
