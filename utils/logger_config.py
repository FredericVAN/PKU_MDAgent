import logging
import coloredlogs
import sys


def setup_logger(log_file_path):
    # 配置 logger
    logging.basicConfig()
    logger = logging.getLogger(name='mylogger')
    logger.setLevel(logging.DEBUG)  # 设置全局最低日志级别

    # 配置 coloredlogs
    # coloredlogs.install(level=logging.DEBUG, logger=logger)
    logger.propagate = False

    # 配置颜色
    coloredFormatter = coloredlogs.ColoredFormatter(
        fmt='[%(name)s] %(asctime)s - PID:[%(process)d] - TNAME:%(threadName)s - FILE:%(filename)s -%(funcName)s %(lineno)-3d  %(message)s',
        level_styles=dict(
            debug=dict(color='white'),
            info=dict(color='blue'),
            warning=dict(color='yellow', bright=True),
            error=dict(color='red', bold=True, bright=True),
            critical=dict(color='black', bold=True, background='red'),
        ),
        field_styles=dict(
            name=dict(color='white'),
            asctime=dict(color='white'),
            funcName=dict(color='white'),
            lineno=dict(color='white'),
        )
    )

    # 配置 StreamHandler 输出到控制台
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(coloredFormatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # 配置 FileHandler 写入本地文件
    file_handler = logging.FileHandler(log_file_path)
    # 对文件日志可能不需要颜色，所以可以使用不同的Formatter，例如简单易读的Formatter
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # 可根据需求设定文件日志的最低记录级别
    logger.addHandler(file_handler)

    return logger


log_file = "../myrun.log"
logger = setup_logger(log_file)
if __name__=='__main__':
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')