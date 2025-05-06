# src/core/logging_utils.py
#
# Evo projesi için merkezi loglama yardımcı fonksiyonlarını içerir.
# Loglama sistemini yapılandırma dosyasına göre ayarlar.

import logging
import sys
import os

logger = logging.getLogger(__name__)

class WebAPIHandler(logging.Handler):
    """
    Log kayıtlarını HTTP POST ile bir web API'ye gönderen handler.
    """
    def __init__(self, url, level=logging.NOTSET):
        super().__init__(level)
        self.url = url

    def emit(self, record):
        try:
            import requests
            log_entry = self.format(record)
            requests.post(self.url, json={"log": log_entry, "level": record.levelname, "module": record.name})
        except Exception:
            self.handleError(record)

class SocketHandler(logging.Handler):
    """
    Log kayıtlarını bir TCP/UDP socket üzerinden gönderen handler.
    """
    def __init__(self, host, port, udp=False, level=logging.NOTSET):
        super().__init__(level)
        self.host = host
        self.port = port
        self.udp = udp

    def emit(self, record):
        import socket
        msg = self.format(record).encode("utf-8")
        try:
            if self.udp:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(msg, (self.host, self.port))
                sock.close()
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
                sock.sendall(msg)
                sock.close()
        except Exception:
            self.handleError(record)

class AnsiColorFormatter(logging.Formatter):
    COLOR_MAP = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[1;31m' # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)

class PlainFormatter(logging.Formatter):
    """Dosya için renksiz formatter."""
    def format(self, record):
        record.levelname = f"{record.levelname:<8}"
        return super().format(record)

def _get_module_logdir_and_file(logger_name, level_name):
    """Logger adından klasör ve dosya yolu üretir."""
    # Örn: src.memory.core, DEBUG → logs/memory/core/debug.log
    parts = logger_name.replace("src.", "").split(".")
    logdir = os.path.join("logs", *parts)
    os.makedirs(logdir, exist_ok=True)
    return os.path.join(logdir, f"{level_name.lower()}.log")

def setup_logging(config=None):
    """
    Evo projesinin loglama sistemini sağlanan yapılandırmaya göre ayarlar.
    Tüm handler'ları config'e göre kurar, modül ve seviye bazlı log dosyalarını oluşturur.
    Console çıktısı renkli olabilir, dosya çıktıları sade tutulur.
    """
    # Mevcut logger ayarlarını temizle
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # --- Logging Level Ayarları ---
    default_level = logging.INFO
    configured_level = default_level
    if config and 'logging' in config and 'level' in config['logging']:
        level_name = str(config['logging']['level']).upper()
        try:
            level_value = logging.getLevelName(level_name)
            if isinstance(level_value, int):
                configured_level = level_value
            else:
                logger.warning(f"Logging: Konfigurasyonda geçersiz log seviyesi adı: '{level_name}'. Varsayılan seviye ({logging.getLevelName(default_level)}) kullanılıyor.")
        except Exception as e:
            logger.warning(f"Logging: Konfigurasyondaki log seviyesi okunurken hata: {e}. Varsayılan seviye ({logging.getLevelName(default_level)}) kullanılıyor.", exc_info=True)
    root_logger.setLevel(configured_level)
    logger.info(f"Logging: Root logger seviyesi ayarlandı: {logging.getLevelName(root_logger.level)}")

    # --- Handler (Çıktı Hedefi) Ayarları ---
    os.makedirs("logs", exist_ok=True)
    handlers_config = []
    if config and 'logging' in config and 'handlers' in config['logging']:
        handlers_config = config['logging']['handlers']
    else:
        handlers_config = [{'type': 'console', 'level': logging.getLevelName(configured_level), 'color': True}]

    for hcfg in handlers_config:
        htype = hcfg.get('type', 'console')
        hlevel = logging.getLevelName(str(hcfg.get('level', 'INFO')).upper())
        color = hcfg.get('color', False)
        if htype == 'console':
            handler = logging.StreamHandler(sys.stdout)
            if color:
                handler.setFormatter(AnsiColorFormatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)-40s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                ))
            else:
                handler.setFormatter(PlainFormatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)-40s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                ))
        elif htype == 'file':
            filename = hcfg.get('filename', 'logs/evo_all.log')
            handler = logging.FileHandler(filename, encoding='utf-8')
            handler.setFormatter(PlainFormatter(
                fmt="%(asctime)s | %(levelname)s | %(name)-40s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
        elif htype == 'webapi':
            url = hcfg.get('url')
            if not url:
                logger.warning("Logging: webapi handler için 'url' belirtilmeli.")
                continue
            handler = WebAPIHandler(url)
            handler.setFormatter(PlainFormatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
        elif htype == 'socket':
            host = hcfg.get('host')
            port = hcfg.get('port')
            udp = hcfg.get('udp', False)
            if not host or not port:
                logger.warning("Logging: socket handler için 'host' ve 'port' belirtilmeli.")
                continue
            handler = SocketHandler(host, port, udp=udp)
            handler.setFormatter(PlainFormatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
        else:
            logger.warning(f"Logging: Bilinmeyen handler tipi: {htype}, atlanıyor.")
            continue
        handler.setLevel(hlevel)
        root_logger.addHandler(handler)
        
    # --- Modül ve seviye bazlı handler'lar ---
    module_levels = {}
    if config and 'logging' in config and 'modules' in config['logging']:
        module_levels = config['logging']['modules']

    for module_name, levels in module_levels.items():
        if isinstance(levels, str):
            levels = [levels]
        for level in levels:
            level_name = str(level).upper()
            mod_logger = logging.getLogger(module_name)
            mod_logfile = _get_module_logdir_and_file(module_name, level_name)
            handler = logging.FileHandler(mod_logfile, encoding='utf-8')
            handler.setFormatter(PlainFormatter(
                fmt="%(asctime)s | %(levelname)s | %(name)-40s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            handler.setLevel(logging.getLevelName(level_name))
            mod_logger.addHandler(handler)
            mod_logger.setLevel(logging.DEBUG)
            mod_logger.propagate = True
