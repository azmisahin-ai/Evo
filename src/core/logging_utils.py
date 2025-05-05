# src/core/logging_utils.py
import logging
import sys # Konsol çıktısı için StreamHandler'a gerekli olabilir

def setup_logging(level=logging.INFO):
    """
    Configures the logging system for the Evo project.

    Sets up console logging with a specific format.
    Sets the overall logging level for the root logger.
    Specifically sets the 'src' package logger to DEBUG level
    to see detailed logs from our project modules.

    Args:
        level: The default minimum logging level to display (e.g., logging.INFO, logging.DEBUG).
               This is applied to the root logger.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    # Clear any existing handlers to prevent duplicate output if basicConfig was called elsewhere
    # (This is a good practice when taking full control of logging)
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Set the root logger level. Logs below this level are ignored by the root logger.
    root_logger.setLevel(level) # Başlangıçta temel seviye (INFO veya DEBUG)

    # Create a console handler
    # Use sys.stdout to ensure compatibility
    console_handler = logging.StreamHandler(sys.stdout)

    # Create a formatter
    # %(name)s will be the logger name (e.g., 'src.run_evo', 'src.senses.vision')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the handler
    console_handler.setFormatter(formatter)

    # Add the console handler to the root logger
    # Logs from any logger (including named loggers like 'src.run_evo') will propagate up to the root logger
    # and be handled by the console_handler, provided their own levels and the root level allow.
    root_logger.addHandler(console_handler)

    # --- Specific Logger Levels ---
    # Set the logging level specifically for the 'src' package and its submodules
    # This ensures that logs within our project code appear at DEBUG level,
    # even if the root logger level is set higher (e.g., INFO).
    # However, the *handler's* level must also be low enough.
    # Let's ensure the root logger's level is low enough to capture DEBUG logs propagating up.
    # A common pattern is to set root logger to DEBUG and handlers to a minimum level,
    # OR set specific loggers (like 'src') to DEBUG and ensure handlers are attached
    # to them or an ancestor.
    # For simplicity and to ensure *all* src DEBUG logs are *processed* by the root logger,
    # let's set the root logger level to DEBUG if the requested level is DEBUG.
    # Otherwise, if level is INFO, src DEBUG logs won't pass the root logger filter anyway.
    # The most reliable way to see src DEBUG logs is to set root logger to DEBUG.

    logging.getLogger('src').setLevel(logging.DEBUG)
    # Ensure root logger level is DEBUG if we want to see src DEBUG logs
    # If root_logger.setLevel(level) was called with INFO, this might override it,
    # but it guarantees DEBUG logs propagate to the root handler.
    # A cleaner way is to just set the root level to DEBUG always or conditionally based on config.
    # Let's modify setup_logging signature or add config loading later.
    # For now, let's ensure root is at least INFO and src is DEBUG,
    # and handler is on root. Debug logs from src will propagate to root,
    # but root's INFO level might filter them.
    # To guarantee src DEBUG logs, root *must* be DEBUG. Let's enforce this for now.

    root_logger.setLevel(logging.DEBUG) # <<< Force root logger to DEBUG to see all src DEBUG logs

# Ana çalıştırma noktası
if __name__ == '__main__':
    # Example logging to show it works
    logger = logging.getLogger(__name__) # Logger for logging_utils module itself
    logger.info("Logging system configured successfully.")
    logger.debug("This is a debug message from logging_utils.") # Should appear now