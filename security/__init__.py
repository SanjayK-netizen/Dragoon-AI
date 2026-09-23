if __package__:
    from .security_utils import (
        get_secret,
        require_secret,
        constant_time_equal,
        log_safe,
        mask_secret,
        redact_dict,
        read_text_file,
        write_json_file,
        write_text_file,
    )
else:
    from security_utils import (
        get_secret,
        require_secret,
        constant_time_equal,
        log_safe,
        mask_secret,
        redact_dict,
        read_text_file,
        write_json_file,
        write_text_file,
    )

__all__ = [
    "get_secret",
    "require_secret",
    "constant_time_equal",
    "log_safe",
    "mask_secret",
    "redact_dict",
    "read_text_file",
    "write_json_file",
    "write_text_file",
]