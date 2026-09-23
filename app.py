"""Command-line entry point for the working Dragoon assistant."""

from Tests.main import main, process_turn, setup_logging


__all__ = ["main", "process_turn", "setup_logging"]


if __name__ == "__main__":
    main()
