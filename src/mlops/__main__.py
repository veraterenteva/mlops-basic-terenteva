"""Allow ``python -m mlops`` as a shortcut for the CLI."""

from mlops.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
