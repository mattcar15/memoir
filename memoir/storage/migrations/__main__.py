"""
CLI entry point for running Alembic migrations.

Usage:
    python -m memoir.storage.migrations upgrade head
    python -m memoir.storage.migrations current
    python -m memoir.storage.migrations revision --autogenerate -m "description"
"""

import os
import sys
from pathlib import Path

from alembic.config import Config
from alembic import command


def get_alembic_config() -> Config:
    """Get Alembic config pointing to our migrations."""
    migrations_dir = Path(__file__).parent
    alembic_ini = migrations_dir / "alembic.ini"
    
    config = Config(str(alembic_ini))
    config.set_main_option("script_location", str(migrations_dir))
    
    return config


def main():
    """Run Alembic with our configuration."""
    config = get_alembic_config()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python -m memoir.storage.migrations <command> [options]")
        print()
        print("Commands:")
        print("  upgrade head         Run all pending migrations")
        print("  downgrade -1         Revert the last migration")
        print("  current              Show current revision")
        print("  history              Show migration history")
        print("  revision -m 'msg'    Create a new migration")
        print("  revision --autogenerate -m 'msg'  Auto-generate migration from model changes")
        sys.exit(1)
    
    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    if cmd == "upgrade":
        revision = args[0] if args else "head"
        command.upgrade(config, revision)
    elif cmd == "downgrade":
        revision = args[0] if args else "-1"
        command.downgrade(config, revision)
    elif cmd == "current":
        command.current(config)
    elif cmd == "history":
        command.history(config)
    elif cmd == "revision":
        # Parse revision args
        message = None
        autogenerate = False
        i = 0
        while i < len(args):
            if args[i] in ("-m", "--message"):
                message = args[i + 1]
                i += 2
            elif args[i] in ("--autogenerate", "-a"):
                autogenerate = True
                i += 1
            else:
                i += 1
        
        command.revision(config, message=message, autogenerate=autogenerate)
    elif cmd == "heads":
        command.heads(config)
    elif cmd == "branches":
        command.branches(config)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()

