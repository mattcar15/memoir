"""
Alembic migrations for Memoir storage.

Usage:
    # Run migrations for the default database (logs/memoir.db)
    python -m memoir.storage.migrations upgrade head
    
    # Run migrations for a specific database
    MEMOIR_DATABASE_URL=sqlite:///path/to/memoir.db python -m memoir.storage.migrations upgrade head
    
    # Show current version
    python -m memoir.storage.migrations current
    
    # Create a new migration (after modifying models)
    python -m memoir.storage.migrations revision --autogenerate -m "description"
"""

