from __future__ import annotations

from dotenv import load_dotenv

from src.db.client import run_sql_file


def main() -> None:
    load_dotenv()
    run_sql_file()
    print("Database migrations applied successfully.")


if __name__ == "__main__":
    main()
