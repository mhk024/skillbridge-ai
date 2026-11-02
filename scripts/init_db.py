"""Create the database, tables and load the question bank. Run from the project root:

    python -m scripts.init_db
"""

from sqlalchemy import inspect

from src.assessment.question_service import load_question_bank, sync_questions
from src.database.connection import get_engine, init_db, session_scope
from src.utils.config import get_settings


def main() -> None:
    settings = get_settings()
    engine = get_engine()
    init_db(engine)

    questions = load_question_bank(settings.questions_dir)
    with session_scope(engine) as session:
        result = sync_questions(session, questions)

    tables = sorted(inspect(engine).get_table_names())
    print(f"Database ready: {settings.database_url}")
    print("Tables:", ", ".join(tables))
    print(
        f"Questions: {len(questions)} in files "
        f"(created {result['created']}, updated {result['updated']}, unchanged {result['unchanged']})"
    )


if __name__ == "__main__":
    main()