import sys

from app.container import create_rag_service


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m app.rag.ask \"your question\"")
        return

    question = " ".join(sys.argv[1:])

    rag = create_rag_service()

    response = rag.answer(question)

    print()
    print("Answer:")
    print(response.answer)

    print()
    print("Sources:")

    for source in response.sources:
        print(f"- {source}")


if __name__ == "__main__":
    main()
