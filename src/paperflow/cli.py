"""
Command-line interface for paperflow.
"""
import argparse

from paperflow import PaperPipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Search for academic papers and display results in a table"
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["arxiv"],
        help="Sources to search (default: arxiv)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results (default: 10)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for PDF extraction (if available)"
    )

    args = parser.parse_args()

    pipeline = PaperPipeline(gpu=args.gpu)
    results = pipeline.search(
        args.query,
        sources=args.sources,
        max_results=args.max_results
    )

    print(results)


if __name__ == "__main__":
    main()