#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "sentence_transformers",
#   "sqlite-vec",
# ]
# ///
"""
A simple script with vector similarity search support

Usage:
./sqlite-vector.py -h

./sqlite-vector.py -v # To log INFO messages
./sqlite-vector.py -vv # To log DEBUG messages
"""

import argparse
import logging
import sqlite3
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from typing import Optional

import sqlite_vec
from sentence_transformers import SentenceTransformer


def setup_logging(verbosity: int) -> None:
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
        ],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    logging.captureWarnings(capture=True)


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="notes.db",
        help="Path to the SQLite database file",
    )
    return parser.parse_args()


model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(text: str) -> bytes:
    return model.encode(text).tobytes()


class DatabaseConnection:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> sqlite3.Cursor:
        try:
            self.conn = sqlite3.connect(self.db_path)
            # Enable and load the vector search extension
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            return self.conn.cursor()
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()


def find_similar_notes(db_path: str, query, limit: int = 5) -> list:
    with DatabaseConnection(db_path) as cursor:
        try:
            cursor.execute(
                """
with vec_matches as (
  select
    rowid,
    row_number() over (order by distance) as rank_number,
    distance
  from vss_notes
  where
    content_embedding match :embedding
    and k = :limit
),
-- the FTS5 search results
fts_matches as (
  select
    rowid,
    row_number() over (order by rank) as rank_number,
    rank as score
  from fts_notes
  where content match :query
  limit :limit
),
-- combine FTS5 + vector search results with RRF
final as (
  select
    notes.id as id,
    notes.title as title,
    notes.content as content,
    notes.created as created_at,
    notes.folder as folder,
    vec_matches.rank_number as vec_rank,
    fts_matches.rank_number as fts_rank,
    (
      coalesce(1.0 / (:rrf_k + fts_matches.rank_number), 0.0) * :weight_fts +
      coalesce(1.0 / (:rrf_k + vec_matches.rank_number), 0.0) * :weight_vec
    ) as combined_rank,
    vec_matches.distance as vec_distance,
    fts_matches.score as fts_score
  from fts_matches
  full outer join vec_matches on vec_matches.rowid = fts_matches.rowid
  join notes on notes.id = coalesce(fts_matches.rowid, vec_matches.rowid)
  order by combined_rank desc
)
select title, folder, content, created_at from final;
                """,
                {
                    "query": query,
                    "embedding": get_embeddings(query),
                    "limit": limit,
                    "rrf_k": 60,
                    "weight_fts": 1.0,
                    "weight_vec": 1.0,
                },
            )
            return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error searching notes: {e}")
            raise


def main(args: argparse.Namespace) -> None:
    query = "Tell me about Car Insurance"
    similar_notes = find_similar_notes(args.db_path, query, limit=2)
    logging.info(f"\nSimilar notes for query '{query}':")
    # for i in similar_notes:
    #     print(i)
    for title, folder, content, created_at in similar_notes:
        logging.info(f"Title: {title}, Folder: {folder}, Created At : {created_at}")
        # logging.info(f"Content: {content}\n")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
