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
from datetime import datetime
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


def initialize_database(db_path: str) -> None:
    with DatabaseConnection(db_path) as cursor:
        try:
            # Create the main notes table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created DATETIME NOT NULL,
                    updated DATETIME NOT NULL,
                    folder TEXT,
                    content TEXT NOT NULL,
                    content_embeddings BLOB
                )
            """
            )

            # Create the VSS virtual table
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS vss_notes USING vec0(
                    content_embedding float[384]
                )
                """
            )
        except sqlite3.Error as e:
            logging.error(f"Error creating tables: {e}")
            raise


def add_note(db_path: str, title: str, folder: str, content: str) -> None:
    with DatabaseConnection(db_path) as cursor:
        try:
            created = datetime.now()
            updated = datetime.now()
            embeddings = get_embeddings(content)

            # Insert into main notes table
            cursor.execute(
                """
                INSERT INTO notes (title, created, updated, folder, content, content_embeddings)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (title, created, updated, folder, content, embeddings),
            )

            # Get the rowid of the inserted note
            note_id = cursor.lastrowid

            # Insert into VSS virtual table
            cursor.execute(
                """
                INSERT INTO vss_notes(rowid, content_embedding)
                VALUES (?, ?)
                """,
                (note_id, embeddings),
            )

            logging.info(f"Added note: {title}")
        except sqlite3.Error as e:
            logging.error(f"Error adding note: {e}")
            raise


def find_similar_notes(db_path: str, query_embedding, limit: int = 5) -> list:
    with DatabaseConnection(db_path) as cursor:
        try:
            cursor.execute(
                """
                select
                    n.title,
                    n.folder,
                    n.content,
                    distance
                from vss_notes
                JOIN notes n ON n.id = rowid
                where content_embedding match ?
                and k = ?
                order by distance;
                """,
                (query_embedding, limit),
            )
            return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error searching notes: {e}")
            raise


def main(args: argparse.Namespace) -> None:
    query = "Car Insurance Company"
    embedded_query = get_embeddings(query)
    similar_notes = find_similar_notes(args.db_path, embedded_query)
    logging.info(f"\nSimilar notes for query '{query}':")
    for title, folder, content, distance in similar_notes:
        logging.info(f"Title: {title}, Folder: {folder}, Distance: {distance:.4f}")
        # logging.info(f"Content: {content}\n")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
