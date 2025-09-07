#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "tiktoken>=0.5.0",
#   "networkx>=3.0",
#   "diskcache>=5.6.0",
#   "grep-ast>=0.3.0",
#   "tree-sitter>=0.20.0",
#   "pygments>=2.14.0",
# ]
# ///
"""
Standalone RepoMap Tool

A command-line tool that generates a "map" of a software repository,
highlighting important files and definitions based on their relevance.
Uses Tree-sitter for parsing and PageRank for ranking importance.
"""

import logging
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import namedtuple, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Callable

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken is required. Install with: pip install tiktoken")
    sys.exit(1)

try:
    import networkx as nx
except ImportError:
    print("Error: networkx is required. Install with: pip install networkx")
    sys.exit(1)

try:
    from grep_ast import TreeContext, filename_to_lang
    from grep_ast.tsl import get_language, get_parser
except ImportError:
    print("Error: grep-ast is required. Install with: pip install grep-ast")
    sys.exit(1)

try:
    import sqlite3
except ImportError:
    print("Error: sqlite3 is part of Python standard library")
    sys.exit(1)

try:
    from tree_sitter import QueryCursor, Query
except ImportError:
    print("Error: tree-sitter is required. Install with: pip install tree-sitter")
    sys.exit(1)

logger = logging.getLogger(__name__)

# Determine the script directory for reliable path resolution
SCRIPT_DIR = Path(__file__).resolve().parent

# Tag namedtuple for storing parsed code definitions and references
Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def read_text(
    filename: str, encoding: str = "utf-8", silent: bool = False
) -> Optional[str]:
    """Read text from file with error handling."""
    try:
        return Path(filename).read_text(encoding=encoding, errors="ignore")
    except FileNotFoundError:
        if not silent:
            logger.error(f"File not found: {filename}")
        return None
    except IsADirectoryError:
        if not silent:
            logger.error(f"Is a directory: {filename}")
        return None
    except OSError as e:
        if not silent:
            logger.error(f"Error reading {filename}: {e}")
        return None
    except UnicodeError as e:
        if not silent:
            logger.error(f"Error decoding {filename}: {e}")
        return None
    except Exception as e:
        if not silent:
            logger.error(f"Unexpected error reading {filename}: {e}")
        return None


def get_scm_fname(lang: str) -> Optional[str]:
    """Get the SCM query file for a language."""
    scm_files = {
        "arduino": "arduino-tags.scm",
        "chatito": "chatito-tags.scm",
        "commonlisp": "commonlisp-tags.scm",
        "cpp": "cpp-tags.scm",
        "csharp": "csharp-tags.scm",
        "c": "c-tags.scm",
        "dart": "dart-tags.scm",
        "d": "d-tags.scm",
        "elisp": "elisp-tags.scm",
        "elixir": "elixir-tags.scm",
        "elm": "elm-tags.scm",
        "gleam": "gleam-tags.scm",
        "go": "go-tags.scm",
        "javascript": "javascript-tags.scm",
        "java": "java-tags.scm",
        "lua": "lua-tags.scm",
        "ocaml_interface": "ocaml_interface-tags.scm",
        "ocaml": "ocaml-tags.scm",
        "pony": "pony-tags.scm",
        "properties": "properties-tags.scm",
        "python": "python-tags.scm",
        "racket": "racket-tags.scm",
        "r": "r-tags.scm",
        "ruby": "ruby-tags.scm",
        "rust": "rust-tags.scm",
        "solidity": "solidity-tags.scm",
        "swift": "swift-tags.scm",
        "udev": "udev-tags.scm",
        "c_sharp": "c_sharp-tags.scm",
        "hcl": "hcl-tags.scm",
        "kotlin": "kotlin-tags.scm",
        "php": "php-tags.scm",
        "ql": "ql-tags.scm",
        "scala": "scala-tags.scm",
        "typescript": "typescript-tags.scm",
    }

    if lang in scm_files:
        scm_filename = scm_files[lang]

        # Use the script's directory (determined at import time)
        # Search in tree-sitter-language-pack
        scm_path = SCRIPT_DIR / "queries" / "tree-sitter-language-pack" / scm_filename
        if scm_path.exists():
            return str(scm_path)

        # Search in tree-sitter-languages
        scm_path = SCRIPT_DIR / "queries" / "tree-sitter-languages" / scm_filename
        if scm_path.exists():
            return str(scm_path)

    return None


def find_src_files(directory: str) -> List[str]:
    """Find source files in a directory."""
    if not os.path.isdir(directory):
        return [directory] if os.path.isfile(directory) else []

    src_files = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-source directories
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d not in {"node_modules", "__pycache__", "venv", "env"}
        ]

        for file in files:
            if not file.startswith("."):
                full_path = os.path.join(root, file)
                src_files.append(full_path)

    return src_files


def tool_output(*messages):
    """Print informational messages."""
    logger.info(" ".join(messages))


def tool_warning(message):
    """Print warning messages."""
    logger.warning(message)


def tool_error(message):
    """Print error messages."""
    logger.error(message)


@dataclass
class FileReport:
    excluded: Dict[str, str]  # File -> exclusion reason with status
    definition_matches: int  # Total definition tags
    reference_matches: int  # Total reference tags
    total_files_considered: int  # Total files provided as input


SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError)


class RepoMap:
    """Main class for generating repository maps."""

    def __init__(
        self,
        map_tokens: int = 1024,
        root: str = None,
        token_counter_func: Callable[[str], int] = count_tokens,
        file_reader_func: Callable[[str], Optional[str]] = read_text,
        output_handler_funcs: Dict[str, Callable] = None,
        repo_content_prefix: Optional[str] = None,
        verbose: bool = False,
        max_context_window: Optional[int] = None,
        map_mul_no_files: int = 8,
        refresh: str = "auto",
        exclude_unranked: bool = False,
    ):
        """Initialize RepoMap instance."""
        self.map_tokens = map_tokens
        self.max_map_tokens = map_tokens
        self.root = Path(root or os.getcwd()).resolve()
        self.token_count_func_internal = token_counter_func
        self.read_text_func_internal = file_reader_func
        self.repo_content_prefix = repo_content_prefix
        self.verbose = verbose
        self.max_context_window = max_context_window
        self.map_mul_no_files = map_mul_no_files
        self.refresh = refresh
        self.exclude_unranked = exclude_unranked

        self.logger = logging.getLogger(self.__class__.__name__)

        # Set up output handlers
        if output_handler_funcs is None:
            output_handler_funcs = {"info": print, "warning": print, "error": print}
        self.output_handlers = output_handler_funcs

        # Initialize caches
        self.tree_context_cache = {}
        self.map_cache = {}

    def token_count(self, text: str) -> int:
        """Count tokens in text with sampling optimization for long texts."""
        if not text:
            return 0

        len_text = len(text)
        if len_text < 200:
            return self.token_count_func_internal(text)

        # Sample for longer texts
        lines = text.splitlines(keepends=True)
        num_lines = len(lines)

        step = max(1, num_lines // 100)
        sampled_lines = lines[::step]
        sample_text = "".join(sampled_lines)

        if not sample_text:
            return self.token_count_func_internal(text)

        sample_tokens = self.token_count_func_internal(sample_text)

        if len(sample_text) == 0:
            return self.token_count_func_internal(text)

        est_tokens = (sample_tokens / len(sample_text)) * len_text
        return int(est_tokens)

    def get_rel_fname(self, fname: str) -> str:
        """Get relative filename from absolute path."""
        try:
            return str(Path(fname).relative_to(self.root))
        except ValueError:
            return fname

    def get_mtime(self, fname: str) -> Optional[float]:
        """Get file modification time."""
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.output_handlers["warning"](f"File not found: {fname}")
            return None

    def get_tags(self, fname: str, rel_fname: str) -> List[Tag]:
        """Get tags for a file, using cache when possible."""
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        # Cache miss or file changed
        self.logger.debug(f"Calling get_tags_raw for {fname}")
        tags = self.get_tags_raw(fname, rel_fname)

        return tags

    def get_tags_raw(self, fname: str, rel_fname: str) -> List[Tag]:
        """Parse file to extract tags using Tree-sitter."""
        lang = filename_to_lang(fname)
        if not lang:
            return []

        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            self.output_handlers["error"](f"Skipping file {fname}: {err}")
            return []

        scm_fname = get_scm_fname(lang)
        if not scm_fname:
            return []

        code = self.read_text_func_internal(fname)
        if not code:
            return []

        try:
            tree = parser.parse(bytes(code, "utf-8"))

            # Load query from SCM file
            query_text = read_text(scm_fname, silent=True)
            if not query_text:
                return []

            query = Query(language, query_text)
            captures = QueryCursor(query).captures(tree.root_node)

            tags = []
            # Process captures as a dictionary
            for capture_name, nodes in captures.items():
                for node in nodes:
                    if "name.definition" in capture_name:
                        kind = "def"
                    elif "name.reference" in capture_name:
                        kind = "ref"
                    else:
                        # Skip other capture types like 'reference.call' if not needed for tagging
                        continue

                    line_num = node.start_point[0] + 1
                    # Handle potential None value
                    name = node.text.decode("utf-8") if node.text else ""

                    tags.append(
                        Tag(
                            rel_fname=rel_fname,
                            fname=fname,
                            line=line_num,
                            name=name,
                            kind=kind,
                        )
                    )

            return tags

        except Exception as e:
            self.output_handlers["error"](f"Error parsing {fname}: {e}")
            return []

    def get_ranked_tags(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
    ) -> Tuple[List[Tuple[float, Tag]], FileReport]:
        """Get ranked tags using PageRank algorithm with file report."""
        # Return empty list and empty report if no files
        if not chat_fnames and not other_fnames:
            return [], FileReport({}, 0, 0, 0)

        # Initialize file report early
        included: List[str] = []
        excluded: Dict[str, str] = {}
        total_definitions = 0
        total_references = 0
        if mentioned_fnames is None:
            mentioned_fnames = set()
        if mentioned_idents is None:
            mentioned_idents = set()

        # Normalize paths to absolute
        def normalize_path(path):
            return str(Path(path).resolve())

        chat_fnames = [normalize_path(f) for f in chat_fnames]
        other_fnames = [normalize_path(f) for f in other_fnames]

        # Initialize file report
        included: List[str] = []
        excluded: Dict[str, str] = {}
        total_definitions = 0
        total_references = 0

        # Collect all tags
        defines = defaultdict(set)
        references = defaultdict(set)
        definitions = defaultdict(set)

        personalization = {}
        chat_rel_fnames = {self.get_rel_fname(f) for f in chat_fnames}

        all_fnames = list(set(chat_fnames + other_fnames))

        for fname in all_fnames:
            rel_fname = self.get_rel_fname(fname)

            if not os.path.exists(fname):
                reason = "File not found"
                excluded[fname] = reason
                self.output_handlers["warning"](
                    f"Repo-map can't include {fname}: {reason}"
                )
                continue

            included.append(fname)

            tags = self.get_tags(fname, rel_fname)
            self.logger.info(f"Tags for {rel_fname}: {len(tags)}")

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    definitions[rel_fname].add(tag.name)
                    total_definitions += 1
                elif tag.kind == "ref":
                    references[tag.name].add(rel_fname)
                    total_references += 1

            # Set personalization for chat files
            if fname in chat_fnames:
                personalization[rel_fname] = 100.0

        # Build graph
        G = nx.MultiDiGraph()

        # Add nodes
        for fname in all_fnames:
            rel_fname = self.get_rel_fname(fname)
            G.add_node(rel_fname)

        # Add edges based on references
        for name, ref_fnames in references.items():
            def_fnames = defines.get(name, set())
            for ref_fname in ref_fnames:
                for def_fname in def_fnames:
                    if ref_fname != def_fname:
                        G.add_edge(ref_fname, def_fname, name=name)

        if not G.nodes():
            file_report = FileReport(
                excluded=excluded,
                definition_matches=total_definitions,
                reference_matches=total_references,
                total_files_considered=len(all_fnames),
            )
            return [], file_report

        # Run PageRank
        try:
            if personalization:
                ranks = nx.pagerank(G, personalization=personalization, alpha=0.85)
            else:
                ranks = {node: 1.0 for node in G.nodes()}
        except Exception:
            logger.error("Error with getting ranks")
            # Fallback to uniform ranking
            ranks = {node: 1.0 for node in G.nodes()}

        # Update excluded dictionary with status information
        for fname in set(chat_fnames + other_fnames):
            if fname in excluded:
                # Add status prefix to existing exclusion reason
                excluded[fname] = f"[EXCLUDED] {excluded[fname]}"
            elif fname not in included:
                excluded[fname] = (
                    "[NOT PROCESSED] File not included in final processing"
                )

        # Create file report
        file_report = FileReport(
            excluded=excluded,
            definition_matches=total_definitions,
            reference_matches=total_references,
            total_files_considered=len(all_fnames),
        )

        # Collect and rank tags
        ranked_tags = []

        for fname in included:
            rel_fname = self.get_rel_fname(fname)
            file_rank = ranks.get(rel_fname, 0.0)

            # Exclude files with low Page Rank if exclude_unranked is True
            if (
                self.exclude_unranked and file_rank <= 0.0001
            ):  # Use a small threshold to exclude near-zero ranks
                continue

            tags = self.get_tags(fname, rel_fname)
            for tag in tags:
                if tag.kind == "def":
                    # Boost for mentioned identifiers
                    boost = 1.0
                    if tag.name in mentioned_idents:
                        boost *= 10.0
                    if rel_fname in mentioned_fnames:
                        boost *= 5.0
                    if rel_fname in chat_rel_fnames:
                        boost *= 20.0

                    final_rank = file_rank * boost
                    ranked_tags.append((final_rank, tag))

        # Sort by rank (descending)
        ranked_tags.sort(key=lambda x: x[0], reverse=True)

        return ranked_tags, file_report

    def render_tree(self, abs_fname: str, rel_fname: str, lois: List[int]) -> str:
        """Render a code snippet with specific lines of interest."""
        code = self.read_text_func_internal(abs_fname)
        if not code:
            return ""

        # Use TreeContext for rendering
        try:
            if rel_fname not in self.tree_context_cache:
                self.tree_context_cache[rel_fname] = TreeContext(
                    rel_fname, code, color=False
                )

            tree_context = self.tree_context_cache[rel_fname]
            return tree_context.format(lois)
        except Exception:
            # Fallback to simple line extraction
            lines = code.splitlines()
            result_lines = [f"{rel_fname}:"]

            for loi in sorted(set(lois)):
                if 1 <= loi <= len(lines):
                    result_lines.append(f"{loi:4d}: {lines[loi-1]}")

            return "\n".join(result_lines)

    def to_tree(self, tags: List[Tuple[float, Tag]], chat_rel_fnames: Set[str]) -> str:
        """Convert ranked tags to formatted tree output."""
        if not tags:
            return ""

        # Group tags by file
        file_tags = defaultdict(list)
        for rank, tag in tags:
            file_tags[tag.rel_fname].append((rank, tag))

        # Sort files by importance (max rank of their tags)
        sorted_files = sorted(
            file_tags.items(),
            key=lambda x: max(rank for rank, tag in x[1]),
            reverse=True,
        )

        tree_parts = []

        for rel_fname, file_tag_list in sorted_files:
            # Get lines of interest
            lois = [tag.line for rank, tag in file_tag_list]

            # Find absolute filename
            abs_fname = str(self.root / rel_fname)

            # Get the max rank for the file
            max_rank = max(rank for rank, tag in file_tag_list)

            # Render the tree for this file
            rendered = self.render_tree(abs_fname, rel_fname, lois)
            if rendered:
                # Add rank value to the output
                rendered_lines = rendered.splitlines()
                first_line = rendered_lines[0]
                code_lines = rendered_lines[1:]

                tree_parts.append(
                    f"{first_line}\n"
                    f"(Rank value: {max_rank:.4f})\n\n" + "\n".join(code_lines)
                )

        return "\n\n".join(tree_parts)

    def get_ranked_tags_map(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        max_map_tokens: int,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        force_refresh: bool = False,
    ) -> Tuple[Optional[str], FileReport]:
        """Get the ranked tags map with caching."""
        cache_key = (
            tuple(sorted(chat_fnames)),
            tuple(sorted(other_fnames)),
            max_map_tokens,
            tuple(sorted(mentioned_fnames or [])),
            tuple(sorted(mentioned_idents or [])),
        )

        if not force_refresh and cache_key in self.map_cache:
            return self.map_cache[cache_key]

        result = self.get_ranked_tags_map_uncached(
            chat_fnames,
            other_fnames,
            max_map_tokens,
            mentioned_fnames,
            mentioned_idents,
        )

        self.map_cache[cache_key] = result
        return result

    def get_ranked_tags_map_uncached(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        max_map_tokens: int,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the ranked tags map without caching."""
        ranked_tags, file_report = self.get_ranked_tags(
            chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
        )
        self.logger.info(f"Ranked tags count: {len(ranked_tags)}")

        if not ranked_tags:
            # Generate basic file information when no tags are found
            basic_map = []
            for fname in other_fnames:
                rel_fname = self.get_rel_fname(fname)
                basic_map.append(f"{rel_fname}:")
                basic_map.append("(No code definitions found)")
                basic_map.append("")
            return "\n".join(basic_map), file_report

        # Binary search to find the right number of tags
        chat_rel_fnames = {self.get_rel_fname(f) for f in chat_fnames}

        def try_tags(num_tags: int) -> Tuple[Optional[str], int]:
            if num_tags <= 0:
                return None, 0

            selected_tags = ranked_tags[:num_tags]
            tree_output = self.to_tree(selected_tags, chat_rel_fnames)

            if not tree_output:
                return None, 0

            tokens = self.token_count(tree_output)
            return tree_output, tokens

        # Binary search for optimal number of tags
        left, right = 0, len(ranked_tags)
        best_tree = None

        while left <= right:
            mid = (left + right) // 2
            tree_output, tokens = try_tags(mid)

            if tree_output and tokens <= max_map_tokens:
                best_tree = tree_output
                left = mid + 1
            else:
                right = mid - 1

        return best_tree, file_report

    def get_repo_map(
        self,
        chat_files: List[str] = None,
        other_files: List[str] = None,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        force_refresh: bool = False,
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the repository map with file report."""
        if chat_files is None:
            chat_files = []
        if other_files is None:
            other_files = []

        # Create empty report for error cases
        empty_report = FileReport({}, 0, 0, 0)

        if self.max_map_tokens <= 0 or not other_files:
            return None, empty_report

        # Adjust max_map_tokens if no chat files
        max_map_tokens = self.max_map_tokens
        if not chat_files and self.max_context_window:
            padding = 1024
            available = self.max_context_window - padding
            max_map_tokens = min(max_map_tokens * self.map_mul_no_files, available)

        try:
            # get_ranked_tags_map returns (map_string, file_report)
            map_string, file_report = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
                force_refresh,
            )
        except RecursionError:
            self.output_handlers["error"]("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return None, FileReport({}, 0, 0, 0)  # Ensure consistent return type

        if map_string is None:
            self.logger.warning("map_string is None")
            return None, file_report

        if self.verbose:
            tokens = self.token_count(map_string)
            self.output_handlers["info"](f"Repo-map: {tokens / 1024:.1f} k-tokens")

        # Format final output
        other = "other " if chat_files else ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += map_string

        return repo_content, file_report


def main():
    """Main CLI entry point."""
    parser = ArgumentParser(
        description="Generate a repository map showing important code structures.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .                    # Map current directory
  %(prog)s src/ --map-tokens 2048  # Map src/ with 2048 token limit
  %(prog)s file1.py file2.py    # Map specific files
  %(prog)s --chat-files main.py --other-files src/  # Specify chat vs other files
        """,
    )

    parser.add_argument(
        "paths", nargs="*", help="Files or directories to include in the map"
    )

    parser.add_argument(
        "--root",
        default=".",
        help="Repository root directory (default: current directory)",
    )

    parser.add_argument(
        "--map-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for the generated map (default: 8192)",
    )

    parser.add_argument(
        "--chat-files",
        nargs="*",
        help="Files currently being edited (given higher priority)",
    )

    parser.add_argument(
        "--other-files", nargs="*", help="Other files to consider for the map"
    )

    parser.add_argument(
        "--mentioned-files",
        nargs="*",
        help="Files explicitly mentioned (given higher priority)",
    )

    parser.add_argument(
        "--mentioned-idents",
        nargs="*",
        help="Identifiers explicitly mentioned (given higher priority)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (-v: INFO, -vv: DEBUG, default: ERROR)",
    )

    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Model name for token counting (default: gpt-4)",
    )

    parser.add_argument(
        "--max-context-window", type=int, help="Maximum context window size"
    )

    parser.add_argument(
        "--force-refresh", action="store_true", help="Force refresh of caches"
    )

    parser.add_argument(
        "--exclude-unranked",
        action="store_true",
        help="Exclude files with Page Rank 0 from the map",
    )

    args = parser.parse_args()

    # Configure logging level based on verbosity
    if args.verbose == 0:
        log_level = logging.ERROR
    elif args.verbose == 1:
        log_level = logging.INFO
    else:  # args.verbose >= 2
        log_level = logging.DEBUG

    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Set up token counter with specified model
    def token_counter(text: str) -> int:
        return count_tokens(text, args.model)

    # Set up output handlers
    output_handlers = {
        "info": tool_output,
        "warning": tool_warning,
        "error": tool_error,
    }

    # Process file arguments
    chat_files_from_args = (
        args.chat_files or []
    )  # These are the paths as strings from the CLI

    # Determine the list of unresolved path specifications that will form the 'other_files'
    # These can be files or directories. find_src_files will expand them.
    unresolved_paths_for_other_files_specs = []
    if args.other_files:  # If --other-files is explicitly provided, it's the source
        unresolved_paths_for_other_files_specs.extend(args.other_files)
    elif args.paths:  # Else, if positional paths are given, they are the source
        unresolved_paths_for_other_files_specs.extend(args.paths)
    # If neither, unresolved_paths_for_other_files_specs remains empty.

    # Now, expand all directory paths in unresolved_paths_for_other_files_specs into actual file lists
    # and collect all file paths. find_src_files handles both files and directories.
    effective_other_files_unresolved = []
    for path_spec_str in unresolved_paths_for_other_files_specs:
        effective_other_files_unresolved.extend(find_src_files(path_spec_str))

    # Convert to absolute paths
    root_path = Path(args.root).resolve()
    # chat_files for RepoMap are from --chat-files argument, resolved.
    chat_files = [str(Path(f).resolve()) for f in chat_files_from_args]
    # other_files for RepoMap are the effective_other_files, resolved after expansion.
    other_files = [str(Path(f).resolve()) for f in effective_other_files_unresolved]

    logger.info(f"Chat files: {chat_files}")
    logger.info(f"Other files: {other_files}")

    # Convert mentioned files to sets
    mentioned_fnames = set(args.mentioned_files) if args.mentioned_files else None
    mentioned_idents = set(args.mentioned_idents) if args.mentioned_idents else None

    # Create RepoMap instance
    repo_map = RepoMap(
        map_tokens=args.map_tokens,
        root=str(root_path),
        token_counter_func=token_counter,
        file_reader_func=read_text,
        output_handler_funcs=output_handlers,
        verbose=bool(args.verbose),
        max_context_window=args.max_context_window,
        exclude_unranked=args.exclude_unranked,
    )

    # Generate the map
    try:
        map_content, _ = repo_map.get_repo_map(
            chat_files=chat_files,
            other_files=other_files,
            mentioned_fnames=mentioned_fnames,
            mentioned_idents=mentioned_idents,
            force_refresh=args.force_refresh,
        )

        if map_content:
            if args.verbose:
                tokens = repo_map.token_count(map_content)
                tool_output(
                    f"Generated map: {len(map_content)} chars, ~{tokens} tokens"
                )

            print(map_content)
        else:
            tool_output("No repository map generated.")

    except KeyboardInterrupt:
        tool_error("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        tool_error(f"Error generating repository map: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
