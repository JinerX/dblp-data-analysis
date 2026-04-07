import csv
import gzip
import os
import tempfile
import logging

import duckdb
from lxml import etree

from src.constants import XML_GZ_PATH, ROOT_TAGS, DB_PATH
from src.utils.logger import get_logger
from typing import Optional, List, Tuple


logger = get_logger("alias_builder", logging.INFO)
BATCH_SIZE = 10000



def normalize_creator_name(name: str) -> Optional[str]:
    """
    Normalize an author/creator name string.

    This function performs basic normalization:
    - removes extra whitespace
    - strips leading/trailing whitespace
    - removes surrounding quotes

    Parameters
    ----------
    name : str
        Raw name string extracted from XML.

    Returns
    -------
    Optional[str]
        Normalized name, or None if the input is empty or invalid.
    """

    if name is None:
        return None

    name = " ".join(name.split())
    name = name.strip()
    name = name.strip("'\"“”‘’")

    if not name:
        return None

    return name



def build_aliases() -> None:
    """
    Build an author alias mapping from the DBLP XML dataset.

    This function:
    1. Parses the compressed XML dataset
    2. Extracts author aliases from homepage (`www`) entries
    3. Writes intermediate results to a temporary TSV file
    4. Loads the data into DuckDB
    5. Deduplicates aliases into a final `author_aliases` table

    The canonical name for each author group is chosen as the
    first author listed in a homepage entry.

    Tables created:
    - author_aliases_staging (temporary)
    - author_aliases (final mapping)
    
    Returns
    -------
    None
    """

    logger.info(f"Starting to scan {XML_GZ_PATH}")

    conn = duckdb.connect(DB_PATH)

    conn.execute("DROP TABLE IF EXISTS author_aliases_staging")
    conn.execute("DROP TABLE IF EXISTS author_aliases")
    conn.execute("""
        CREATE TABLE author_aliases_staging (
            alias_name TEXT,
            canonical_name TEXT
        );
    """)

    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "dblp_author_aliases.tsv")

    try:
        with gzip.open(XML_GZ_PATH, "rb") as f, open(tmp_path, "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out, delimiter="\t", quoting=csv.QUOTE_MINIMAL)

            context = etree.iterparse(
                f,
                events=("end",),
                tag=ROOT_TAGS,
                load_dtd=True,
                resolve_entities=True,
                no_network=True,
            )

            buffer = []
            processed_www = 0

            logger.info("Starting to alias")
            for _, elem in context:
                if elem.tag == "www":
                    key = elem.get("key") or ""
                    if key.startswith("homepages/"):
                        authors = [a.text for a in elem.iterchildren(tag="author") if a.text]
                        if authors:
                            canon_name = normalize_creator_name(authors[0])
                            buffer.extend((normalize_creator_name(author), canon_name) for author in authors)
                            processed_www += 1

                        if len(buffer) >= BATCH_SIZE:
                            writer.writerows(buffer)
                            buffer.clear()
                            logger.info(f"Processed {processed_www} homepage records")

                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

            if buffer:
                writer.writerows(buffer)

    except FileNotFoundError:
        logger.error(f"File not found {XML_GZ_PATH}")
        conn.close()
        return

    logger.info("Loading aliases into DuckDB")

    conn.execute(f"""
        COPY author_aliases_staging
        FROM '{tmp_path.replace("'", "''")}'
        (DELIMITER '\t', HEADER false, QUOTE '"', ESCAPE '"');
    """)

    conn.execute("""
        CREATE TABLE author_aliases AS
        SELECT DISTINCT alias_name, canonical_name FROM author_aliases_staging;
    """)

    total_aliases = conn.execute("SELECT COUNT(*) FROM author_aliases").fetchone()
    if total_aliases is None:
        logger.error("NO entries were added to the database")
        return
    
    logger.info(f"Aliasing complete with {total_aliases[0]} entries")

    logger.info(f"Removing aliases_staging table")
    conn.execute("DROP TABLE IF EXISTS author_aliases_staging")


    conn.close()


build_aliases()