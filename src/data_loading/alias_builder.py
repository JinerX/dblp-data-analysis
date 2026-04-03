import csv
import gzip
import os
import tempfile
import logging

import duckdb
from lxml import etree

from src.constants import XML_GZ_PATH, ROOT_TAGS, DB_PATH
from src.utils.logger import get_logger

logger = get_logger("alias_builder", logging.INFO)
BATCH_SIZE = 10000


def build_aliases():
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
                            canon_name = authors[0]
                            buffer.extend((author, canon_name) for author in authors)
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

    conn.close()


build_aliases()