import csv
import gzip
import logging
import os
import tempfile

import duckdb
from lxml import etree

from src.constants import XML_GZ_PATH, ROOT_TAGS, DB_PATH
from src.utils.logger import get_logger


logger = get_logger("person_work_builder", logging.INFO)

BATCH_SIZE = 20000


def safe_text(elem, tag_name):
    value = elem.findtext(tag_name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def sql_escape_path(path):
    return path.replace("'", "''")


def normalize_creator_name(name):
    if name is None:
        return None

    name = " ".join(name.split())
    name = name.strip()
    name = name.strip("'\"“”‘’")

    if not name:
        return None

    return name




def build_person_and_work_tables(
    db_path = DB_PATH,
    compressed_path = XML_GZ_PATH,
):
    conn = duckdb.connect(db_path)

    conn.execute("DROP TABLE IF EXISTS person")
    conn.execute("DROP TABLE IF EXISTS work")
    conn.execute("DROP TABLE IF EXISTS person_names_staging")
    conn.execute("DROP TABLE IF EXISTS work_staging")

    conn.execute("""
        CREATE TABLE person_names_staging (
            name TEXT
        );
    """)

    conn.execute("""
        CREATE TABLE work (
            work_id BIGINT PRIMARY KEY,
            dblp_key TEXT,
            record_type TEXT,
            title TEXT,
            year INTEGER,
            mdate TEXT,
            volume TEXT,
            number TEXT,
            pages TEXT,
            journal TEXT,
            booktitle TEXT,
            publisher TEXT,
            school TEXT,
            series TEXT,
            crossref TEXT,
            publtype TEXT,
            url TEXT,
            ee TEXT,
            isbn TEXT
        );
    """)

    work_fd, work_tsv = tempfile.mkstemp(prefix="dblp_work_", suffix=".tsv")
    names_fd, names_tsv = tempfile.mkstemp(prefix="dblp_names_", suffix=".tsv")
    os.close(work_fd)
    os.close(names_fd)

    logger.info("Starting XML scan")

    try:
        with gzip.open(compressed_path, "rb") as f, \
             open(work_tsv, "w", newline="", encoding="utf-8") as work_out, \
             open(names_tsv, "w", newline="", encoding="utf-8") as names_out:

            work_writer = csv.writer(work_out, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            names_writer = csv.writer(names_out, delimiter="\t", quoting=csv.QUOTE_MINIMAL)

            context = etree.iterparse(
                f,
                events=("end",),
                tag=ROOT_TAGS,
                load_dtd=True,
                resolve_entities=True,
                no_network=True,
            )

            work_buffer = []
            names_buffer = []
            processed_work = 0

            for _, elem in context:
                try:
                    if elem.tag != "www":
                        work_id = processed_work + 1
                        work_buffer.append((
                            work_id,
                            elem.get("key"),
                            elem.tag,
                            safe_text(elem, "title"),
                            int(y) if (y := safe_text(elem, "year")) and y.isdigit() else None,
                            elem.get("mdate"),
                            safe_text(elem, "volume"),
                            safe_text(elem, "number"),
                            safe_text(elem, "pages"),
                            safe_text(elem, "journal"),
                            safe_text(elem, "booktitle"),
                            safe_text(elem, "publisher"),
                            safe_text(elem, "school"),
                            safe_text(elem, "series"),
                            safe_text(elem, "crossref"),
                            elem.get("publtype"),
                            safe_text(elem, "url"),
                            safe_text(elem, "ee"),
                            safe_text(elem, "isbn"),
                        ))

                        for role in ("author", "editor"):
                            for node in elem.findall(role):
                                if node.text:
                                    name = normalize_creator_name(node.text)
                                    if name:
                                        names_buffer.append((name,))

                        processed_work += 1

                        if len(work_buffer) >= BATCH_SIZE:
                            work_writer.writerows(work_buffer)
                            work_buffer.clear()
                            logger.info(f"Written work entries into the tsv, processedworks: {processed_work}")

                        if len(names_buffer) >= BATCH_SIZE:
                            names_writer.writerows(names_buffer)
                            names_buffer.clear()
                            logger.info(f"Written name entries into the tsv, processedworks: {processed_work}")

                finally:
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

            if work_buffer:
                work_writer.writerows(work_buffer)

            if names_buffer:
                names_writer.writerows(names_buffer)

        logger.info("Loading staging tables with COPY")

        conn.execute(f"""
            COPY person_names_staging
            FROM '{sql_escape_path(names_tsv)}'
            (DELIMITER '\t', HEADER false, QUOTE '"', ESCAPE '"');
        """)

        conn.execute(f"""
            COPY work
            FROM '{sql_escape_path(work_tsv)}'
            (DELIMITER '\t', HEADER false, QUOTE '"', ESCAPE '"');
        """)

        logger.info("Building final person table")
        conn.execute("""
            CREATE TABLE person AS
            SELECT
                row_number() OVER (ORDER BY canonical_name) AS person_id,
                canonical_name
            FROM (
                SELECT DISTINCT
                    COALESCE(aa.canonical_name, p.name) AS canonical_name
                FROM person_names_staging p
                LEFT JOIN author_aliases aa
                    ON aa.alias_name = p.name
            ) x
            ORDER BY canonical_name;
        """)

        person_call = conn.execute("SELECT COUNT(*) FROM person").fetchone()
        work_call = conn.execute("SELECT COUNT(*) FROM work").fetchone()
        if person_call is not None:
            person_count = person_call[0]
        if work_call is not None:
            work_count = work_call[0]

        logging.info(f"Inserted {person_count} people")
        logging.info(f"Inserted {work_count} works")

        logging.info("Deleting staging tables")
        conn.execute("DROP TABLE IF EXISTS person_names_staging")
        conn.execute("DROP TABLE IF EXISTS work_staging")

        return conn

    finally:
        try:
            os.remove(work_tsv)
        except OSError:
            pass
        try:
            os.remove(names_tsv)
        except OSError:
            pass

def build_work_contributor_and_relation_tables(
    db_path=DB_PATH,
    compressed_path=XML_GZ_PATH,
):
    conn = duckdb.connect(db_path)

    conn.execute("DROP TABLE IF EXISTS work_contributor")
    conn.execute("DROP TABLE IF EXISTS work_contributor_staging")
    conn.execute("DROP TABLE IF EXISTS work_relation")

    conn.execute("""
        CREATE TABLE work_contributor_staging (
            work_key TEXT,
            contributor_name TEXT,
            role TEXT,
            position INTEGER
        );
    """)

    contrib_fd, contrib_tsv = tempfile.mkstemp(
        prefix="dblp_work_contrib_",
        suffix=".tsv"
    )
    os.close(contrib_fd)

    logger.info("Starting XML scan for work_contributor")

    try:
        with gzip.open(compressed_path, "rb") as f, \
             open(contrib_tsv, "w", newline="", encoding="utf-8") as contrib_out:

            contrib_writer = csv.writer(
                contrib_out,
                delimiter="\t",
                quoting=csv.QUOTE_MINIMAL
            )

            context = etree.iterparse(
                f,
                events=("end",),
                tag=ROOT_TAGS,
                load_dtd=True,
                resolve_entities=True,
                no_network=True,
            )

            contrib_buffer = []
            processed_work = 0

            for _, elem in context:
                try:
                    if elem.tag != "www":
                        work_key = elem.get("key")
                        if work_key:
                            for role in ("author", "editor"):
                                position = 1
                                for node in elem.findall(role):
                                    if node.text:
                                        contrib_buffer.append((
                                            work_key,
                                            node.text.strip(),
                                            role,
                                            position
                                        ))
                                        position += 1

                            processed_work += 1

                            if len(contrib_buffer) >= BATCH_SIZE:
                                contrib_writer.writerows(contrib_buffer)
                                contrib_buffer.clear()
                                logger.info(
                                    f"Written contributor entries into the tsv, "
                                    f"processed works: {processed_work}"
                                )

                finally:
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

            if contrib_buffer:
                contrib_writer.writerows(contrib_buffer)

        logger.info("Loading work_contributor staging table with COPY")
        conn.execute(f"""
            COPY work_contributor_staging
            FROM '{sql_escape_path(contrib_tsv)}'
            (DELIMITER '\t', HEADER false, QUOTE '"', ESCAPE '"');
        """)

        logger.info("Building final work_contributor table")
        conn.execute("""
            CREATE TABLE work_contributor AS
            WITH resolved AS (
                SELECT
                    s.work_key,
                    COALESCE(aa.canonical_name, s.contributor_name) AS canonical_name,
                    s.role,
                    s.position
                FROM work_contributor_staging s
                LEFT JOIN author_aliases aa
                    ON aa.alias_name = s.contributor_name
            )
            SELECT
                w.work_id,
                p.person_id,
                r.role,
                r.position
            FROM resolved r
            JOIN work w
                ON w.dblp_key = r.work_key
            JOIN person p
                ON p.canonical_name = r.canonical_name;
        """)

        logger.info("Building final work_relation table")
        conn.execute("""
            CREATE TABLE work_relation AS
            SELECT
                child.work_id AS child_work_id,
                parent.work_id AS parent_work_id,
                'crossref' AS relation_type
            FROM work child
            JOIN work parent
                ON parent.dblp_key = child.crossref
            WHERE child.crossref IS NOT NULL;
        """)

        contrib_call = conn.execute(
            "SELECT COUNT(*) FROM work_contributor"
        ).fetchone()
        rel_call = conn.execute(
            "SELECT COUNT(*) FROM work_relation"
        ).fetchone()
        if contrib_call is not None:
            contrib_count = contrib_call[0]
        if rel_call is not None:
            rel_count = rel_call[0]

        logger.info(f"Inserted {contrib_count} work_contributor rows")
        logger.info(f"Inserted {rel_count} work_relation rows")

        logging.info("Deleting staging tables")
        conn.execute("DROP TABLE IF EXISTS work_contributor_staging")
        return conn

    finally:
        try:
            os.remove(contrib_tsv)
        except OSError:
            pass



build_person_and_work_tables()
build_work_contributor_and_relation_tables()