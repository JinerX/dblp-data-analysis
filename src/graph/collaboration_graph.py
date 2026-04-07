import math
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt



def resolve_author_id(conn, author_name: str):
    """
    Try to resolve an author name to a person_id.
    First checks person.canonical_name, then author_aliases.alias_name.
    Returns None if not found.
    """
    row = conn.execute(
        """
        SELECT person_id
        FROM person
        WHERE canonical_name = ?
        LIMIT 1
        """,
        [author_name],
    ).fetchone()

    if row is not None:
        return row[0]

    row = conn.execute(
        """
        SELECT p.person_id
        FROM author_aliases aa
        JOIN person p
          ON aa.canonical_name = p.canonical_name
        WHERE aa.alias_name = ?
        LIMIT 1
        """,
        [author_name],
    ).fetchone()

    if row is not None:
        return row[0]

    return None


def fetch_collaboration_edges(
    conn,
    person_ids,
    min_year=None,
    max_year=None,
    mode="incident",
):
    """
    Fetch coauthor edges from DuckDB.

    mode="incident":
        return edges where at least one endpoint is in person_ids
        useful for expanding the BFS frontier

    mode="induced":
        return edges where both endpoints are in person_ids
        useful for building the final subgraph
    """
    person_ids = list(dict.fromkeys(person_ids))
    if not person_ids:
        return pd.DataFrame(columns=["person_a", "person_b", "weight"])

    placeholders = ",".join(["?"] * len(person_ids))

    if mode == "incident":
        where_ids = f"(wc1.person_id IN ({placeholders}) OR wc2.person_id IN ({placeholders}))"
        params = person_ids + person_ids
    elif mode == "induced":
        where_ids = f"(wc1.person_id IN ({placeholders}) AND wc2.person_id IN ({placeholders}))"
        params = person_ids + person_ids
    else:
        raise ValueError("mode must be 'incident' or 'induced'")

    year_clause = ""
    year_params = []
    if min_year is not None:
        year_clause += " AND w.year >= ?"
        year_params.append(min_year)
    if max_year is not None:
        year_clause += " AND w.year <= ?"
        year_params.append(max_year)

    sql = f"""
        SELECT
            LEAST(wc1.person_id, wc2.person_id) AS person_a,
            GREATEST(wc1.person_id, wc2.person_id) AS person_b,
            COUNT(DISTINCT wc1.work_id) AS weight
        FROM work_contributor wc1
        JOIN work_contributor wc2
          ON wc1.work_id = wc2.work_id
         AND wc1.person_id < wc2.person_id
        JOIN work w
          ON wc1.work_id = w.work_id
        WHERE {where_ids}
        {year_clause}
        GROUP BY 1, 2
        ORDER BY weight DESC, person_a, person_b
    """

    return conn.execute(sql, params + year_params).fetch_df()


def fetch_person_names(conn, person_ids):
    """
    Fetch person_id -> canonical_name mapping.
    """
    person_ids = list(dict.fromkeys(person_ids))
    if not person_ids:
        return pd.DataFrame(columns=["person_id", "canonical_name"])

    placeholders = ",".join(["?"] * len(person_ids))
    return conn.execute(
        f"""
        SELECT person_id, canonical_name
        FROM person
        WHERE person_id IN ({placeholders})
        """,
        person_ids,
    ).fetch_df()


def build_collaboration_graph(
    conn,
    author_name: str,
    max_hops: int = 2,
    min_year=None,
    max_year=None,
    max_nodes: int = 250,
):
    """
    Build a local collaboration graph around one author.

    Returns:
      G: networkx.Graph with node attributes:
         - name
         - distance
         - degree
    """
    seed_id = resolve_author_id(conn, author_name)
    if seed_id is None:
        raise ValueError(f"Author not found: {author_name}")

    visited = {seed_id}
    distances = {seed_id: 0}
    frontier = {seed_id}

    for hop in range(1, max_hops + 1):
        if not frontier:
            break

        edges = fetch_collaboration_edges(
            conn,
            list(frontier),
            min_year=min_year,
            max_year=max_year,
            mode="incident",
        )

        new_frontier = set()

        for row in edges.itertuples(index=False):
            a = int(row.person_a)
            b = int(row.person_b)

            if a not in visited:
                visited.add(a)
                distances[a] = hop
                new_frontier.add(a)

            if b not in visited:
                visited.add(b)
                distances[b] = hop
                new_frontier.add(b)

            if len(visited) >= max_nodes:
                break

        frontier = new_frontier

        if len(visited) >= max_nodes:
            break

    induced_edges = fetch_collaboration_edges(
        conn,
        list(visited),
        min_year=min_year,
        max_year=max_year,
        mode="induced",
    )

    names_df = fetch_person_names(conn, list(visited))
    name_map = dict(zip(names_df["person_id"], names_df["canonical_name"]))

    G = nx.Graph()

    for pid in visited:
        G.add_node(
            pid,
            name=name_map.get(pid, str(pid)),
            distance=distances.get(pid, None),
        )

    for row in induced_edges.itertuples(index=False):
        a = int(row.person_a)
        b = int(row.person_b)
        w = int(row.weight)
        if a in G and b in G:
            G.add_edge(a, b, weight=w)

    degree_map = dict(G.degree())
    nx.set_node_attributes(G, degree_map, "degree")

    return G


def plot_collaboration_graph(
    G: nx.Graph,
    center_author: str = None,
    figsize=(14, 10),
    seed: int = 42,
    show_labels: bool = True,
    label_distance: int = 1,
):
    """
    Visualize the collaboration graph.

    Styling:
      - node size = degree
      - node color = distance from seed author
      - edge width = collaboration weight
    """
    if G.number_of_nodes() == 0:
        print("Graph is empty.")
        return

    plt.figure(figsize=figsize)

    k = 1 / math.sqrt(max(G.number_of_nodes(), 1))
    pos = nx.spring_layout(G, seed=seed, k=k)

    distances = nx.get_node_attributes(G, "distance")
    degrees = nx.get_node_attributes(G, "degree")

    node_order = list(G.nodes())
    node_sizes = [
        300 + 120 * degrees.get(n, 1)
        for n in node_order
    ]
    node_colors = [
        distances.get(n, 999)
        for n in node_order
    ]

    edge_widths = [
        0.5 + 0.8 * math.sqrt(G[u][v].get("weight", 1))
        for u, v in G.edges()
    ]

    edges = nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.35,
        width=edge_widths,
    )

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=node_order,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        linewidths=0.8,
        edgecolors="black",
    )

    if show_labels:
        labels = {}
        for n in G.nodes():
            dist = distances.get(n, None)
            if dist == 0 or dist is not None and dist <= label_distance:
                labels[n] = G.nodes[n].get("name", str(n))

        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=8,
        )

    cbar = plt.colorbar(nodes)
    cbar.set_label("Distance from seed author")

    title = "Collaboration graph"
    if center_author is not None:
        title = f"Collaboration graph around {center_author}"
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()




def resolve_work_id(conn, paper):
    """
    Resolve a paper identifier to work_id.

    Accepted inputs:
    - int -> treated as work_id directly
    - str containing digits -> treated as work_id if exact match exists
    - str title -> exact case-insensitive title match
    """
    if isinstance(paper, int):
        row = conn.execute(
            "SELECT work_id FROM work WHERE work_id = ?",
            [paper],
        ).fetchone()
        return row[0] if row else None

    paper = str(paper).strip()

    if paper.isdigit():
        row = conn.execute(
            "SELECT work_id FROM work WHERE work_id = ?",
            [int(paper)],
        ).fetchone()
        if row:
            return row[0]

    row = conn.execute(
        """
        SELECT work_id
        FROM work
        WHERE LOWER(title) = LOWER(?)
        LIMIT 1
        """,
        [paper],
    ).fetchone()

    if row:
        return row[0]

    return None

def get_work_authors(conn, work_id):
    """
    Return a DataFrame with authors of a given work:
    person_id, canonical_name, position
    """
    return conn.execute(
        """
        SELECT
            p.person_id,
            p.canonical_name,
            wc.position
        FROM work_contributor wc
        JOIN person p
          ON wc.person_id = p.person_id
        WHERE wc.work_id = ?
        ORDER BY wc.position, p.canonical_name
        """,
        [work_id],
    ).fetch_df()


def get_author_neighborhood(
    conn,
    seed_author_ids,
    max_hops=2,
    min_year=None,
    max_year=None,
):
    """
    Return a dict: author_id -> distance from seed set.
    """
    visited = set(seed_author_ids)
    distances = {aid: 0 for aid in seed_author_ids}
    frontier = set(seed_author_ids)

    for hop in range(1, max_hops + 1):
        if not frontier:
            break

        placeholders = ",".join(["?"] * len(frontier))

        year_clause = ""
        params = list(frontier)

        if min_year is not None:
            year_clause += " AND w.year >= ?"
            params.append(min_year)
        if max_year is not None:
            year_clause += " AND w.year <= ?"
            params.append(max_year)

        neighbors_df = conn.execute(
            f"""
            SELECT DISTINCT
                CASE
                    WHEN wc1.person_id IN ({placeholders}) THEN wc2.person_id
                    ELSE wc1.person_id
                END AS neighbor_id
            FROM work_contributor wc1
            JOIN work_contributor wc2
              ON wc1.work_id = wc2.work_id
             AND wc1.person_id <> wc2.person_id
            JOIN work w
              ON wc1.work_id = w.work_id
            WHERE (wc1.person_id IN ({placeholders}) OR wc2.person_id IN ({placeholders}))
            {year_clause}
            """,
            params + list(frontier) + list(frontier) + ([] if min_year is None and max_year is None else []),
        ).fetch_df()

        new_frontier = set()

        for aid in neighbors_df["neighbor_id"].dropna().astype(int):
            if aid not in visited:
                visited.add(aid)
                distances[aid] = hop
                new_frontier.add(aid)

        frontier = new_frontier

    return distances


def recommend_papers_from_paper(
    conn,
    paper,
    top_n=10,
    max_hops=2,
    min_year=None,
    max_year=None,
    exclude_seed_author_papers=False,
):
    """
    Recommend papers related to a seed paper using collaboration proximity.

    Parameters
    ----------
    paper : int or str
        work_id or exact title
    top_n : int
        maximum number of recommendations
    max_hops : int
        how far to expand in the collaboration graph
    exclude_seed_author_papers : bool
        if True, exclude papers that share any author with the seed paper

    Returns
    -------
    pandas.DataFrame with:
      work_id, title, year, score, shared_nearby_authors, min_author_distance, authors
    """
    seed_work_id = resolve_work_id(conn, paper)
    if seed_work_id is None:
        raise ValueError(f"Paper not found: {paper}")

    seed_meta = conn.execute(
        """
        SELECT work_id, title, year
        FROM work
        WHERE work_id = ?
        """,
        [seed_work_id],
    ).fetch_df().iloc[0]

    seed_authors_df = get_work_authors(conn, seed_work_id)
    if seed_authors_df.empty:
        return pd.DataFrame(columns=[
            "work_id", "title", "year", "score",
            "shared_nearby_authors", "min_author_distance", "authors"
        ])

    seed_author_ids = seed_authors_df["person_id"].astype(int).tolist()
    seed_author_set = set(seed_author_ids)

    author_dist = get_author_neighborhood(
        conn,
        seed_author_ids,
        max_hops=max_hops,
        min_year=min_year,
        max_year=max_year,
    )

    neighborhood_author_ids = list(author_dist.keys())
    if not neighborhood_author_ids:
        return pd.DataFrame(columns=[
            "work_id", "title", "year", "score",
            "shared_nearby_authors", "min_author_distance", "authors"
        ])

    placeholders = ",".join(["?"] * len(neighborhood_author_ids))

    year_clause = ""
    params = list(neighborhood_author_ids)
    if min_year is not None:
        year_clause += " AND w.year >= ?"
        params.append(min_year)
    if max_year is not None:
        year_clause += " AND w.year <= ?"
        params.append(max_year)

    candidates = conn.execute(
        f"""
        SELECT DISTINCT
            w.work_id,
            w.title,
            w.year,
            p.person_id,
            p.canonical_name
        FROM work_contributor wc
        JOIN work w
          ON wc.work_id = w.work_id
        JOIN person p
          ON wc.person_id = p.person_id
        WHERE wc.person_id IN ({placeholders})
          AND w.title IS NOT NULL
          AND TRIM(w.title) <> ''
          {year_clause}
        ORDER BY w.work_id, p.canonical_name
        """,
        params,
    ).fetch_df()

    if candidates.empty:
        return pd.DataFrame(columns=[
            "work_id", "title", "year", "score",
            "shared_nearby_authors", "min_author_distance", "authors"
        ])

    grouped = []
    for work_id, grp in candidates.groupby("work_id"):
        title = grp["title"].iloc[0]
        year = grp["year"].iloc[0]
        authors = grp[["person_id", "canonical_name"]].drop_duplicates()

        author_ids = authors["person_id"].astype(int).tolist()
        author_names = authors["canonical_name"].tolist()

        if exclude_seed_author_papers and any(aid in seed_author_set for aid in author_ids):
            continue

        distances = [author_dist.get(aid, None) for aid in author_ids]
        nearby_distances = [d for d in distances if d is not None]

        if not nearby_distances:
            continue

        score = sum(1.0 / (1.0 + d) for d in nearby_distances)

        shared_nearby_authors = len(nearby_distances)
        min_author_distance = min(nearby_distances)

        grouped.append({
            "work_id": work_id,
            "title": title,
            "year": year,
            "score": score,
            "shared_nearby_authors": shared_nearby_authors,
            "min_author_distance": min_author_distance,
            "authors": "; ".join(author_names),
        })

    result = pd.DataFrame(grouped)

    if result.empty:
        return result

    result = result[result["work_id"] != seed_work_id]

    result = result.sort_values(
        by=["score", "shared_nearby_authors", "year", "work_id"],
        ascending=[False, False, False, True],
    )

    return result.head(top_n).reset_index(drop=True)