#!/usr/bin/env python3
"""
PostgreSQL MCP Server (Python) — Streamlit-compatible version with connection string support

This script can run on **Streamlit Cloud** and expose PostgreSQL helper tools.

- Supports connection string (`postgresql://...`).
- Safe SQL enforcement by default (only SELECT/WITH/SHOW/EXPLAIN allowed unless ALLOW_DANGEROUS_WRITE=true).
- Streamlit UI for database operations.

Usage on Streamlit Cloud:
1. Add a `requirements.txt` with:
   ```
   streamlit
   psycopg[binary]
   # Optional: mcp
   ```
2. Deploy this file as `streamlit_app.py` (Streamlit Cloud auto-runs `streamlit run`).
3. Provide either:
   - Connection string in the sidebar, OR
   - Environment variable `DATABASE_URL`.

"""

import os
import re
import streamlit as st
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# PostgreSQL driver handling
# ---------------------------------------------------------------------------
_psycopg_mod = None
_dict_row = None
try:
    import psycopg as _psycopg  # psycopg3
    try:
        from psycopg.rows import dict_row as _dict_row  # type: ignore
    except Exception:
        _dict_row = None
    _psycopg_mod = _psycopg
except Exception:
    try:
        import psycopg2 as _psycopg  # type: ignore
        _dict_row = None
        _psycopg_mod = _psycopg
    except Exception:
        _psycopg_mod = None
        _dict_row = None


def _connect(conn_str: Optional[str] = None, custom: Optional[Dict[str, str]] = None):
    if _psycopg_mod is None:
        raise RuntimeError(
            "No PostgreSQL driver found. Install with:\n"
            "  pip install psycopg[binary]   (recommended)\n"
            "  pip install psycopg2-binary\n"
        )

    if conn_str:
        if _dict_row is not None and hasattr(_psycopg_mod, "connect"):
            return _psycopg_mod.connect(conn_str, row_factory=_dict_row)
        return _psycopg_mod.connect(conn_str)

    conn_kwargs = {
        "host": os.getenv("PGHOST"),
        "port": os.getenv("PGPORT"),
        "dbname": os.getenv("PGDATABASE"),
        "user": os.getenv("PGUSER"),
        "password": os.getenv("PGPASSWORD"),
    }
    if custom:
        conn_kwargs.update(custom)
    conn_kwargs = {k: v for k, v in conn_kwargs.items() if v}

    if hasattr(_psycopg_mod, "connect"):
        if _dict_row is not None:
            return _psycopg_mod.connect(**conn_kwargs, row_factory=_dict_row)
        return _psycopg_mod.connect(**conn_kwargs)
    return _psycopg_mod.connect(**conn_kwargs)  # type: ignore


def _is_safe_sql(sql: str) -> bool:
    if os.getenv("ALLOW_DANGEROUS_WRITE", "false").lower() == "false":
        return True
    READ_ONLY_STATEMENTS = (
        r"^\s*SELECT\\b",
        r"^\s*WITH\\b",
        r"^\s*SHOW\\b",
        r"^\s*EXPLAIN\\b",
    )
    return bool(re.search("|".join(READ_ONLY_STATEMENTS), sql, re.I | re.S))


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PostgreSQL MCP Server", layout="wide")
st.title("📦 PostgreSQL MCP Server — Streamlit UI")

# Sidebar connection string
st.sidebar.header("Database Connection")
def_conn_str = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_oZTFpVus5S1N@ep-broad-king-adhmh6wn-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
)

conn_str = st.sidebar.text_input("Connection String", def_conn_str)

menu = st.sidebar.radio("Choose action", ["Health Check", "List Tables", "Describe Table", "Run SQL"])

if menu == "Health Check":
    st.subheader("🔍 Health Check")
    try:
        with _connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        st.success("Database connection OK ✅")
    except Exception as e:
        st.error(f"Database connection failed: {e}")

elif menu == "List Tables":
    schema = st.text_input("Schema", "public")
    if st.button("List Tables"):
        sql = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = %s
        ORDER BY table_name
        """
        try:
            with _connect(conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (schema,))
                    cols = [d[0] for d in cur.description] if cur.description else []
                    rows = cur.fetchall()
                    st.dataframe([dict(zip(cols, r)) for r in rows])
        except Exception as e:
            st.error(f"Error: {e}")

elif menu == "Describe Table":
    schema = st.text_input("Schema", "public")
    table = st.text_input("Table")
    if st.button("Describe"):
        sql = """
        SELECT column_name, ordinal_position, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """
        try:
            with _connect(conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (schema, table))
                    cols = [d[0] for d in cur.description] if cur.description else []
                    rows = cur.fetchall()
                    st.dataframe([dict(zip(cols, r)) for r in rows])
        except Exception as e:
            st.error(f"Error: {e}")

elif menu == "Run SQL":
    query = st.text_area("SQL Query", "SELECT 1")
    max_rows = st.number_input("Max Rows", value=500, min_value=1, step=100)
    if st.button("Execute"):
        if not _is_safe_sql(query):
            st.warning("⚠️ Unsafe SQL blocked. Set ALLOW_DANGEROUS_WRITE=true to allow writes.")
        else:
            try:
                with _connect(conn_str) as conn:
                    with conn.cursor() as cur:
                        cur.execute(query)
                        if cur.description:
                            cols = [d[0] for d in cur.description]
                            rows = cur.fetchmany(size=max_rows + 1)
                            if len(rows) > max_rows:
                                st.info(f"Results truncated at {max_rows} rows")
                                rows = rows[:max_rows]
                            st.dataframe([dict(zip(cols, r)) for r in rows])
                        else:
                            st.success(f"Query executed. Rowcount: {cur.rowcount}")
            except Exception as e:
                st.error(f"Error: {e}")
