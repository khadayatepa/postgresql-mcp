#!/usr/bin/env python3
"""
PostgreSQL MCP Server (Python) — Streamlit-compatible version with NLP-to-SQL support + Streaming + Toggle

This script can run on **Streamlit Cloud** and expose PostgreSQL helper tools with **natural language query support**.

- Supports connection string (`postgresql://...`).
- Safe SQL enforcement by default (only SELECT/WITH/SHOW/EXPLAIN allowed unless ALLOW_DANGEROUS_WRITE=true).
- Streamlit UI for database operations.
- NLP layer to convert plain English questions into SQL queries.
- **Streaming toggle** for SQL generation (user can switch between streaming and non-streaming).

Usage on Streamlit Cloud:
1. Add a `requirements.txt` with:
   ```
   streamlit
   psycopg[binary]
   openai
   # Optional: mcp
   ```
2. Deploy this file as `streamlit_app.py` (Streamlit Cloud auto-runs `streamlit run`).
3. Provide either:
   - Connection string in the sidebar, OR
   - Environment variable `DATABASE_URL`.
   - API key via environment variable `OPENAI_API_KEY` (required for NLP).

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
    if os.getenv("ALLOW_DANGEROUS_WRITE", "false").lower() == "true":
        return True
    READ_ONLY_STATEMENTS = (
        r"^\s*SELECT\\b",
        r"^\s*WITH\\b",
        r"^\s*SHOW\\b",
        r"^\s*EXPLAIN\\b",
    )
    return bool(re.search("|".join(READ_ONLY_STATEMENTS), sql, re.I | re.S))


# ---------------------------------------------------------------------------
# NLP-to-SQL conversion using OpenAI (v1.x client, streaming + non-streaming)
# ---------------------------------------------------------------------------
def nl_to_sql(nl_query: str, schema_hint: str = "public", stream_mode: bool = True) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    system_prompt = f"You are a helpful assistant that converts natural language to SQL for PostgreSQL. Default schema is {schema_hint}. Only generate SQL without explanation."

    if stream_mode:
        sql_fragments = []
        with client.chat.completions.stream(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": nl_query},
            ],
        ) as stream:
            sql_box = st.empty()
            for event in stream:
                if event.type == "token":
                    sql_fragments.append(event.token)
                    sql_box.code("".join(sql_fragments), language="sql")
            final = stream.get_final_completion()
            if final.choices:
                return final.choices[0].message.content.strip()
        return ""
    else:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": nl_query},
            ],
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PostgreSQL MCP Server", layout="wide")
st.title("📦 PostgreSQL MCP Server — NLP Powered (Streaming Toggle)")

# Sidebar connection string
st.sidebar.header("Database Connection")
def_conn_str = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_oZTFpVus5S1N@ep-broad-king-adhmh6wn-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
)

conn_str = st.sidebar.text_input("Connection String", def_conn_str)

menu = st.sidebar.radio("Choose action", ["Health Check", "Ask in Natural Language", "Run SQL", "List Tables", "Describe Table"])

if menu == "Health Check":
    st.subheader("🔍 Health Check")
    try:
        with _connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        st.success("Database connection OK ✅")
    except Exception as e:
        st.error(f"Database connection failed: {e}")

elif menu == "Ask in Natural Language":
    nl_query = st.text_area("Ask a question (e.g., 'Show me the last 10 users')")
    max_rows = st.number_input("Max Rows", value=500, min_value=1, step=100)
    stream_mode = st.checkbox("Enable Streaming", value=True)
    if st.button("Generate & Run SQL"):
        try:
            sql = nl_to_sql(nl_query, stream_mode=stream_mode)
            if not sql:
                st.error("No SQL generated.")
            else:
                st.code(sql, language="sql")
                if not _is_safe_sql(sql):
                    st.warning("⚠️ Unsafe SQL blocked. Set ALLOW_DANGEROUS_WRITE=true to allow writes.")
                else:
                    with _connect(conn_str) as conn:
                        with conn.cursor() as cur:
                            cur.execute(sql)
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
