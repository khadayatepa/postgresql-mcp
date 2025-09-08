#!/usr/bin/env python3
"""
PostgreSQL MCP Server with Fixed AI Query Execution
==================================================

Complete application with working AI-powered natural language to SQL conversion
and reliable query execution.

Author: Prashant Khadayate
Version: 2.0.1
"""

import os
import re
import json
import time
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Enhanced imports for professional features
try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False

PANDAS_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration and Data Models
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Enhanced query result with metadata"""
    sql: str
    data: List[Dict]
    columns: List[str]
    execution_time: float
    row_count: int
    timestamp: datetime
    query_id: str
    status: str = "success"
    error: Optional[str] = None

@dataclass
class TableInfo:
    """Enhanced table metadata"""
    name: str
    schema: str
    row_count: int
    size: str
    columns: List[Dict]
    indexes: List[Dict]
    foreign_keys: List[Dict]
    created_at: Optional[datetime] = None

# ---------------------------------------------------------------------------
# Enhanced PostgreSQL Connection & Operations
# ---------------------------------------------------------------------------

class DatabaseManager:
    """Professional database management class with improved error handling"""
    
    def __init__(self, conn_str: str):
        self.conn_str = conn_str
        self._test_connection()
    
    def _test_connection(self):
        """Test the database connection on initialization"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
        except Exception as e:
            st.error(f"Failed to establish database connection: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling and cleanup"""
        conn = None
        try:
            # Try psycopg3 first
            try:
                import psycopg
                from psycopg.rows import dict_row
                conn = psycopg.connect(self.conn_str, row_factory=dict_row)
                yield conn
                return
            except ImportError:
                pass
            except Exception as e:
                st.warning(f"psycopg3 connection failed: {e}, trying psycopg2...")
            
            # Fall back to psycopg2
            try:
                import psycopg2
                import psycopg2.extras
                conn = psycopg2.connect(self.conn_str)
                conn.cursor_factory = psycopg2.extras.RealDictCursor
                yield conn
                return
            except ImportError:
                raise RuntimeError("No PostgreSQL driver found. Please install psycopg[binary] or psycopg2-binary")
            except Exception as e:
                raise RuntimeError(f"Database connection failed: {e}")
                
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except:
                    pass
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def execute_query(self, sql: str, params: Optional[Tuple] = None, fetch: bool = True) -> QueryResult:
        """Execute query with comprehensive result tracking and better error handling"""
        start_time = time.time()
        query_id = hashlib.md5(f"{sql}{time.time()}".encode()).hexdigest()[:8]
        
        # Clean and validate SQL
        sql_clean = sql.strip()
        if not sql_clean:
            return QueryResult(
                sql=sql,
                data=[],
                columns=[],
                execution_time=0,
                row_count=0,
                timestamp=datetime.now(),
                query_id=query_id,
                status="error",
                error="Empty SQL query"
            )
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Execute the query
                    if params:
                        cur.execute(sql_clean, params)
                    else:
                        cur.execute(sql_clean)
                    
                    execution_time = time.time() - start_time
                    
                    # Handle different query types
                    if fetch and cur.description:
                        columns = [desc[0] for desc in cur.description]
                        rows = cur.fetchall()
                        
                        # Convert to list of dicts - handle both psycopg2 and psycopg3
                        if rows:
                            if hasattr(rows[0], 'keys'):
                                # Already dict-like (RealDict or psycopg3 dict_row)
                                data = [dict(row) for row in rows]
                            elif isinstance(rows[0], (list, tuple)):
                                # Tuple/list format
                                data = [dict(zip(columns, row)) for row in rows]
                            else:
                                # Fallback
                                data = [dict(zip(columns, row)) for row in rows]
                        else:
                            data = []
                        
                        return QueryResult(
                            sql=sql,
                            data=data,
                            columns=columns,
                            execution_time=execution_time,
                            row_count=len(data),
                            timestamp=datetime.now(),
                            query_id=query_id
                        )
                    else:
                        # Non-SELECT query or no fetch requested
                        row_count = getattr(cur, 'rowcount', 0)
                        if hasattr(conn, 'commit'):
                            conn.commit()  # Commit for INSERT/UPDATE/DELETE
                        
                        return QueryResult(
                            sql=sql,
                            data=[],
                            columns=[],
                            execution_time=execution_time,
                            row_count=row_count if row_count >= 0 else 0,
                            timestamp=datetime.now(),
                            query_id=query_id
                        )
        
        except Exception as e:
            error_msg = str(e)
            
            # Provide more helpful error messages
            if "does not exist" in error_msg.lower():
                error_msg = f"Table or column does not exist: {error_msg}"
            elif "syntax error" in error_msg.lower():
                error_msg = f"SQL syntax error: {error_msg}"
            elif "permission denied" in error_msg.lower():
                error_msg = f"Permission denied: {error_msg}"
            elif "connection" in error_msg.lower():
                error_msg = f"Database connection error: {error_msg}"
            
            return QueryResult(
                sql=sql,
                data=[],
                columns=[],
                execution_time=time.time() - start_time,
                row_count=0,
                timestamp=datetime.now(),
                query_id=query_id,
                status="error",
                error=error_msg
            )
    
    def get_enhanced_schema_info(self, schema: str = "public") -> List[TableInfo]:
        """Get comprehensive schema information"""
        sql = """
        SELECT 
            t.table_name,
            t.table_schema,
            COALESCE(s.n_tup_ins + s.n_tup_upd + s.n_tup_del, 0) as row_count,
            pg_size_pretty(pg_total_relation_size(c.oid)) as size
        FROM information_schema.tables t
        LEFT JOIN pg_class c ON c.relname = t.table_name
        LEFT JOIN pg_stat_user_tables s ON s.relname = t.table_name
        WHERE t.table_schema = %s AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_name
        """
        
        result = self.execute_query(sql, (schema,))
        tables = []
        
        for row in result.data:
            # Get column information
            col_sql = """
            SELECT column_name, data_type, is_nullable, column_default,
                   character_maximum_length, numeric_precision, numeric_scale
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """
            col_result = self.execute_query(col_sql, (schema, row['table_name']))
            
            # Get index information
            idx_sql = """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = %s AND tablename = %s
            """
            idx_result = self.execute_query(idx_sql, (schema, row['table_name']))
            
            # Get foreign key information
            fk_sql = """
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = %s AND tc.table_name = %s
            """
            fk_result = self.execute_query(fk_sql, (schema, row['table_name']))
            
            tables.append(TableInfo(
                name=row['table_name'],
                schema=schema,
                row_count=row['row_count'] or 0,
                size=row['size'] or '0 bytes',
                columns=col_result.data,
                indexes=idx_result.data,
                foreign_keys=fk_result.data
            ))
        
        return tables

# ---------------------------------------------------------------------------
# AI Query Generation
# ---------------------------------------------------------------------------

class AIQueryGenerator:
    """AI-powered SQL generation"""
    
    def __init__(self):
        self.providers = {
            "OpenAI GPT-4": self._openai_generate,
        }
    
    def generate_sql(self, nl_query: str, schema_context: str, provider: str = "OpenAI GPT-4", 
                    stream: bool = True) -> str:
        """Generate SQL from natural language"""
        if provider in self.providers:
            return self.providers[provider](nl_query, schema_context, stream)
        else:
            raise ValueError(f"Provider {provider} not supported")
    
    def _openai_generate(self, nl_query: str, schema_context: str, stream: bool) -> str:
        """OpenAI implementation"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            system_prompt = f"""You are an expert PostgreSQL developer. Generate efficient, secure SQL queries.

Database Schema:
{schema_context}

Rules:
1. Generate PostgreSQL-compatible SQL only
2. Use proper indexing strategies when possible
3. Handle edge cases (NULL values, empty results)
4. Use appropriate JOINs and subqueries
5. Return only the SQL query, no explanations
6. Always include appropriate LIMIT clauses for safety

Current timestamp: {datetime.now()}
"""
            
            if stream:
                sql_fragments = []
                response_container = st.empty()
                
                stream_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": nl_query}
                    ],
                    stream=True,
                    temperature=0.1
                )
                
                for chunk in stream_response:
                    if chunk.choices[0].delta.content:
                        sql_fragments.append(chunk.choices[0].delta.content)
                        response_container.code(
                            self._clean_sql("".join(sql_fragments)), 
                            language="sql"
                        )
                
                return self._clean_sql("".join(sql_fragments))
            else:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": nl_query}
                    ],
                    temperature=0.1
                )
                return self._clean_sql(response.choices[0].message.content)
        
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
            return ""
    
    def _clean_sql(self, sql: str) -> str:
        """Enhanced SQL cleaning and formatting"""
        if not sql:
            return ""
        
        # Remove code fences
        sql = re.sub(r'```sql\n?', '', sql)
        sql = re.sub(r'```\n?', '', sql)
        sql = sql.strip()
        
        # Format SQL if sqlparse is available
        if SQLPARSE_AVAILABLE and sql:
            try:
                sql = sqlparse.format(sql, reindent=True, keyword_case='upper')
            except:
                pass
        
        return sql

# ---------------------------------------------------------------------------
# Query Management
# ---------------------------------------------------------------------------

class QueryManager:
    """Query history and favorites management"""
    
    def __init__(self):
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'favorite_queries' not in st.session_state:
            st.session_state.favorite_queries = []
    
    def add_to_history(self, result: QueryResult):
        """Add query result to history"""
        st.session_state.query_history.insert(0, asdict(result))
        # Keep only last 50 queries
        st.session_state.query_history = st.session_state.query_history[:50]
    
    def add_to_favorites(self, sql: str, name: str):
        """Add query to favorites"""
        favorite = {
            "name": name,
            "sql": sql,
            "created_at": datetime.now().isoformat(),
            "id": hashlib.md5(f"{name}{sql}".encode()).hexdigest()[:8]
        }
        st.session_state.favorite_queries.append(favorite)
    
    def get_history(self) -> List[Dict]:
        """Get query history"""
        return st.session_state.query_history
    
    def get_favorites(self) -> List[Dict]:
        """Get favorite queries"""
        return st.session_state.favorite_queries

# ---------------------------------------------------------------------------
# UI Functions
# ---------------------------------------------------------------------------

def setup_professional_theme():
    """Setup professional styling"""
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_connection_sidebar() -> Tuple[str, Dict]:
    """Render connection sidebar"""
    st.sidebar.markdown("## 🔗 Database Connection")
    
    connection_method = st.sidebar.radio(
        "Connection Method",
        ["Connection String", "Individual Parameters"],
        help="Choose how to connect to your PostgreSQL database"
    )
    
    if connection_method == "Connection String":
        default_conn = os.getenv("DATABASE_URL", "")
        conn_str = st.sidebar.text_input(
            "Connection String",
            value=default_conn,
            type="password",
            help="postgresql://user:password@host:port/database"
        )
        
        # Validate connection string format
        if conn_str and not conn_str.startswith(("postgresql://", "postgres://")):
            st.sidebar.error("Connection string must start with postgresql:// or postgres://")
            conn_str = ""
            
    else:
        host = st.sidebar.text_input("Host", value="localhost")
        port = st.sidebar.number_input("Port", value=5432, min_value=1, max_value=65535)
        database = st.sidebar.text_input("Database", value="postgres")
        username = st.sidebar.text_input("Username", value="postgres")
        password = st.sidebar.text_input("Password", type="password")
        
        if all([host, port, database, username, password]):
            conn_str = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        else:
            conn_str = ""
    
    # Connection test
    if conn_str and st.sidebar.button("🔍 Test Connection"):
        # Create placeholder for status updates
        status_placeholder = st.sidebar.empty()
        status_placeholder.info("Testing connection...")
        
        try:
            db_manager = DatabaseManager(conn_str)
            result = db_manager.execute_query("SELECT version() as version")
            if result.status == "success":
                status_placeholder.success("✅ Connection successful!")
                if result.data:
                    version = result.data[0].get('version', 'Unknown')
                    st.sidebar.info(f"PostgreSQL: {version[:50]}...")
            else:
                status_placeholder.error(f"❌ Connection failed: {result.error}")
        except Exception as e:
            status_placeholder.error(f"❌ Connection failed: {e}")
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        max_rows = st.number_input("Max Rows per Query", value=1000, min_value=1, max_value=10000)
        ai_provider = st.selectbox("AI Provider", ["OpenAI GPT-4"])
        enable_streaming = st.checkbox("Enable AI Streaming", value=True)
    
    return conn_str, {
        "max_rows": max_rows,
        "ai_provider": ai_provider,
        "enable_streaming": enable_streaming
    }


def render_ai_query_interface(db_manager: DatabaseManager, ai_generator: AIQueryGenerator, 
                             query_manager: QueryManager, settings: Dict):
    """Render AI-powered query interface with fixed execution"""
    st.markdown("### 🤖 Natural Language to SQL")
    st.markdown("Ask questions about your data in plain English, and I'll generate optimized SQL queries for you.")
    
    # Schema context
    col1, col2 = st.columns([3, 1])
    with col1:
        nl_query = st.text_area(
            "What would you like to know?",
            placeholder="e.g., Show me all tables in the database",
            height=100,
            key="nl_query_input"
        )
    
    with col2:
        schema = st.selectbox("Schema", ["public"], help="Select the database schema to query")
        max_rows = st.number_input("Max Results", value=settings['max_rows'], min_value=1, max_value=10000)
    
    # Initialize session state for generated SQL
    if 'generated_sql' not in st.session_state:
        st.session_state.generated_sql = ""
    if 'show_execute_button' not in st.session_state:
        st.session_state.show_execute_button = False
    
    # Generate SQL button
    if st.button("🚀 Generate SQL Query", type="primary"):
        if not nl_query.strip():
            st.warning("Please enter your question first.")
            st.session_state.show_execute_button = False
            return
        
        try:
            # Get schema context
            with st.spinner("Loading database schema..."):
                tables = db_manager.get_enhanced_schema_info(schema)
                schema_context = "\n".join([
                    f"Table: {t.name} ({t.row_count} rows, {t.size})\nColumns: " + 
                    ", ".join([f"{c['column_name']} ({c['data_type']})" for c in t.columns[:5]])
                    for t in tables[:10]  # Limit context size
                ])
            
            # Generate SQL
            with st.spinner("🧠 AI is generating your SQL query..."):
                sql = ai_generator.generate_sql(
                    nl_query, schema_context, 
                    settings.get('ai_provider', 'OpenAI GPT-4'), 
                    settings.get('enable_streaming', True)
                )
            
            if sql:
                st.session_state.generated_sql = sql
                st.session_state.show_execute_button = True
                st.success("✅ SQL query generated successfully!")
            else:
                st.error("❌ Failed to generate SQL query. Please try rephrasing your question.")
                st.session_state.show_execute_button = False
            
        except Exception as e:
            st.error(f"Error generating query: {e}")
            st.session_state.show_execute_button = False
    
    # Display generated SQL and execute button
    if st.session_state.generated_sql:
        st.markdown("**Generated SQL:**")
        
        # Allow editing of generated SQL
        edited_sql = st.text_area(
            "You can modify the generated SQL if needed:",
            value=st.session_state.generated_sql,
            height=150,
            key="generated_sql_editor"
        )
        
        # Update session state if user modified the SQL
        if edited_sql != st.session_state.generated_sql:
            st.session_state.generated_sql = edited_sql
        
        st.code(st.session_state.generated_sql, language="sql")
        
        # Execute button with proper logic
        col_exec, col_explain, col_save = st.columns([2, 2, 2])
        
        with col_exec:
            if st.button("▶️ Execute Query", key="ai_execute_btn"):
                execute_ai_generated_query(db_manager, query_manager, st.session_state.generated_sql, max_rows)
        
        with col_explain:
            if st.button("📋 Explain Query", key="ai_explain_btn"):
                explain_ai_query(st.session_state.generated_sql)
        
        with col_save:
            if st.button("⭐ Save to Favorites", key="ai_save_btn"):
                save_ai_query_to_favorites(query_manager, st.session_state.generated_sql, nl_query)


def execute_ai_generated_query(db_manager: DatabaseManager, query_manager: QueryManager, sql: str, max_rows: int):
    """Execute AI-generated SQL query with comprehensive handling"""
    
    if not sql or not sql.strip():
        st.error("No SQL query to execute")
        return
    
    # Show execution progress
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0, text="Preparing to execute query...")
        
        try:
            # Validate SQL
            progress_bar.progress(20, text="Validating SQL query...")
            sql_clean = sql.strip()
            
            # Add LIMIT if it's a SELECT query and max_rows is specified
            if (max_rows > 0 and 
                re.search(r'^\s*SELECT\b', sql_clean, re.IGNORECASE) and 
                not re.search(r'\bLIMIT\b', sql_clean, re.IGNORECASE)):
                if not sql_clean.endswith(';'):
                    sql_clean += f' LIMIT {max_rows};'
                else:
                    sql_clean = sql_clean.rstrip(';') + f' LIMIT {max_rows};'
            
            # Execute query
            progress_bar.progress(50, text="Executing query...")
            result = db_manager.execute_query(sql_clean)
            
            progress_bar.progress(100, text="Query completed!")
            
            # Clear progress bar after a brief delay
            time.sleep(0.5)
            progress_bar.empty()
            
            # Handle results
            if result.status == "success":
                # Add to history
                query_manager.add_to_history(result)
                
                # Success message
                st.markdown(f"""
                <div style="background-color: #d4edda; color: #155724; padding: 1rem; 
                           border-radius: 0.5rem; border-left: 4px solid #28a745; margin: 1rem 0;">
                    <strong>✅ Query executed successfully!</strong><br>
                    📊 Returned {result.row_count:,} rows in {result.execution_time:.2f}s
                </div>
                """, unsafe_allow_html=True)
                
                # Display results
                if result.data:
                    df = pd.DataFrame(result.data)
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Execution Time", f"{result.execution_time:.3f}s")
                    
                    # Data table
                    st.markdown("**📊 Query Results:**")
                    if len(df) > 100:
                        st.info(f"Showing first 100 rows of {len(df):,} total rows")
                        st.dataframe(df.head(100), use_container_width=True)
                    else:
                        st.dataframe(df, use_container_width=True)
                    
                    # Export and visualization options
                    col_download, col_viz, col_copy = st.columns(3)
                    
                    with col_download:
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            f"ai_query_result_{result.query_id}.csv",
                            "text/csv"
                        )
                    
                    with col_viz:
                        if st.button("📈 Auto-Visualize", key="ai_viz_btn"):
                            create_auto_visualization(df)
                    
                    with col_copy:
                        if st.button("📋 Copy to SQL Editor", key="ai_copy_btn"):
                            st.session_state['editor_sql'] = sql_clean
                            st.success("Query copied to SQL Editor!")
                
                else:
                    if result.row_count > 0:
                        st.success(f"Query executed successfully! {result.row_count} rows affected.")
                    else:
                        st.info("Query executed successfully (no rows returned)")
            
            else:
                # Error handling
                st.markdown(f"""
                <div style="background-color: #f8d7da; color: #721c24; padding: 1rem; 
                           border-radius: 0.5rem; border-left: 4px solid #dc3545; margin: 1rem 0;">
                    <strong>❌ Query execution failed</strong><br>
                    <code>{result.error}</code>
                </div>
                """, unsafe_allow_html=True)
                
                # Provide helpful suggestions
                provide_error_suggestions(result.error)
        
        except Exception as e:
            progress_bar.empty()
            st.error(f"Unexpected error during query execution: {e}")
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.code(f"Generated SQL:\n{sql}", language="sql")
                st.text(f"Error: {str(e)}")


def create_auto_visualization(df: pd.DataFrame):
    """Create automatic visualization for query results"""
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        
        if not numeric_cols and not categorical_cols:
            st.info("No suitable columns found for visualization")
            return
        
        # Choose visualization type based on data
        if len(numeric_cols) >= 2:
            # Scatter plot for two numeric columns
            fig = px.scatter(
                df.head(1000), 
                x=numeric_cols[0], 
                y=numeric_cols[1],
                title=f"{numeric_cols[1]} vs {numeric_cols[0]}"
            )
        elif len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            # Bar chart for categorical vs numeric
            if len(df) > 50:
                sample_df = df.sample(n=min(50, len(df)))
            else:
                sample_df = df
            
            fig = px.bar(
                sample_df, 
                x=categorical_cols[0], 
                y=numeric_cols[0],
                title=f"{numeric_cols[0]} by {categorical_cols[0]}"
            )
            fig.update_xaxis(tickangle=45)
        elif len(categorical_cols) >= 1:
            # Pie chart for categorical data
            value_counts = df[categorical_cols[0]].value_counts().head(10)
            fig = px.pie(
                values=value_counts.values, 
                names=value_counts.index,
                title=f"Distribution of {categorical_cols[0]}"
            )
        elif len(numeric_cols) >= 1:
            # Histogram for single numeric column
            fig = px.histogram(
                df, 
                x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}"
            )
        else:
            st.info("Unable to create visualization with available data types")
            return
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Visualization error: {e}")


def explain_ai_query(sql: str):
    """Explain the AI-generated SQL query in plain English"""
    st.markdown("**📋 Query Explanation:**")
    
    # Basic SQL parsing to explain query components
    sql_upper = sql.upper().strip()
    
    explanation_parts = []
    
    if sql_upper.startswith('SELECT'):
        explanation_parts.append("📊 This is a **SELECT** query that retrieves data from the database")
        
        # Check for JOIN
        if 'JOIN' in sql_upper:
            explanation_parts.append("🔗 The query **joins multiple tables** to combine related data")
        
        # Check for WHERE
        if 'WHERE' in sql_upper:
            explanation_parts.append("🔍 It includes **filtering conditions** to narrow down the results")
        
        # Check for GROUP BY
        if 'GROUP BY' in sql_upper:
            explanation_parts.append("📈 The results are **grouped** to perform aggregation")
        
        # Check for ORDER BY
        if 'ORDER BY' in sql_upper:
            explanation_parts.append("📋 The results are **sorted** in a specific order")
        
        # Check for LIMIT
        if 'LIMIT' in sql_upper:
            explanation_parts.append("🎯 The number of results is **limited** to prevent overwhelming output")
    
    elif sql_upper.startswith('INSERT'):
        explanation_parts.append("➕ This is an **INSERT** query that adds new data to the database")
    
    elif sql_upper.startswith('UPDATE'):
        explanation_parts.append("✏️ This is an **UPDATE** query that modifies existing data")
    
    elif sql_upper.startswith('DELETE'):
        explanation_parts.append("🗑️ This is a **DELETE** query that removes data from the database")
    
    for part in explanation_parts:
        st.markdown(f"- {part}")
    
    # Show the formatted SQL
    st.markdown("**SQL Code:**")
    st.code(sql, language="sql")


def provide_error_suggestions(error_message: str):
    """Provide helpful suggestions based on error message"""
    error_lower = error_message.lower() if error_message else ""
    
    suggestions = []
    
    if "does not exist" in error_lower:
        suggestions.extend([
            "🔍 Check if the table or column names are spelled correctly",
            "📋 Use the Schema Explorer tab to see available tables and columns",
            "🔧 Make sure you're connected to the correct database"
        ])
    
    elif "syntax error" in error_lower:
        suggestions.extend([
            "📝 Check your SQL syntax - there might be a missing comma, quote, or bracket",
            "🎨 Try using the Format SQL button to identify syntax issues",
            "📚 Verify that column names with spaces are properly quoted"
        ])
    
    elif "permission denied" in error_lower:
        suggestions.extend([
            "🔐 Your database user might not have sufficient permissions",
            "👤 Contact your database administrator for access",
            "🔑 Check if you're using the correct database credentials"
        ])
    
    elif "connection" in error_lower:
        suggestions.extend([
            "🌐 Check your database connection settings",
            "🔄 Try reconnecting to the database",
            "⚡ Verify that the database server is running"
        ])
    
    else:
        suggestions.extend([
            "🤖 Try rephrasing your natural language query",
            "📝 Review the generated SQL for any obvious issues",
            "🔍 Check the database schema to ensure the query matches your data structure"
        ])
    
    if suggestions:
        st.markdown("**💡 Suggestions:**")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")


def save_ai_query_to_favorites(query_manager: QueryManager, sql: str, nl_query: str):
    """Save AI-generated query to favorites"""
    with st.form("save_ai_query_form"):
        st.markdown("**Save Query to Favorites**")
        
        # Suggest a name based on the natural language query
        suggested_name = nl_query[:50] + "..." if len(nl_query) > 50 else nl_query
        
        query_name = st.text_input(
            "Query Name",
            value=suggested_name,
            help="Enter a descriptive name for this query"
        )
        
        query_description = st.text_area(
            "Description (optional)",
            value=f"AI-generated query from: {nl_query}",
            help="Add any additional notes about this query"
        )
        
        if st.form_submit_button("💾 Save Query"):
            if query_name.strip():
                query_manager.add_to_favorites(sql, query_name.strip())
                st.success(f"✅ Query '{query_name}' saved to favorites!")
            else:
                st.warning("Please enter a query name")


def render_schema_explorer(db_manager: DatabaseManager):
    """Render enhanced schema explorer"""
    st.markdown("### 📋 Database Schema Explorer")
    
    schema = st.selectbox("Select Schema", ["public", "information_schema"], key="schema_explorer")
    
    try:
        tables = db_manager.get_enhanced_schema_info(schema)
        
        if not tables:
            st.info("No tables found in this schema.")
            return
        
        # Tables overview
        st.markdown(f"**Found {len(tables)} tables:**")
        
        # Create summary table
        summary_data = []
        for table in tables:
            summary_data.append({
                "Table": table.name,
                "Rows": f"{table.row_count:,}",
                "Size": table.size,
                "Columns": len(table.columns),
                "Indexes": len(table.indexes),
                "Foreign Keys": len(table.foreign_keys)
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Detailed table explorer
        st.markdown("---")
        selected_table = st.selectbox("Explore Table Details", [t.name for t in tables])
        
        if selected_table:
            table_info = next(t for t in tables if t.name == selected_table)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Columns**")
                cols_df = pd.DataFrame(table_info.columns)
                if not cols_df.empty:
                    st.dataframe(cols_df, use_container_width=True)
                
                st.markdown("**🔑 Indexes**")
                if table_info.indexes:
                    idx_df = pd.DataFrame(table_info.indexes)
                    st.dataframe(idx_df, use_container_width=True)
                else:
                    st.info("No indexes found")
            
            with col2:
                st.markdown("**🔗 Foreign Keys**")
                if table_info.foreign_keys:
                    fk_df = pd.DataFrame(table_info.foreign_keys)
                    st.dataframe(fk_df, use_container_width=True)
                else:
                    st.info("No foreign keys found")
                
                # Sample data
                st.markdown("**🔍 Sample Data**")
                sample_result = db_manager.execute_query(f"SELECT * FROM {schema}.{selected_table} LIMIT 5")
                if sample_result.data:
                    sample_df = pd.DataFrame(sample_result.data)
                    st.dataframe(sample_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error exploring schema: {e}")


def render_sql_editor(db_manager: DatabaseManager, query_manager: QueryManager, settings: Dict):
    """Render advanced SQL editor"""
    st.markdown("### ⚡ Advanced SQL Editor")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get SQL from session state or use default
        default_query = st.session_state.get('editor_sql', 
                                           "SELECT * FROM information_schema.tables WHERE table_schema = 'public' LIMIT 10;")
        
        sql_query = st.text_area(
            "SQL Query",
            value=default_query,
            height=200,
            help="Write your SQL query here",
            key="sql_editor_main"
        )
        
        # Query controls
        col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 2])
        with col_a:
            execute_btn = st.button("▶️ Execute", type="primary")
        with col_b:
            if SQLPARSE_AVAILABLE and st.button("🎨 Format SQL"):
                try:
                    formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
                    st.session_state['editor_sql'] = formatted_sql
                    st.rerun()
                except Exception as e:
                    st.error(f"Error formatting SQL: {e}")
        with col_c:
            if st.button("🗑️ Clear"):
                st.session_state['editor_sql'] = ""
                st.rerun()
        with col_d:
            if st.button("📋 Sample"):
                st.session_state['editor_sql'] = "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"
                st.rerun()
    
    with col2:
        st.markdown("**Query Options**")
        max_rows = st.number_input("Max Rows", value=settings['max_rows'], min_value=1, key="sql_max_rows")
    
    # Execute query
    if execute_btn and sql_query.strip():
        execute_regular_sql_query(db_manager, query_manager, sql_query, max_rows)


def execute_regular_sql_query(db_manager: DatabaseManager, query_manager: QueryManager, sql: str, max_rows: int):
    """Execute regular SQL query (non-AI generated)"""
    
    if not sql.strip():
        st.warning("Please enter a SQL query")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Validating query...")
        progress_bar.progress(20)
        
        sql_clean = sql.strip()
        if not sql_clean.endswith(';'):
            sql_clean += ';'
        
        status_text.text("Executing query...")
        progress_bar.progress(50)
        
        # Add LIMIT if it's a SELECT query without LIMIT
        if (max_rows > 0 and 
            re.search(r'^\s*SELECT\b', sql_clean, re.IGNORECASE) and 
            not re.search(r'\bLIMIT\b', sql_clean, re.IGNORECASE)):
            sql_clean = sql_clean.rstrip(';') + f' LIMIT {max_rows};'
        
        result = db_manager.execute_query(sql_clean)
        
        progress_bar.progress(100)
        status_text.text("Query completed!")
        
        # Clear progress indicators
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        if result.status == "success":
            query_manager.add_to_history(result)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows Returned", f"{result.row_count:,}")
            with col2:
                st.metric("Execution Time", f"{result.execution_time:.3f}s")
            with col3:
                st.metric("Query ID", result.query_id)
            
            if result.data:
                df = pd.DataFrame(result.data)
                st.success(f"Query executed successfully! Returned {result.row_count} rows.")
                
                if len(df) > 100:
                    st.info(f"Showing first 100 rows of {len(df)} total rows")
                    st.dataframe(df.head(100), use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
                
                # Export options
                col_export1, col_export2, col_export3 = st.columns(3)
                with col_export1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV", 
                        csv, 
                        f"query_result_{result.query_id}.csv", 
                        "text/csv"
                    )
                with col_export2:
                    json_str = df.to_json(orient='records', indent=2)
                    st.download_button(
                        "📥 Download JSON", 
                        json_str, 
                        f"query_result_{result.query_id}.json", 
                        "application/json"
                    )
                with col_export3:
                    if st.button("📈 Create Visualization"):
                        create_auto_visualization(df)
            else:
                if result.row_count > 0:
                    st.success(f"Query executed successfully! {result.row_count} rows affected.")
                else:
                    st.success("Query executed successfully! (No rows returned)")
        
        else:
            st.error(f"Query failed: {result.error}")
            provide_error_suggestions(result.error)
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Unexpected error: {e}")


def render_favorites_manager(db_manager: DatabaseManager, query_manager: QueryManager, settings: Dict):
    """Render favorites and query management interface"""
    st.markdown("### ⭐ Query Favorites & History")
    
    tab1, tab2 = st.tabs(["⭐ Favorites", "📚 History"])
    
    with tab1:
        favorites = query_manager.get_favorites()
        
        if not favorites:
            st.info("No favorite queries yet. Save queries from the AI Query or SQL Editor tabs.")
        else:
            st.success(f"You have {len(favorites)} saved queries")
            
            for i, fav in enumerate(favorites):
                with st.expander(f"⭐ {fav['name']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.code(fav['sql'], language="sql")
                    
                    with col2:
                        if st.button("▶️ Execute", key=f"exec_fav_{i}"):
                            execute_regular_sql_query(db_manager, query_manager, fav['sql'], settings['max_rows'])
                        
                        if st.button("📋 Copy to Editor", key=f"copy_fav_{i}"):
                            st.session_state['editor_sql'] = fav['sql']
                            st.success("Copied to SQL editor!")
                        
                        if st.button("🗑️ Delete", key=f"del_fav_{i}"):
                            st.session_state.favorite_queries.pop(i)
                            st.success("Query deleted!")
                            st.rerun()
    
    with tab2:
        history = query_manager.get_history()
        
        if not history:
            st.info("No query history available")
        else:
            st.info(f"Showing last {min(10, len(history))} queries")
            
            for i, query_data in enumerate(history[:10]):
                with st.expander(f"Query {i+1}: {query_data['query_id']} ({query_data['status']})"):
                    col_x, col_y, col_z = st.columns([2, 1, 1])
                    with col_x:
                        st.code(query_data['sql'][:200] + "..." if len(query_data['sql']) > 200 else query_data['sql'])
                    with col_y:
                        st.metric("Rows", query_data['row_count'])
                    with col_z:
                        st.metric("Time", f"{query_data['execution_time']:.2f}s")
                    
                    if st.button(f"Reuse Query {i+1}", key=f"reuse_{i}"):
                        st.session_state['editor_sql'] = query_data['sql']
                        st.success("Query copied to SQL editor!")


def render_main_dashboard(db_manager: DatabaseManager, settings: Dict):
    """Render the main application dashboard"""
    
    # Header with metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get basic database info
        db_info = db_manager.execute_query("SELECT version() as version")
        db_size = db_manager.execute_query("SELECT pg_size_pretty(pg_database_size(current_database())) as size")
        table_count = db_manager.execute_query("SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = 'public'")
        
        with col1:
            st.metric("Database Status", "🟢 Connected", "Healthy")
        with col2:
            st.metric("Database Size", db_size.data[0]['size'] if db_size.data else "Unknown")
        with col3:
            st.metric("Tables", table_count.data[0]['count'] if table_count.data else "0")
        with col4:
            st.metric("AI Provider", settings['ai_provider'])
    
    except Exception as e:
        st.error(f"Failed to load database metrics: {e}")
    
    # Main navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 AI Query", "📋 Schema Explorer", "⚡ SQL Editor", "⭐ Favorites & History"
    ])
    
    # Initialize components
    ai_generator = AIQueryGenerator()
    query_manager = QueryManager()
    
    with tab1:
        render_ai_query_interface(db_manager, ai_generator, query_manager, settings)
    
    with tab2:
        render_schema_explorer(db_manager)
    
    with tab3:
        render_sql_editor(db_manager, query_manager, settings)
    
    with tab4:
        render_favorites_manager(db_manager, query_manager, settings)


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="PostgreSQL MCP Server Pro | By Prashant Khadayate",
        page_icon="🐘",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Professional theme setup
    setup_professional_theme()
    
    # Header
    st.markdown("""
    # 🐘 PostgreSQL MCP Server Pro
    ### *Enterprise-Grade Database Management & AI-Powered Analytics*
    
    ---
    """)
    
    # Connection sidebar
    conn_str, settings = render_connection_sidebar()
    
    if not conn_str:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Welcome to PostgreSQL MCP Server Pro!</h3>
            <p><strong>Get started in 3 simple steps:</strong></p>
            <ol>
                <li>🔗 Enter your PostgreSQL connection details in the sidebar</li>
                <li>🔍 Test your connection to ensure everything works</li>
                <li>🤖 Start exploring your database with AI assistance!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 AI-Powered Queries
            - Natural language to SQL conversion
            - Smart query optimization
            - Real-time streaming responses
            - Query explanation and debugging
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Advanced Analytics
            - Interactive data visualizations
            - Automatic chart generation
            - Export capabilities (CSV, JSON)
            - Schema exploration tools
            """)
        
        with col3:
            st.markdown("""
            ### 🔐 Enterprise Features
            - Secure connection management
            - Query history & favorites
            - Professional UI/UX
            - Progress tracking & feedback
            """)
        
        return
    
    # Initialize database manager
    try:
        db_manager = DatabaseManager(conn_str)
        
        # Connection success message
        st.markdown("""
        <div class="success-box" style="margin-bottom: 2rem;">
            <strong>🎉 Successfully connected to your PostgreSQL database!</strong><br>
            Ready to explore your data with AI-powered assistance.
        </div>
        """, unsafe_allow_html=True)
        
        # Render main dashboard
        render_main_dashboard(db_manager, settings)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>🚀 <strong>PostgreSQL MCP Server Pro v2.0.1</strong> | 
            Built with ❤️ using Streamlit | 
            Created by <strong>Prashant Khadayate</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        
        # Troubleshooting guide
        with st.expander("🔧 Troubleshooting Guide", expanded=True):
            st.markdown("""
            **Common Connection Issues & Solutions:**
            
            1. **🔌 Connection String Format**
               ```
               postgresql://username:password@host:port/database
               ```
               Example: `postgresql://postgres:mypass@localhost:5432/mydb`
            
            2. **🌐 Network Issues**
               - Verify the database server is running
               - Check if the host and port are correct
               - Ensure firewall allows connections
            
            3. **🔐 Authentication Problems**
               - Verify username and password
               - Check if user has necessary permissions
               - Confirm database name exists
            
            4. **📦 Missing Dependencies**
               ```bash
               pip install psycopg[binary] pandas plotly streamlit openai sqlparse
               ```
            
            5. **🤖 OpenAI API Key**
               ```bash
               export OPENAI_API_KEY="your-api-key-here"
               ```
            """)


if __name__ == "__main__":
    main()
