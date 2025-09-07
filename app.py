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

# ---------------------------------------------------------------------------
# UI Functions
# ---------------------------------------------------------------------------

def render_connection_sidebar() -> Tuple[str, Dict]:
    """Render connection sidebar with better validation"""
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
            if st.sidebar.button("🔍 Test Connection", disabled=True):
                st.sidebar.warning("Please fill in all connection details")
    
    # Connection test with better feedback
    if conn_str and st.sidebar.button("🔍 Test Connection"):
        with st.sidebar.spinner("Testing connection..."):
            try:
                db_manager = DatabaseManager(conn_str)
                result = db_manager.execute_query("SELECT version() as version")
                if result.status == "success":
                    st.sidebar.success("✅ Connection successful!")
                    if result.data:
                        version = result.data[0].get('version', 'Unknown')
                        st.sidebar.info(f"PostgreSQL: {version[:50]}...")
                else:
                    st.sidebar.error(f"❌ Connection failed: {result.error}")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {e}")
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        max_rows = st.number_input("Max Rows per Query", value=1000, min_value=1, max_value=10000)
        query_timeout = st.number_input("Query Timeout (seconds)", value=30, min_value=5, max_value=300)
    
    return conn_str, {
        "max_rows": max_rows,
        "query_timeout": query_timeout
    }


def execute_sql_query_improved(db_manager: DatabaseManager, sql: str, max_rows: int):
    """Improved SQL query execution with better error handling and feedback"""
    
    if not sql.strip():
        st.warning("Please enter a SQL query")
        return
    
    # Add progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Validating query...")
        progress_bar.progress(20)
        
        # Basic SQL validation
        sql_clean = sql.strip()
        if not sql_clean.endswith(';'):
            sql_clean += ';'
        
        # Check for dangerous operations
        dangerous_patterns = [
            r'\bDROP\s+TABLE\b',
            r'\bDROP\s+DATABASE\b',
            r'\bDELETE\s+FROM\s+\w+\s*(?:WHERE|$)',
            r'\bTRUNCATE\b'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_clean, re.IGNORECASE):
                if not st.checkbox(f"⚠️ Confirm execution of potentially dangerous query"):
                    st.error("Query execution cancelled for safety")
                    progress_bar.empty()
                    status_text.empty()
                    return
        
        status_text.text("Executing query...")
        progress_bar.progress(50)
        
        # Add LIMIT if it's a SELECT query without LIMIT and max_rows is set
        if (max_rows > 0 and 
            re.search(r'^\s*SELECT\b', sql_clean, re.IGNORECASE) and 
            not re.search(r'\bLIMIT\b', sql_clean, re.IGNORECASE)):
            # Remove trailing semicolon, add LIMIT, add semicolon back
            sql_clean = sql_clean.rstrip(';') + f' LIMIT {max_rows};'
        
        # Execute query
        result = db_manager.execute_query(sql_clean)
        
        progress_bar.progress(100)
        status_text.text("Query completed!")
        
        # Clear progress indicators
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        if result.status == "success":
            # Success metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows Returned", f"{result.row_count:,}")
            with col2:
                st.metric("Execution Time", f"{result.execution_time:.3f}s")
            with col3:
                st.metric("Query ID", result.query_id)
            
            # Display data
            if result.data:
                st.success(f"Query executed successfully! Returned {result.row_count} rows.")
                
                # Convert to DataFrame
                df = pd.DataFrame(result.data)
                
                # Show data with pagination for large results
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
                        # Simple auto-visualization
                        try:
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                            
                            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                                fig = px.bar(df.head(20), x=categorical_cols[0], y=numeric_cols[0])
                                st.plotly_chart(fig, use_container_width=True)
                            elif len(numeric_cols) >= 2:
                                fig = px.scatter(df.head(100), x=numeric_cols[0], y=numeric_cols[1])
                                st.plotly_chart(fig, use_container_width=True)
                            elif len(numeric_cols) >= 1:
                                fig = px.histogram(df, x=numeric_cols[0])
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No suitable columns found for visualization")
                        except Exception as viz_error:
                            st.error(f"Visualization error: {viz_error}")
            
            else:
                if result.row_count > 0:
                    st.success(f"Query executed successfully! {result.row_count} rows affected.")
                else:
                    st.success("Query executed successfully! (No rows returned)")
        
        else:
            # Error handling
            st.error(f"Query failed: {result.error}")
            
            # Provide helpful suggestions
            if "does not exist" in result.error.lower():
                st.info("💡 Tip: Check table and column names. Use the Schema Explorer to see available tables.")
            elif "syntax error" in result.error.lower():
                st.info("💡 Tip: Check your SQL syntax. Try using the Format SQL button.")
            elif "permission denied" in result.error.lower():
                st.info("💡 Tip: Your database user may not have sufficient permissions for this operation.")
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Unexpected error: {e}")
        
        # Additional debugging info
        with st.expander("🔍 Debugging Information"):
            st.code(f"SQL Query:\n{sql}")
            st.code(f"Error Details:\n{str(e)}")


def render_sql_editor_improved(db_manager: DatabaseManager, settings: Dict):
    """Improved SQL editor interface"""
    st.markdown("### ⚡ Advanced SQL Editor")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # SQL Editor
        default_query = st.session_state.get('reused_query', 
                                           st.session_state.get('editor_sql', 
                                           "SELECT * FROM information_schema.tables WHERE table_schema = 'public' LIMIT 10;"))
        
        sql_query = st.text_area(
            "SQL Query",
            value=default_query,
            height=200,
            help="Write your SQL query here",
            key="sql_editor"
        )
        
        # Clear any reused query from session state
        if 'reused_query' in st.session_state:
            del st.session_state['reused_query']
        
        # Query controls
        col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 2])
        with col_a:
            execute_btn = st.button("▶️ Execute Query", type="primary")
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
            if st.button("📋 Sample Query"):
                sample_queries = [
                    "SELECT * FROM information_schema.tables WHERE table_schema = 'public';",
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'your_table';",
                    "SELECT pg_size_pretty(pg_database_size(current_database())) as database_size;"
                ]
                st.session_state['editor_sql'] = sample_queries[0]
                st.rerun()
    
    with col2:
        st.markdown("**Query Options**")
        max_rows = st.number_input("Max Rows", value=settings['max_rows'], min_value=1, key="sql_max_rows")
        
        st.markdown("**Quick Actions**")
        if st.button("📊 Show Tables"):
            st.session_state['editor_sql'] = "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"
            st.rerun()
        
        if st.button("🔍 Database Info"):
            st.session_state['editor_sql'] = "SELECT current_database() as database, current_user as user, version() as version;"
            st.rerun()
    
    # Execute query
    if execute_btn and sql_query.strip():
        execute_sql_query_improved(db_manager, sql_query, max_rows)


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="PostgreSQL Query Tool",
        page_icon="🐘",
        layout="wide"
    )
    
    st.title("🐘 PostgreSQL Query Tool")
    st.markdown("Enterprise database management with improved query execution")
    
    # Connection sidebar
    conn_str, settings = render_connection_sidebar()
    
    if not conn_str:
        st.info("Please configure your database connection in the sidebar")
        return
    
    # Initialize database manager
    try:
        db_manager = DatabaseManager(conn_str)
        st.success("✅ Connected to PostgreSQL database")
        
        # Render SQL editor
        render_sql_editor_improved(db_manager, settings)
        
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        
        # Troubleshooting section
        with st.expander("🔧 Troubleshooting Tips"):
            st.markdown("""
            **Common Issues:**
            1. **Wrong connection string format**: Use `postgresql://user:pass@host:port/db`
            2. **Database not running**: Ensure PostgreSQL server is started
            3. **Firewall issues**: Check if port 5432 is accessible
            4. **Wrong credentials**: Verify username, password, and database name
            5. **Missing dependencies**: Install `pip install psycopg[binary]`
            
            **For local development**: Try `postgresql://postgres:password@localhost:5432/postgres`
            """)


if __name__ == "__main__":
    main()
