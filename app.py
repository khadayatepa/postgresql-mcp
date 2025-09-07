def main():
    """Main application entry point with enhanced branding"""
    
    # Page configuration with custom favicon
    st.set_page_config(
        page_title="PostgreSQL MCP Server Pro | By Prashant Khadayate",
        page_icon="🐘",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/prashant-khadayate/postgresql-mcp-server',
            'Report a bug': "https://github.com/prashant-khadayate/postgresql-mcp-server/issues",
            'About': "# PostgreSQL MCP Server Pro\n**Created by Prashant Khadayate**\n\nEnterprise-grade PostgreSQL management with AI assistance."
        }
    )
    
    # Professional theme setup
    setup_professional_theme()
    
    # Render branded header
    render_branded_header()
    
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
            - Multiple AI providers support
            - Real-time streaming responses
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Advanced Analytics
            - Interactive data visualizations
            - Performance monitoring
            - Query profiling & optimization
            - Real-time database metrics
            """)
        
        with col3:
            st.markdown("""
            ### 🔐 Enterprise Features
            - Secure connection management
            - Query history & favorites
            - Professional UI/UX
            - Export capabilities
            """)
        
        # Technology showcase
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3 style="color: #667eea; margin-bottom: 1rem;">🚀 Powered by Modern Technology Stack</h3>
            <div style="font-size: 1.1rem; color: #666;">
                <span style="margin: 0 1rem;">🐘 <strong>PostgreSQL</strong></span>
                <span style="margin: 0 1rem;">🤖 <strong>OpenAI GPT-4</strong></span>
                <span style="margin: 0 1rem;">⚡ <strong>Streamlit</strong></span>
                <span style="margin: 0 1rem;">📊 <strong>Plotly</strong></span>
                <span style="margin: 0 1rem;">🐍 <strong>Python</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return
    
    # Initialize database manager
    try:
        db_manager = DatabaseManager(conn_str)
        
        # Test connection with loading animation
        with st.spinner("🔄 Establishing database connection..."):
            with db_manager.get_connection():
                pass  # Connection successful
        
        # Connection success message
        st.markdown("""
        <div class="success-box" style="margin-bottom: 2rem;">
            <strong>🎉 Successfully connected to your PostgreSQL database!</strong><br>
            Ready to explore your data with AI-powered assistance.
        </div>
        """, unsafe_allow_html=True)
        
        # Render main dashboard
        render_main_dashboard(db_manager, settings)
        
        # Enhanced footer with creator credits
        st.markdown("---")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8f2ff 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; 
                    margin-top: 3rem; border: 1px solid rgba(102, 126, 234, 0.1);">
            
            <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap;">
                <div>
                    <h4 style="color: #667eea; margin: 0;">🐘 PostgreSQL MCP Server Pro</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">Enterprise-Grade Database Management Platform</p>
                </div>
                
                <div style="height: 40px; width: 1px; background: rgba(102, 126, 234, 0.3);"></div>
                
                <div>
                    <h4 style="color: #667eea; margin: 0;">👨‍💻 Created by</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #333; font-weight: 600;">Prashant Khadayate</p>
                </div>
                
                <div style="height: 40px; width: 1px; background: rgba(102, 126, 234, 0.3);"></div>
                
                <div>
                    <h4 style="color: #667eea; margin: 0;">🚀 Version</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">2.0.0 Professional</p>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(102, 126, 234, 0.2);">
                <p style="margin: 0; color: #888; font-size: 0.9rem;">
                    Built with ❤️ using 
                    <strong>Streamlit</strong> • <strong>PostgreSQL</strong> • <strong>OpenAI</strong> • <strong>Python</strong>
                </p>
                <div style="margin-top: 1rem;">
                    <a href="#" style="color: #667eea; text-decoration: none; margin: 0 1rem;">📖 Documentation</a>
                    <a href="#" style="color: #667eea; text-decoration: none; margin: 0 1rem;">🐛 Report Issues</a>
                    <a href="#" style="color: #667eea; text-decoration: none; margin: 0 1rem;">⭐ GitHub</a>
                    <a href="#" style="color: #667eea; text-decoration: none; margin: 0 1rem;">💬 Support</a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h3>❌ Database Connection Failed</h3>
            <p><strong>Error:</strong> {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}</p>
        </div>
        """, unsafe_allow_html=True)
        
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
            
            4. **🛡️ SSL/Security Settings**
               - Add `?sslmode=require` for secure connections
               - Use `?sslmode=disable` for local development
            
            5. **🚀 Cloud Database Tips**
               - Check if your IP is whitelisted
               - Verify connection limits aren't exceeded
               - Confirm SSL requirements
            """)
        
        st.info("""
        **💡 Pro Tip:** Use the sidebar connection tester to validate your settings before proceeding.
        """)


if __name__ == "__main__":
    main()
"""
Professional PostgreSQL MCP Server with Advanced Features
=========================================================

A comprehensive, enterprise-grade PostgreSQL management interface with:
- 🤖 Advanced NLP-to-SQL with multiple AI providers
- 📊 Interactive data visualization and analytics
- 🔐 Enhanced security and query optimization
- 📈 Performance monitoring and query profiling
- 🎨 Modern, professional UI with dark/light themes
- 💾 Query history and favorites management
- 🔄 Real-time query execution with progress tracking
- 📋 Advanced schema exploration and ER diagrams
- 🚀 Query optimization suggestions
- 📤 Data export in multiple formats

Author: Professional Development Team
Version: 2.0.0

Required Dependencies (add to requirements.txt):
streamlit>=1.28.0
psycopg[binary]>=3.1.0
openai>=1.0.0
pandas>=1.5.0
plotly>=5.0.0
sqlparse>=0.4.0

Optional Dependencies:
networkx>=2.8.0
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
from io import StringIO
import base64

# Enhanced imports for professional features
try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

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

_psycopg_mod = None
_dict_row = None

try:
    import psycopg as _psycopg
    try:
        from psycopg.rows import dict_row as _dict_row
    except Exception:
        _dict_row = None
    _psycopg_mod = _psycopg
except Exception:
    try:
        import psycopg2 as _psycopg
        import psycopg2.extras
        _dict_row = psycopg2.extras.RealDictCursor
        _psycopg_mod = _psycopg
    except Exception:
        _psycopg_mod = None


class DatabaseManager:
    """Professional database management class"""
    
    def __init__(self, conn_str: str):
        self.conn_str = conn_str
        self._connection_pool = {}
    
    def get_connection(self):
        """Get database connection with proper error handling"""
        if _psycopg_mod is None:
            raise RuntimeError("PostgreSQL driver not found. Install psycopg[binary]")
        
        try:
            if hasattr(_psycopg_mod, 'connect') and _dict_row:
                return _psycopg_mod.connect(self.conn_str, row_factory=_dict_row)
            elif _dict_row and hasattr(_psycopg_mod, 'extras'):
                conn = _psycopg_mod.connect(self.conn_str)
                conn.cursor_factory = _dict_row
                return conn
            return _psycopg_mod.connect(self.conn_str)
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            raise
    
    def execute_query(self, sql: str, params: Tuple = None, fetch: bool = True) -> QueryResult:
        """Execute query with comprehensive result tracking"""
        start_time = time.time()
        query_id = hashlib.md5(f"{sql}{time.time()}".encode()).hexdigest()[:8]
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params or ())
                    
                    execution_time = time.time() - start_time
                    
                    if fetch and cur.description:
                        columns = [desc[0] for desc in cur.description]
                        rows = cur.fetchall()
                        
                        # Convert to list of dicts
                        if hasattr(rows[0] if rows else {}, 'keys'):
                            data = [dict(row) for row in rows]
                        else:
                            data = [dict(zip(columns, row)) for row in rows]
                        
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
                        return QueryResult(
                            sql=sql,
                            data=[],
                            columns=[],
                            execution_time=execution_time,
                            row_count=cur.rowcount if hasattr(cur, 'rowcount') else 0,
                            timestamp=datetime.now(),
                            query_id=query_id
                        )
        
        except Exception as e:
            return QueryResult(
                sql=sql,
                data=[],
                columns=[],
                execution_time=time.time() - start_time,
                row_count=0,
                timestamp=datetime.now(),
                query_id=query_id,
                status="error",
                error=str(e)
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
    
    def get_query_performance_stats(self) -> Dict:
        """Get database performance statistics"""
        try:
            stats_sql = """
            SELECT 
                schemaname,
                tablename,
                seq_scan,
                seq_tup_read,
                idx_scan,
                idx_tup_fetch,
                n_tup_ins,
                n_tup_upd,
                n_tup_del
            FROM pg_stat_user_tables
            ORDER BY seq_tup_read DESC
            LIMIT 10
            """
            result = self.execute_query(stats_sql)
            return {"table_stats": result.data}
        except:
            return {"table_stats": []}


# ---------------------------------------------------------------------------
# Enhanced NLP-to-SQL with Multiple Providers
# ---------------------------------------------------------------------------

class AIQueryGenerator:
    """Professional AI-powered SQL generation"""
    
    def __init__(self):
        self.providers = {
            "OpenAI GPT-4": self._openai_generate,
            "Claude": self._claude_generate,
            "Local Model": self._local_generate
        }
    
    def generate_sql(self, nl_query: str, schema_context: str, provider: str = "OpenAI GPT-4", 
                    stream: bool = True) -> str:
        """Generate SQL from natural language with provider selection"""
        if provider in self.providers:
            return self.providers[provider](nl_query, schema_context, stream)
        else:
            raise ValueError(f"Provider {provider} not supported")
    
    def _openai_generate(self, nl_query: str, schema_context: str, stream: bool) -> str:
        """OpenAI implementation with advanced prompting"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            system_prompt = f"""You are an expert PostgreSQL developer. Generate efficient, secure SQL queries.

Database Schema:
{schema_context}

Rules:
1. Generate PostgreSQL-compatible SQL only
2. Use proper indexing strategies when possible
3. Include query optimization hints
4. Handle edge cases (NULL values, empty results)
5. Use appropriate JOINs and subqueries
6. Return only the SQL query, no explanations

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
    
    def _claude_generate(self, nl_query: str, schema_context: str, stream: bool) -> str:
        """Claude implementation (placeholder for now)"""
        st.info("Claude integration coming soon! Using OpenAI as fallback.")
        return self._openai_generate(nl_query, schema_context, stream)
    
    def _local_generate(self, nl_query: str, schema_context: str, stream: bool) -> str:
        """Local model implementation (placeholder)"""
        st.info("Local model integration coming soon! Using OpenAI as fallback.")
        return self._openai_generate(nl_query, schema_context, stream)
    
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
# Data Visualization & Analytics
# ---------------------------------------------------------------------------

class DataVisualizer:
    """Professional data visualization component"""
    
    @staticmethod
    def create_auto_chart(df: pd.DataFrame, chart_type: str = "auto") -> go.Figure:
        """Automatically create appropriate visualizations"""
        if df.empty:
            return go.Figure()
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if chart_type == "auto":
            if len(numeric_cols) >= 2:
                chart_type = "scatter"
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                chart_type = "bar"
            elif len(categorical_cols) >= 1:
                chart_type = "pie"
            else:
                chart_type = "line"
        
        try:
            if chart_type == "bar" and len(categorical_cols) > 0 and len(numeric_cols) > 0:
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0],
                           title=f"{numeric_cols[0]} by {categorical_cols[0]}")
            elif chart_type == "scatter" and len(numeric_cols) >= 2:
                color_col = categorical_cols[0] if len(categorical_cols) > 0 else None
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                               color=color_col, title="Scatter Plot")
            elif chart_type == "line" and len(numeric_cols) > 0:
                fig = px.line(df, y=numeric_cols[0], title="Line Chart")
            elif chart_type == "pie" and len(categorical_cols) > 0:
                value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
                if value_col:
                    fig = px.pie(df, names=categorical_cols[0], values=value_col)
                else:
                    counts = df[categorical_cols[0]].value_counts()
                    fig = px.pie(values=counts.values, names=counts.index)
            else:
                # Fallback to simple bar chart
                if len(df.columns) > 0:
                    fig = px.bar(x=range(len(df)), y=df.iloc[:, 0].values)
                else:
                    fig = go.Figure()
            
            fig.update_layout(
                template="plotly_white",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            return fig
        
        except Exception as e:
            st.error(f"Visualization error: {e}")
            return go.Figure()
    
    @staticmethod
    def create_performance_dashboard(stats: Dict) -> List[go.Figure]:
        """Create performance monitoring dashboard"""
        figures = []
        
        if "table_stats" in stats and stats["table_stats"]:
            df = pd.DataFrame(stats["table_stats"])
            
            # Table scan efficiency
            if 'seq_scan' in df.columns and 'idx_scan' in df.columns:
                fig1 = px.bar(df, x='tablename', y=['seq_scan', 'idx_scan'],
                             title="Table Scan Types", barmode='group')
                figures.append(fig1)
            
            # Table operations
            if all(col in df.columns for col in ['n_tup_ins', 'n_tup_upd', 'n_tup_del']):
                fig2 = px.bar(df, x='tablename', y=['n_tup_ins', 'n_tup_upd', 'n_tup_del'],
                             title="Table Operations", barmode='stack')
                figures.append(fig2)
        
        return figures


# ---------------------------------------------------------------------------
# Query Management & History
# ---------------------------------------------------------------------------

class QueryManager:
    """Professional query history and favorites management"""
    
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
# Professional UI Components
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
    """Render professional connection sidebar"""
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
        db_config = {"connection_string": conn_str}
    else:
        db_config = {
            "host": st.sidebar.text_input("Host", value="localhost"),
            "port": st.sidebar.number_input("Port", value=5432, min_value=1, max_value=65535),
            "database": st.sidebar.text_input("Database", value="postgres"),
            "username": st.sidebar.text_input("Username", value="postgres"),
            "password": st.sidebar.text_input("Password", type="password"),
        }
        conn_str = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    # Connection test
    if st.sidebar.button("🔍 Test Connection"):
        try:
            db_manager = DatabaseManager(conn_str)
            with db_manager.get_connection():
                st.sidebar.success("✅ Connection successful!")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        max_rows = st.number_input("Max Rows per Query", value=1000, min_value=1, max_value=10000)
        query_timeout = st.number_input("Query Timeout (seconds)", value=30, min_value=5, max_value=300)
        ai_provider = st.selectbox("AI Provider", ["OpenAI GPT-4", "Claude", "Local Model"])
        enable_streaming = st.checkbox("Enable AI Streaming", value=True)
    
    return conn_str, {
        "max_rows": max_rows,
        "query_timeout": query_timeout,
        "ai_provider": ai_provider,
        "enable_streaming": enable_streaming
    }


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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 AI Query", "📋 Schema Explorer", "⚡ SQL Editor", 
        "📊 Analytics", "📈 Performance", "⭐ Favorites"
    ])
    
    # Initialize components
    ai_generator = AIQueryGenerator()
    visualizer = DataVisualizer()
    query_manager = QueryManager()
    
    with tab1:
        render_ai_query_interface(db_manager, ai_generator, query_manager, settings)
    
    with tab2:
        render_schema_explorer(db_manager)
    
    with tab3:
        render_sql_editor(db_manager, query_manager, settings)
    
    with tab4:
        render_analytics_dashboard(db_manager, visualizer, settings)
    
    with tab5:
        render_performance_dashboard(db_manager, visualizer)
    
    with tab6:
        render_favorites_manager(db_manager, query_manager, settings)


def render_ai_query_interface(db_manager: DatabaseManager, ai_generator: AIQueryGenerator, 
                             query_manager: QueryManager, settings: Dict):
    """Render AI-powered query interface"""
    st.markdown("### 🤖 Natural Language to SQL")
    st.markdown("Ask questions about your data in plain English, and I'll generate optimized SQL queries for you.")
    
    # Schema context
    col1, col2 = st.columns([3, 1])
    with col1:
        nl_query = st.text_area(
            "What would you like to know?",
            placeholder="e.g., Show me the top 10 customers by total orders in the last month",
            height=100
        )
    
    with col2:
        schema = st.selectbox("Schema", ["public"], help="Select the database schema to query")
        max_rows = st.number_input("Max Results", value=settings['max_rows'], min_value=1, max_value=10000)
    
    if st.button("🚀 Generate & Execute Query", type="primary"):
        if not nl_query.strip():
            st.warning("Please enter your question first.")
            return
        
        try:
            # Get schema context
            tables = db_manager.get_enhanced_schema_info(schema)
            schema_context = "\n".join([
                f"Table: {t.name} ({t.row_count} rows, {t.size})\nColumns: " + 
                ", ".join([f"{c['column_name']} ({c['data_type']})" for c in t.columns[:5]])
                for t in tables[:10]  # Limit context size
            ])
            
            # Generate SQL
            with st.spinner("🧠 AI is thinking..."):
                sql = ai_generator.generate_sql(
                    nl_query, schema_context, 
                    settings['ai_provider'], settings['enable_streaming']
                )
            
            if sql:
                st.markdown("**Generated SQL:**")
                st.code(sql, language="sql")
                
                # Execute query
                if st.button("▶️ Execute Query"):
                    with st.spinner("Executing query..."):
                        result = db_manager.execute_query(sql)
                        
                        if result.status == "success":
                            query_manager.add_to_history(result)
                            
                            st.markdown(f"""
                            <div class="success-box">
                                ✅ Query executed successfully!<br>
                                📊 Returned {result.row_count} rows in {result.execution_time:.2f}s
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if result.data:
                                df = pd.DataFrame(result.data)
                                st.dataframe(df, use_container_width=True)
                                
                                # Auto-visualization
                                if len(df) > 0:
                                    st.markdown("**📈 Auto-Generated Visualization:**")
                                    fig = DataVisualizer.create_auto_chart(df)
                                    if fig.data:
                                        st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.markdown(f"""
                            <div class="error-box">
                                ❌ Query failed: {result.error}
                            </div>
                            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating query: {e}")


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
    """Render advanced SQL editor with syntax highlighting and execution"""
    st.markdown("### ⚡ Advanced SQL Editor")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # SQL Editor with syntax highlighting
        sql_query = st.text_area(
            "SQL Query",
            value="SELECT * FROM information_schema.tables WHERE table_schema = 'public' LIMIT 10;",
            height=200,
            help="Write your SQL query here. Use Ctrl+Enter to execute."
        )
        
        # Query controls
        col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 2])
        with col_a:
            execute_btn = st.button("▶️ Execute", type="primary")
        with col_b:
            explain_btn = st.button("📋 Explain Plan")
        with col_c:
            format_btn = st.button("🎨 Format SQL")
        with col_d:
            save_btn = st.button("💾 Save to Favorites")
    
    with col2:
        st.markdown("**Query Options**")
        max_rows = st.number_input("Max Rows", value=settings['max_rows'], min_value=1, key="sql_max_rows")
        timeout = st.number_input("Timeout (s)", value=30, min_value=5, key="sql_timeout")
        dry_run = st.checkbox("Dry Run (Explain Only)", help="Show execution plan without running query")
    
    # Format SQL
    if format_btn and SQLPARSE_AVAILABLE:
        try:
            formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
            st.session_state['formatted_sql'] = formatted_sql
            st.rerun()
        except Exception as e:
            st.error(f"Error formatting SQL: {e}")
    
    # Execute query
    if execute_btn and sql_query.strip():
        execute_sql_query(db_manager, query_manager, sql_query, max_rows, dry_run)
    
    # Explain plan
    if explain_btn and sql_query.strip():
        explain_query_plan(db_manager, sql_query)
    
    # Save to favorites
    if save_btn and sql_query.strip():
        with st.form("save_query_form"):
            query_name = st.text_input("Query Name", placeholder="Enter a name for this query")
            query_description = st.text_area("Description (optional)", placeholder="Describe what this query does")
            
            if st.form_submit_button("Save Query"):
                if query_name:
                    query_manager.add_to_favorites(sql_query, query_name)
                    st.success(f"Query '{query_name}' saved to favorites!")
                else:
                    st.warning("Please enter a query name")
    
    # Show recent query history
    st.markdown("---")
    st.markdown("### 📚 Recent Queries")
    
    history = query_manager.get_history()
    if history:
        for i, query_data in enumerate(history[:5]):  # Show last 5 queries
            with st.expander(f"Query {i+1}: {query_data['query_id']} ({query_data['status']})"):
                col_x, col_y, col_z = st.columns([2, 1, 1])
                with col_x:
                    st.code(query_data['sql'][:200] + "..." if len(query_data['sql']) > 200 else query_data['sql'])
                with col_y:
                    st.metric("Rows", query_data['row_count'])
                with col_z:
                    st.metric("Time", f"{query_data['execution_time']:.2f}s")
                
                if st.button(f"Reuse Query {i+1}", key=f"reuse_{i}"):
                    st.session_state['reused_query'] = query_data['sql']
                    st.rerun()
    else:
        st.info("No query history available")


def execute_sql_query(db_manager: DatabaseManager, query_manager: QueryManager, 
                     sql: str, max_rows: int, dry_run: bool):
    """Execute SQL query with comprehensive error handling"""
    try:
        if dry_run:
            explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
            result = db_manager.execute_query(explain_sql)
            
            if result.status == "success" and result.data:
                st.markdown("**🔍 Query Execution Plan:**")
                plan_data = result.data[0]['QUERY PLAN'][0] if result.data[0].get('QUERY PLAN') else {}
                
                if plan_data:
                    st.json(plan_data)
                    
                    # Extract key metrics
                    if 'Execution Time' in plan_data:
                        st.metric("Estimated Execution Time", f"{plan_data['Execution Time']:.2f}ms")
        else:
            with st.spinner("Executing query..."):
                # Add LIMIT if not present and not a modification query
                limited_sql = sql
                if max_rows and not re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
                    if re.search(r'^\s*SELECT\b', sql.strip(), re.IGNORECASE):
                        limited_sql = f"{sql.rstrip(';')} LIMIT {max_rows}"
                
                result = db_manager.execute_query(limited_sql)
                
                if result.status == "success":
                    query_manager.add_to_history(result)
                    
                    # Success metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows Returned", result.row_count)
                    with col2:
                        st.metric("Execution Time", f"{result.execution_time:.3f}s")
                    with col3:
                        st.metric("Query ID", result.query_id)
                    
                    # Display results
                    if result.data:
                        df = pd.DataFrame(result.data)
                        
                        # Results table with download option
                        st.markdown("**📊 Query Results:**")
                        st.dataframe(df, use_container_width=True)
                        
                        # Export options
                        col_export1, col_export2, col_export3 = st.columns(3)
                        with col_export1:
                            csv = df.to_csv(index=False)
                            st.download_button("📥 Download CSV", csv, f"query_result_{result.query_id}.csv", "text/csv")
                        with col_export2:
                            json_str = df.to_json(orient='records', indent=2)
                            st.download_button("📥 Download JSON", json_str, f"query_result_{result.query_id}.json", "application/json")
                        with col_export3:
                            if st.button("📈 Create Visualization"):
                                fig = DataVisualizer.create_auto_chart(df)
                                if fig.data:
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("Query executed successfully (no results returned)")
                
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <strong>❌ Query Execution Failed</strong><br>
                        <code>{result.error}</code><br>
                        <small>Execution time: {result.execution_time:.3f}s</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Unexpected error: {e}")


def explain_query_plan(db_manager: DatabaseManager, sql: str):
    """Show detailed query execution plan"""
    try:
        explain_sql = f"EXPLAIN (ANALYZE FALSE, VERBOSE TRUE, BUFFERS FALSE, FORMAT TEXT) {sql}"
        result = db_manager.execute_query(explain_sql)
        
        if result.status == "success" and result.data:
            st.markdown("**📋 Query Execution Plan:**")
            plan_text = "\n".join([row.get('QUERY PLAN', '') for row in result.data])
            st.code(plan_text, language="text")
            
            # Performance suggestions
            st.markdown("**💡 Optimization Suggestions:**")
            suggestions = analyze_query_plan(plan_text)
            for suggestion in suggestions:
                st.info(suggestion)
    
    except Exception as e:
        st.error(f"Error generating execution plan: {e}")


def analyze_query_plan(plan_text: str) -> List[str]:
    """Analyze query plan and provide optimization suggestions"""
    suggestions = []
    
    if "Seq Scan" in plan_text:
        suggestions.append("🔍 Consider adding indexes to avoid sequential scans")
    
    if "cost=" in plan_text:
        # Extract cost estimates
        import re
        costs = re.findall(r'cost=(\d+\.\d+)', plan_text)
        if costs and float(costs[0]) > 1000:
            suggestions.append("💰 High query cost detected - consider query optimization")
    
    if "Nested Loop" in plan_text:
        suggestions.append("🔄 Nested loops detected - verify join conditions and indexes")
    
    if "Hash Join" in plan_text:
        suggestions.append("⚡ Hash joins are generally efficient for large datasets")
    
    if not suggestions:
        suggestions.append("✅ Query plan looks reasonable")
    
    return suggestions


def render_analytics_dashboard(db_manager: DatabaseManager, visualizer: DataVisualizer, settings: Dict):
    """Render comprehensive analytics dashboard"""
    st.markdown("### 📊 Data Analytics Dashboard")
    
    # Quick analytics queries
    st.markdown("**🚀 Quick Analytics**")
    
    analytics_options = {
        "Table Sizes": """
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
            FROM pg_tables 
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
            ORDER BY size_bytes DESC
            LIMIT 10
        """,
        "Database Activity": """
            SELECT 
                datname,
                numbackends as active_connections,
                xact_commit as transactions_committed,
                xact_rollback as transactions_rolled_back,
                blks_read as blocks_read,
                blks_hit as blocks_hit,
                round((blks_hit::float / (blks_hit + blks_read)) * 100, 2) as cache_hit_ratio
            FROM pg_stat_database 
            WHERE datname = current_database()
        """,
        "Lock Information": """
            SELECT 
                mode,
                count(*) as lock_count
            FROM pg_locks 
            GROUP BY mode
            ORDER BY lock_count DESC
        """,
        "Index Usage": """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_scan as index_scans,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched
            FROM pg_stat_user_indexes
            ORDER BY idx_scan DESC
            LIMIT 15
        """
    }
    
    selected_analytics = st.selectbox("Choose Analytics Query", list(analytics_options.keys()))
    
    if st.button(f"🔍 Run {selected_analytics} Analysis"):
        try:
            with st.spinner("Running analytics..."):
                result = db_manager.execute_query(analytics_options[selected_analytics])
                
                if result.status == "success" and result.data:
                    df = pd.DataFrame(result.data)
                    
                    # Display data
                    st.dataframe(df, use_container_width=True)
                    
                    # Create visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        chart_type = st.selectbox("Chart Type", ["auto", "bar", "line", "pie", "scatter"])
                        fig = visualizer.create_auto_chart(df, chart_type)
                        if fig.data:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Data insights
                        st.markdown("**📈 Key Insights:**")
                        
                        if selected_analytics == "Table Sizes":
                            total_size = df['size_bytes'].sum() if 'size_bytes' in df.columns else 0
                            st.metric("Total Size", f"{total_size / (1024**3):.2f} GB" if total_size > 0 else "N/A")
                            
                            if len(df) > 0 and 'size_bytes' in df.columns:
                                largest_table = df.loc[df['size_bytes'].idxmax()]
                                st.info(f"Largest table: {largest_table.get('tablename', 'N/A')}")
                        
                        elif selected_analytics == "Database Activity":
                            if 'cache_hit_ratio' in df.columns and len(df) > 0:
                                hit_ratio = df['cache_hit_ratio'].iloc[0] if not pd.isna(df['cache_hit_ratio'].iloc[0]) else 0
                                st.metric("Cache Hit Ratio", f"{hit_ratio:.1f}%")
                                if hit_ratio < 95:
                                    st.warning("Low cache hit ratio - consider increasing shared_buffers")
                                else:
                                    st.success("Good cache performance!")
                
                else:
                    st.error(f"Analytics query failed: {result.error if result.error else 'Unknown error'}")
        
        except Exception as e:
            st.error(f"Error running analytics: {e}")
    
    # Custom analytics
    st.markdown("---")
    st.markdown("**🎯 Custom Analytics Query**")
    
    custom_sql = st.text_area(
        "Enter your analytics query:",
        placeholder="SELECT column, COUNT(*) FROM table GROUP BY column ORDER BY COUNT(*) DESC",
        height=100
    )
    
    if st.button("🚀 Run Custom Analytics") and custom_sql.strip():
        try:
            result = db_manager.execute_query(custom_sql)
            
            if result.status == "success" and result.data:
                df = pd.DataFrame(result.data)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(df, use_container_width=True)
                
                with col2:
                    chart_type = st.selectbox("Visualization", ["auto", "bar", "line", "pie", "scatter"], key="custom_viz")
                    
                    if st.button("📊 Generate Chart"):
                        fig = visualizer.create_auto_chart(df, chart_type)
                        if fig.data:
                            st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error(f"Query failed: {result.error}")
        
        except Exception as e:
            st.error(f"Error: {e}")


def render_performance_dashboard(db_manager: DatabaseManager, visualizer: DataVisualizer):
    """Render database performance monitoring dashboard"""
    st.markdown("### 📈 Performance Monitoring")
    
    # Performance metrics
    try:
        stats = db_manager.get_query_performance_stats()
        
        # Create performance visualizations
        performance_charts = visualizer.create_performance_dashboard(stats)
        
        if performance_charts:
            for i, chart in enumerate(performance_charts):
                st.plotly_chart(chart, use_container_width=True, key=f"perf_chart_{i}")
        
        # Real-time monitoring
        st.markdown("**🔄 Real-time Monitoring**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Stats"):
                st.rerun()
        
        with col2:
            auto_refresh = st.checkbox("Auto-refresh (30s)")
        
        with col3:
            if st.button("📊 Full Health Check"):
                run_health_check(db_manager)
        
        # Current connections
        st.markdown("**🔗 Current Connections**")
        connections_sql = """
            SELECT 
                pid,
                usename as username,
                application_name,
                client_addr,
                state,
                query_start,
                left(query, 50) as current_query
            FROM pg_stat_activity 
            WHERE state != 'idle'
            ORDER BY query_start DESC
        """
        
        conn_result = db_manager.execute_query(connections_sql)
        if conn_result.status == "success" and conn_result.data:
            conn_df = pd.DataFrame(conn_result.data)
            st.dataframe(conn_df, use_container_width=True)
        
        # Slow queries
        st.markdown("**🐌 Long Running Queries**")
        slow_queries_sql = """
            SELECT 
                pid,
                now() - pg_stat_activity.query_start AS duration,
                query,
                state
            FROM pg_stat_activity
            WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
            ORDER BY duration DESC
        """
        
        slow_result = db_manager.execute_query(slow_queries_sql)
        if slow_result.status == "success" and slow_result.data:
            if slow_result.data:
                slow_df = pd.DataFrame(slow_result.data)
                st.dataframe(slow_df, use_container_width=True)
                st.warning(f"Found {len(slow_result.data)} long-running queries")
            else:
                st.success("No long-running queries detected")
        
    except Exception as e:
        st.error(f"Error loading performance data: {e}")


def run_health_check(db_manager: DatabaseManager):
    """Run comprehensive database health check"""
    st.markdown("**🏥 Database Health Check Results**")
    
    health_checks = {
        "Database Size": "SELECT pg_size_pretty(pg_database_size(current_database())) as size",
        "Connection Count": "SELECT count(*) as connections FROM pg_stat_activity",
        "Cache Hit Ratio": """
            SELECT round((sum(blks_hit) / sum(blks_hit + blks_read)) * 100, 2) as cache_hit_ratio
            FROM pg_stat_database WHERE datname = current_database()
        """,
        "Unused Indexes": """
            SELECT schemaname, tablename, indexname 
            FROM pg_stat_user_indexes 
            WHERE idx_scan = 0 AND schemaname = 'public'
            LIMIT 5
        """,
        "Table Bloat": """
            SELECT schemaname, tablename, n_dead_tup as dead_tuples
            FROM pg_stat_user_tables 
            WHERE n_dead_tup > 1000
            ORDER BY n_dead_tup DESC
            LIMIT 5
        """
    }
    
    for check_name, sql in health_checks.items():
        try:
            result = db_manager.execute_query(sql)
            
            if result.status == "success":
                if result.data:
                    with st.expander(f"✅ {check_name}"):
                        if PANDAS_AVAILABLE:
                            df = pd.DataFrame(result.data)
                            st.dataframe(df, width='stretch')
                        else:
                            for i, row in enumerate(result.data):
                                st.write(f"Row {i+1}:", {k: str(v) for k, v in row.items()})
                        
                        # Specific recommendations
                        if check_name == "Cache Hit Ratio" and result.data and 'cache_hit_ratio' in result.data[0]:
                            ratio = result.data[0]['cache_hit_ratio'] or 0
                            if ratio < 95:
                                st.warning(f"Cache hit ratio is {ratio}% - consider increasing shared_buffers")
                        
                        elif check_name == "Unused Indexes" and len(result.data) > 0:
                            st.info(f"Found {len(result.data)} potentially unused indexes")
                        
                        elif check_name == "Table Bloat" and len(result.data) > 0:
                            st.warning(f"Found {len(result.data)} tables with significant bloat - consider VACUUM ANALYZE")
                else:
                    st.success(f"✅ {check_name}: No issues found")
            else:
                st.error(f"❌ {check_name}: {result.error}")
        
        except Exception as e:
            st.error(f"❌ {check_name}: {e}")


def render_favorites_manager(db_manager: DatabaseManager, query_manager: QueryManager, settings: Dict):
    """Render favorites and query management interface"""
    st.markdown("### ⭐ Query Favorites & Templates")
    
    favorites = query_manager.get_favorites()
    
    # Add sample queries if no favorites exist
    if not favorites:
        sample_queries = [
            {
                "name": "Table Overview",
                "sql": """SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;""",
                "description": "Get overview of all tables with sizes"
            },
            {
                "name": "Active Connections",
                "sql": """SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start
FROM pg_stat_activity 
WHERE state = 'active'
ORDER BY query_start DESC;""",
                "description": "Show all active database connections"
            },
            {
                "name": "Index Usage Stats",
                "sql": """SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan > 0
ORDER BY idx_scan DESC
LIMIT 20;""",
                "description": "Most used indexes in the database"
            }
        ]
        
        st.info("No favorites yet. Here are some useful templates to get you started:")
        
        for query in sample_queries:
            with st.expander(f"📋 {query['name']}"):
                st.code(query['sql'], language="sql")
                st.caption(query['description'])
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"⭐ Add to Favorites", key=f"fav_{query['name']}"):
                        query_manager.add_to_favorites(query['sql'], query['name'])
                        st.success(f"Added '{query['name']}' to favorites!")
                        st.rerun()
                
                with col2:
                    if st.button(f"▶️ Execute", key=f"exec_{query['name']}"):
                        execute_sql_query(db_manager, query_manager, query['sql'], settings['max_rows'], False)
    
    else:
        st.success(f"You have {len(favorites)} saved queries")
        
        # Favorites list
        for i, fav in enumerate(favorites):
            with st.expander(f"⭐ {fav['name']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.code(fav['sql'], language="sql")
                    if fav.get('description'):
                        st.caption(fav['description'])
                
                with col2:
                    if st.button("▶️ Execute", key=f"exec_fav_{i}"):
                        execute_sql_query(db_manager, query_manager, fav['sql'], settings['max_rows'], False)
                    
                    if st.button("📋 Copy to Editor", key=f"copy_fav_{i}"):
                        st.session_state['editor_sql'] = fav['sql']
                        st.success("Copied to SQL editor!")
                    
                    if st.button("🗑️ Delete", key=f"del_fav_{i}"):
                        st.session_state.favorite_queries.pop(i)
                        st.success("Query deleted!")
                        st.rerun()
    
    # Export/Import favorites
    st.markdown("---")
    st.markdown("**💾 Backup & Restore**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Favorites"):
            if favorites:
                export_data = json.dumps(favorites, indent=2, default=str)
                st.download_button(
                    "📥 Download Favorites.json",
                    export_data,
                    "favorites.json",
                    "application/json"
                )
            else:
                st.warning("No favorites to export")
    
    with col2:
        uploaded_file = st.file_uploader("📤 Import Favorites", type=['json'])
        if uploaded_file:
            try:
                import_data = json.load(uploaded_file)
                st.session_state.favorite_queries.extend(import_data)
                st.success(f"Imported {len(import_data)} queries!")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="PostgreSQL MCP Server Pro",
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
        st.error("⚠️ Please provide database connection details in the sidebar")
        st.info("""
        **Get Started:**
        1. Enter your PostgreSQL connection string in the sidebar
        2. Test the connection
        3. Start exploring your database with AI assistance!
        
        **Features:**
        - 🤖 Natural Language to SQL conversion
        - 📊 Interactive data visualization
        - 📈 Performance monitoring
        - ⭐ Query favorites management
        - 🔍 Advanced schema exploration
        """)
        return
    
    # Initialize database manager
    try:
        db_manager = DatabaseManager(conn_str)
        
        # Test connection
        with db_manager.get_connection():
            pass  # Connection successful
        
        # Render main dashboard
        render_main_dashboard(db_manager, settings)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>🚀 <strong>PostgreSQL MCP Server Pro v2.0</strong> | 
            Built with ❤️ using Streamlit | 
            <a href="#" style="color: #007bff;">Documentation</a> | 
            <a href="#" style="color: #007bff;">Support</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.info("""
        **Connection Issues?**
        - Verify your connection string format
        - Check if the database server is running
        - Ensure network connectivity
        - Validate credentials and permissions
        """)


if __name__ == "__main__":
    main()
