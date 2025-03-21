import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv
from database import Database

# Load environment variables
load_dotenv()

# Initialize database
db = Database()

# Page configuration
st.set_page_config(
    page_title="Tool Management System",
    page_icon="🔧",
    layout="wide"
)

# Authentication
def login():
    with st.sidebar:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["Warehouse Manager", "Technician", "Admin"])
        if st.button("Login"):
            # In a production environment, implement proper authentication
            if username and password:
                return {"id": username, "role": role}
    return None

def main():
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None

    # Login handling
    if not st.session_state.user:
        user = login()
        if user:
            st.session_state.user = user
            st.rerun()
        st.stop()

    # Main navigation
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.user['role']}")
        page = st.radio("Navigation", ["Dashboard", "Tool Management", "Reports"])
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

    if page == "Dashboard":
        show_dashboard()
    elif page == "Tool Management":
        show_tool_management()
    elif page == "Reports":
        show_reports()

def show_dashboard():
    st.title("Tool Management Dashboard")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    try:
        # Get tools data
        tools_df = db.fetch_as_dataframe("SELECT * FROM tools")
        
        total_tools = len(tools_df)
        checked_out = len(tools_df[tools_df['status'] == 'checked_out'])
        maintenance = len(tools_df[tools_df['status'] == 'maintenance'])
        
        col1.metric("Total Tools", total_tools)
        col2.metric("Checked Out", checked_out)
        col3.metric("In Maintenance", maintenance)
        
        # Tool status chart
        st.subheader("Tool Status Distribution")
        status_counts = tools_df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig = px.pie(status_counts, values='Count', names='Status', title='Tool Status Distribution')
        st.plotly_chart(fig)
        
        # Recent activity
        st.subheader("Recent Activity")
        logs_df = db.fetch_as_dataframe("""
            SELECT * FROM tool_logs 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        if not logs_df.empty:
            st.dataframe(logs_df)
        else:
            st.info("No recent activity")
            
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")

def show_tool_management():
    st.title("Tool Management")
    
    tab1, tab2 = st.tabs(["Check Out", "Check In"])
    
    with tab1:
        st.subheader("Check Out Tool")
        try:
            available_tools = db.fetch_as_dataframe("""
                SELECT * FROM tools 
                WHERE status = 'available'
            """)
            
            if not available_tools.empty:
                tool_id = st.selectbox("Select Tool", available_tools['id'].tolist())
                if st.button("Check Out Tool"):
                    # Update tool status
                    db.execute_query("""
                        UPDATE tools 
                        SET status = 'checked_out', 
                            checked_out_by = :user_id, 
                            checkout_time = :checkout_time
                        WHERE id = :tool_id
                    """, {
                        "user_id": st.session_state.user['id'],
                        "checkout_time": datetime.now(),
                        "tool_id": tool_id
                    })
                    
                    # Log the transaction
                    db.execute_query("""
                        INSERT INTO tool_logs (tool_id, user_id, action, timestamp)
                        VALUES (:tool_id, :user_id, 'checkout', :timestamp)
                    """, {
                        "tool_id": tool_id,
                        "user_id": st.session_state.user['id'],
                        "timestamp": datetime.now()
                    })
                    
                    st.success("Tool checked out successfully!")
                    st.rerun()
            else:
                st.info("No tools available for checkout")
                
        except Exception as e:
            st.error(f"Error in tool checkout: {str(e)}")
    
    with tab2:
        st.subheader("Check In Tool")
        try:
            checked_out_tools = db.fetch_as_dataframe("""
                SELECT * FROM tools 
                WHERE status = 'checked_out' 
                AND checked_out_by = :user_id
            """, {"user_id": st.session_state.user['id']})
            
            if not checked_out_tools.empty:
                tool_id = st.selectbox("Select Tool to Return", checked_out_tools['id'].tolist())
                if st.button("Return Tool"):
                    # Update tool status
                    db.execute_query("""
                        UPDATE tools 
                        SET status = 'available', 
                            checked_out_by = NULL, 
                            checkout_time = NULL
                        WHERE id = :tool_id
                    """, {"tool_id": tool_id})
                    
                    # Log the transaction
                    db.execute_query("""
                        INSERT INTO tool_logs (tool_id, user_id, action, timestamp)
                        VALUES (:tool_id, :user_id, 'checkin', :timestamp)
                    """, {
                        "tool_id": tool_id,
                        "user_id": st.session_state.user['id'],
                        "timestamp": datetime.now()
                    })
                    
                    st.success("Tool checked in successfully!")
                    st.rerun()
            else:
                st.info("No tools checked out by you")
                
        except Exception as e:
            st.error(f"Error in tool check-in: {str(e)}")

def show_reports():
    st.title("Reports")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Tool Usage", "Maintenance History", "User Activity"]
    )
    
    try:
        if report_type == "Tool Usage":
            logs_df = db.fetch_as_dataframe("SELECT * FROM tool_logs")
            
            if not logs_df.empty:
                # Tool usage frequency
                usage_counts = logs_df['tool_id'].value_counts().reset_index()
                usage_counts.columns = ['Tool ID', 'Usage Count']
                fig = px.bar(usage_counts, x='Tool ID', y='Usage Count', title='Tool Usage Frequency')
                st.plotly_chart(fig)
                
                st.subheader("Detailed Usage Logs")
                st.dataframe(logs_df)
            else:
                st.info("No usage data available")
                
        elif report_type == "Maintenance History":
            maintenance_df = db.fetch_as_dataframe("""
                SELECT * FROM tools 
                WHERE status = 'maintenance'
            """)
            
            if not maintenance_df.empty:
                st.subheader("Tools in Maintenance")
                st.dataframe(maintenance_df)
            else:
                st.info("No tools currently in maintenance")
                
        elif report_type == "User Activity":
            user_logs_df = db.fetch_as_dataframe("""
                SELECT * FROM tool_logs 
                ORDER BY timestamp DESC
            """)
            
            if not user_logs_df.empty:
                st.subheader("User Activity Timeline")
                fig = px.timeline(user_logs_df, x_start='timestamp', y='user_id', color='action')
                st.plotly_chart(fig)
                
                st.subheader("Detailed Activity Logs")
                st.dataframe(user_logs_df)
            else:
                st.info("No user activity data available")
                
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main()