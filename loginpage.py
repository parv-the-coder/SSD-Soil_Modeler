import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.database import UserManager

class LoginPage:
    def __init__(self):
        self.user_manager = UserManager()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'login_attempted' not in st.session_state:
            st.session_state.login_attempted = False
    
    def show_login_page(self):
        """Display the login page"""
        st.title("🔐 Login Required")
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    self.attempt_login(username, password)
                else:
                    st.error("Please enter both username and password")
        
        st.markdown("---")
        st.subheader("Don't have an account?")
        
        with st.form("register_form"):
            new_username = st.text_input("👤 Choose Username", placeholder="Choose a username", key="reg_user")
            new_password = st.text_input("🔒 Choose Password", type="password", placeholder="Choose a password", key="reg_pass")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm password", key="reg_confirm")
            register_button = st.form_submit_button("Register")
            
            if register_button:
                self.attempt_registration(new_username, new_password, confirm_password)
    
    def attempt_login(self, username, password):
        if self.user_manager.login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.login_attempted = True
            st.success(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.session_state.login_attempted = True
            st.error("Invalid username or password")
    
    def attempt_registration(self, username, password, confirm_password):
        if not username or not password:
            st.error("Please enter both username and password")
            return
        
        if password != confirm_password:
            st.error("Passwords do not match")
            return
        
        if self.user_manager.registerUser(username, password):
            st.success(f"Account created successfully for {username}! You can now login.")
        else:
            st.error("Username already exists or registration failed")
    
    def show_logout_button(self):
        if st.session_state.logged_in:
            st.sidebar.markdown("---")
            st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
            if st.sidebar.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.rerun()
    
    def protect_app(self, main_app_function):
        if not st.session_state.logged_in:
            self.show_login_page()
            st.stop()
        else:
            self.show_logout_button()
            main_app_function()