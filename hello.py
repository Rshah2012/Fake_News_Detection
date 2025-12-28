import streamlit as st

# This will show in your TERMINAL
print("Checking terminal logs: Script is running!")

# This will show in your BROWSER
st.sidebar.success("Sidebar Loaded")
st.title("Hello World")
st.write("If you see this, Streamlit is working perfectly.")