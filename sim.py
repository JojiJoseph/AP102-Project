import streamlit as st

print("Hello")

"""
TODO

[] Load arena

[] Find astar path

[] Apply DWA
"""

st.title("Choose a map")

option = st.selectbox("Select a map", [1, 2, 3])

st.button("Simulate")
