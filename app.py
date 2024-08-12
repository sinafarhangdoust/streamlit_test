import streamlit as st

# Title of the app
st.title("Hello, World!")

# Simple text
st.write("This is your first Streamlit app!")

# Add a button to interact with
if st.button('Click me'):
    st.write("Hello, World!")

# Another interactive widget
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}!")

# Displaying a slider
age = st.slider("Select your age", 0, 100, 25)
st.write(f"You selected: {age} years old")