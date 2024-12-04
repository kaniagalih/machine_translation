import streamlit as st

st.write("Hello world!")
x = st.text_input("Favourite Movie?")
st.write(f"Your favourite movie list: {x}")