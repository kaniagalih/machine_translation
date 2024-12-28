import streamlit as st
import sqlite3
from db import create_table, insert_data
from startup import translate

# Initialize the database table
create_table()

# Helper function: Initialize session state
def initialize_session_state():
    defaults = {
        'translations': [],
        'feedback_submitted': False,
        'name': "",
        'age': 0,
        'jawa_text': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Application title
st.title("Javanese Translation App")

# Helper function: Handle user input
def handle_user_input():
    st.session_state.name = st.text_input("Enter your name:", value=st.session_state.name, key="name_input")
    st.session_state.age = st.number_input("Enter your age:", min_value=0, max_value=120, value=st.session_state.age, key="age_input")
    
    if st.button("Submit"):
        if not st.session_state.name or st.session_state.age == 0:
            st.error("Please fill in both name and age fields.")
            st.stop()

# Helper function: Perform translation
def perform_translation():
    if st.button("Translate"):
        if st.session_state.jawa_text:
            translated_text = translate(st.session_state.jawa_text)
            st.session_state.translations.append({
                'jawa_text': st.session_state.jawa_text,
                'indonesia_text': translated_text,
                'expected': ""
            })
        else:
            st.error("Please enter text to translate!")

# Helper function: Display previous translations
def display_previous_translations():
    if st.session_state.translations:
        # Allow user to add an expected correct translation
        last_translation = st.session_state.translations[-1]

        st.success(f"Translated Text: {last_translation['indonesia_text']}")

        if not last_translation['expected']:
            expected = st.text_input("Insert the correct translation:", key="expected_translation_input")
            if st.button("Submit Correction"):
                st.session_state.translations[-1]['expected'] = expected
        
        if last_translation['expected']:
            st.write("Previous Translations:")
            for idx, translation in enumerate(st.session_state.translations):
                st.write(f"{idx + 1}. Javanese Text: {translation['jawa_text']} | Translation: {translation['indonesia_text']} | Expected: {translation['expected']}")

# Helper function: Handle feedback submission
def handle_feedback():
    if st.button("Give Feedback"):
        if st.session_state.translations:
            st.session_state.feedback_submitted = True
        else:
            st.warning("Please perform a translation first.")
    
    if st.session_state.feedback_submitted:
        rating = st.slider("Give Rating (1-5):", min_value=1, max_value=5, value=3)
        suggestion = st.text_input("Give Your Feedback:")
        
        if st.button("Send Feedback"):
            for translation in st.session_state.translations:
                insert_data(
                    st.session_state.name,
                    st.session_state.age,
                    translation['jawa_text'],
                    translation['indonesia_text'],
                    rating,
                    translation['expected'],
                    suggestion
                )
            st.success("Feedback successfully saved to the database!")
            
            # Reset session state
            # initialize_session_state()
            # st.rerun()

# Main app flow
if st.session_state.name and st.session_state.age > 0:
    st.session_state.jawa_text = st.text_area("Enter the input text in Javanese:", value=st.session_state.jawa_text)
    perform_translation()
    display_previous_translations()
    handle_feedback()
else:
    handle_user_input()
