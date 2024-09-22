# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:43:24 2024

@author: worldcontroller
"""

import streamlit as st

# st.set_page_config(
#     page_title="Hello",
#     page_icon="👋",
# )
st.set_page_config(layout="wide",
                    page_title="Hello",
                    page_icon="👋"
)


logo, header = st.columns([1,9])
with logo:
    # Display the image
    st.image("/Users/tariromashongamhende/Downloads/recolored_image.png", width=75)

    # Custom CSS to position the image
    st.markdown(
        """
        <style>
        [data-testid="stImage"] {
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
with header:
    st.title("About")


st.markdown("\n\n\n\n")
st.markdown(
    """
This is a research project looking to showcase how location based data and modern machine learning models can be used to predict major events such as the UK's 2024 election.

This is a demonstration of what is possible.

"""
)