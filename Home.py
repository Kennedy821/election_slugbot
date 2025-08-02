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
    st.image("recolored_image.png", width=75)

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
# with header:
st.title("Project Election Slugbot")


st.markdown("\n\n\n\n")
st.markdown(
    """
This is a research project looking to showcase how location based data and modern machine learning models can be used to predict major events such as the UK's 2024 election.

This is a demonstration of what is possible.

## How to predict an election without using polls

* Ways people predict elections mostly at the moment is through the use of polls. 
* New polls are different sure, but they are ultimately at their heart still polls.
* What’s so wrong with polls, not much but in world where we have advances in machine learning like language models it all feels a bit old. Maybe the reason polls are used is because they are the best predictors we’ve developed. Maybe they aren’t. This project was a flawed and quick attempt to challenge the dominance of this somewhat 20th-century concept. 

## Results

It is possible to predict the outcome of an election at the headline number of seats level, as well as somewhat possible at the seat level

What’s not possible - accounting for all available information at the seat level as well as temporally - it remains unclear if incorporating the latest political developments during a campaign makes any substantive difference in the likelihood of party will win a particular seat. With the exception of new parties standing and or parties standing down (tactical voting)



### Best performing model:

| Party        | Seats |
|--------------|-------|
| Labour       | 417   |
| Conservative | 96    |
| Lib Dems     | 41    |
| Marginal     | 33    |
| SNP          | 31    |
| Other        | 19    |
| Reform       | 15    |

Okay, so how did this compare to the actual exit poll?

### Exit poll (2024):
| Party | Seats | Seat Change |
|-------|-------|-------------|
| LAB   | 412   | +211        |
| CON   | 121   | -251        |
| LD    | 72    | +64         |
| SNP   | 9     | -39         |
| SF    | 7     | —           |
| OTH   | 29    | +15         |

The main story is that a relatively unsophisticated approach was able to make a reasonable prediction on what would happen based mostly on previous voting patterns.

"""
)




