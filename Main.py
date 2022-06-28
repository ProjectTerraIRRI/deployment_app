#!/usr/bin/env python
# coding: utf-8

### Contents of the main file ###
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")
image = Image.open('assets/irri.jpg')

col1, col2 = st.columns([2,10])
with col1:
    st.image(image, width=100)
with col2:
    st.title('Welcome!')
st.header("Do you know your cluster membership?")

