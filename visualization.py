import streamlit as st
import matplotlib.pyplot as plt

def plot_results(results):
    st.subheader("Model Performance")
    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values())
    st.pyplot(fig)
