import streamlit as st
import pandas as pd

# Load some example data
df = pd.DataFrame({
    'Column 1': [1, 2, 3, 4],
    'Column 2': [10, 20, 30, 40]
})

# Streamlit app code
st.title('My Streamlit App')

# Display the data frame
st.write('Here is a DataFrame:')
st.write(df)

# Create a chart
st.line_chart(df)

# Add user input widgets
user_input = st.slider('Select a value:', 0, 100, 50)
st.write('You selected:', user_input)
