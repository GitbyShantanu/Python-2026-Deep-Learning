import streamlit as st 

st.title("Marvellous Infosystems") 

st.header("Marvellous Addition Calculator") 

no1 = st.number_input("Enter first number : ", value=0, step=1) 
no2 = st.number_input("Enter second number : ", value=0, step=1)  

if st.button("Addition"):
    result = no1 + no2 

    st.success("Addition is : "+str(result)) 