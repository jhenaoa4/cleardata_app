import streamlit as st
import pandas as pd
import numpy as np
import re
from data_cleaning import validate_dataframe
from comprenhensive_dup_cleansing import dups_manage
from full_data_enrichment import enrich_full_data

def main():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("clearview_logo.png",  use_container_width=True)
    with col2:
        st.title('CLEARdata:')
        st.header('Data Cleansing, Enrichment Automated Refinement')
    st.write("This app helps you clean, deduplicate and enrich your data efficiently.")
    st.write("Upload your CSV file and follow the instructions to refine your data.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Read the CSV file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                # Sheet name
                sheet_name = st.text_input("Please enter the Sheet Name", "Sheet1")
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        except Exception as e:
            st.error(f"Error reading the file, check the sheet name and Excel or CSV format: {e}")
            return
        # Enter the column names
        st.write("Please enter the column names for the following:")    
        id_column = st.text_input("ID Column Name Column", "id")
        if id_column not in df.columns:
            st.error(f"Column '{id_column}' not found in the uploaded file. Please check the column names.")
        sheet_name = st.text_input("Sheet Name", "Sheet1")
        if sheet_name not in df.columns:
            st.error(f"Sheet '{sheet_name}' not found in the uploaded file. Please check the sheet names.")
        company_name = st.text_input("Company Name Column", "Company")
        if company_name not in df.columns:
            st.error(f"Column '{company_name}' not found in the uploaded file. Please check the column names.")
        contact_name = st.text_input("Contact Name Column (must be full name)", "Contact")
        if contact_name not in df.columns:
            st.error(f"Column '{contact_name}' not found in the uploaded file. Please check the column names.")
        contact_email_col = st.text_input("Contact Email Column", "Email")
        if contact_email_col not in df.columns:
            st.error(f"Column '{contact_email_col}' not found in the uploaded file. Please check the column names.")
        company_email_col = st.text_input("Company Email Column", "Company Email")
        if company_email_col not in df.columns:
            st.error(f"Column '{company_email_col}' not found in the uploaded file. Please check the column names.")
        company_phone_col = st.text_input("Company Phone Column", "Phone")
        if company_phone_col not in df.columns:
            st.error(f"Column '{company_phone_col}' not found in the uploaded file. Please check the column names.")
        contact_phone_col = st.text_input("Contact Phone Column", "Contact Phone")
        if contact_phone_col not in df.columns:
            st.error(f"Column '{contact_phone_col}' not found in the uploaded file. Please check the column names.")
        url_column = st.text_input("URL Column", "url")
        if url_column not in df.columns:
            st.error(f"Column '{url_column}' not found in the uploaded file. Please check the column names.")
        
        # Refinement options
        st.header("Data Refinement Options")
        
        clean = st.checkbox("Data Cleaning (invalid data removal)")
        if clean:
            state_col = st.text_input("State Column", "State")
            if state_col not in df.columns:
                st.error(f"Column '{state_col}' not found in the uploaded file. Please check the column names.")
            address_col = st.text_input("Address Column", "Address")
            if address_col not in df.columns:
                st.error(f"Column '{address_col}' not found in the uploaded file. Please check the column names.")
            st.write("Data cleaning will remove invalid data based on the provided columns.")

        st.divider()
        enrich = st.checkbox("Data Enrichment")
        if enrich:
            phone_email = st.checkbox("Phone and Email Enrichment")
            st.write("Enriching algorith will add phone numbers and emails scrapped from the company website.")
            pms_gateway = st.checkbox("PMS and Gateway Enrichment")
            st.write("Enriching algorith will add PMS and Gateway information from the company website.")


        # Duplicate merging
        st.divider()
        merge_duplicates = st.checkbox("Manage Duplicates within the list")

        # Duplicate merge with another dataset
        st.divider()
        merge_with_another = st.checkbox("Merge dups with Another Dataset")
        if merge_with_another:
            another_file = st.file_uploader("Upload another CSV file to merge dups with", type="csv")
            if another_file is not None:
                df2 = pd.read_csv(another_file)
        
        # Perform cleaning when button is pressed
        if st.button("Clean Data"):
            if clean:
                # Perform data cleaning
                df = validate_dataframe(df, id_column, company_name, contact_name, contact_email_col, company_email_col, company_phone_col, contact_phone_col, url_column, state_col, address_col)

            if enrich:
                # Perform data enrichment
                progress_bar = st.progress(0)
                if phone_email:
                    if pms_gateway:
                        df = enrich_full_data.enrich_pms(df, company_name, phone_email= True, pms_gateway = True)
                    else:
                        df = enrich_full_data.enrich_pms(df, company_name, phone_email= True)
                else:
                    if pms_gateway:
                        df = enrich_full_data.enrich_pms(df, company_name, pms_gateway = True)

            if merge_duplicates:
                # Merge duplicates
                df = dups_manage.merge_duplicates(df, company_name, contact_name, contact_email_col, company_email_col, company_phone_col, contact_phone_col)

            if merge_with_another and another_file is not None:
                # Merge with another dataset
                df2 = pd.read_csv(another_file)
                df = dups_manage.merge_with_another(df, df2, company_name, contact_name, contact_email_col, company_email_col, company_phone_col, contact_phone_col)
            
            # Display the cleaned data
            st.write("Final Cleaned Data Sample:")
            st.dataframe(df.head(10))
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Cleaned CSV",
                data=csv,
                file_name='cleaned_data.csv',
                mime='text/csv'
            )

if __name__ == '__main__':
    main()