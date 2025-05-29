import streamlit as st
import pandas as pd
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
    
    # File uploader
    st.header("1. Upload your file")
    uploaded_file = st.file_uploader("Choose an Excel or CSV file and follow the instructions to refine your data.", type=["csv", "xlsx"])
    
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
            st.error(f"Check the sheet name: {e}")
            return
        # Enter the column names
        st.header("2. Enter Column Names")
        st.write("Please enter the exact names of the columns as they appear in your Excel file for the following fields.")
        st.write("Note: Your file may not include all the columns listed below — only fill in the ones that are present.")
        col1, col2 = st.columns([2, 2])
        with col1:  
            id_column = st.text_input("ID Column*", "id")
            if id_column not in df.columns:
                st.error(f"Column '{id_column}' not found in the uploaded file. Please check the column names.")
            company_name = st.text_input("Company Name Column*", "Company")
            if company_name not in df.columns:
                st.error(f"Column '{company_name}' not found in the uploaded file. Please check the column names.")
            contact_name = st.text_input("Contact Name Column (must be full name)", "")
            if contact_name and contact_name not in df.columns:
                st.warning(f"Column '{contact_name}' not found in the uploaded file. Please check the column names. If there's not Contact Name column please leave it empty.")
            contact_email_col = st.text_input("Contact Email Column", "")
            if contact_email_col and contact_email_col not in df.columns:
                st.warning(f"Column '{contact_email_col}' not found in the uploaded file. Please check the column names. If there's no Contact Email column, please leave it empty.")
            
        with col2:
            url_column = st.text_input("URL Column", "")
            if url_column and url_column not in df.columns:
                st.warning(f"Column '{url_column}' not found in the uploaded file. Please check the column names. If there's no URL column, please leave it empty.")
            company_email_col = st.text_input("Company Email Column", "")
            if company_email_col and company_email_col not in df.columns:
                st.warning(f"Column '{company_email_col}' not found in the uploaded file. Please check the column names. If there's no Company Email column, please leave it empty.")
            company_phone_col = st.text_input("Company Phone Column", "")
            if company_phone_col and company_phone_col not in df.columns:
                st.warning(f"Column '{company_phone_col}' not found in the uploaded file. Please check the column names. If there's no Company Phone column, please leave it empty.")
            contact_phone_col = st.text_input("Contact Phone Column", "")
            if contact_phone_col and contact_phone_col not in df.columns:
                st.warning(f"Column '{contact_phone_col}' not found in the uploaded file. Please check the column names. If there's no Contact Phone column, please leave it empty.")
            
        # Refinement options
        st.header("3. Data Refinement Options")
        st.subheader("Data Cleaning")
        st.write("Data cleaning will remove invalid data based on the selected columns.")
        
        clean_email = st.checkbox("Clean Company Email columns")
        clean_phone = st.checkbox("Clean Company Phone columns")
        clean_contact_phone = st.checkbox("Clean Contact Phone column")
        clean_contact_email = st.checkbox("Clean Contact Email column")
        clean_url = st.checkbox("Clean URL column")
        clean_state = st.checkbox("Clean State column")
        state_col = "na"
        address_col = "na"
        if clean_state:
            state_col = st.text_input("State Column", "State")
            if state_col not in df.columns:
                st.error(f"Column '{state_col}' not found in the uploaded file. Please check the column names.")
        clean_address = st.checkbox("Clean Address column")
        if clean_address:
            address_col = st.text_input("Address Column", "Address")
            if address_col not in df.columns:
                st.error(f"Column '{address_col}' not found in the uploaded file. Please check the column names.")
            
        st.divider()
        st.subheader("Data Enrichment")
        st.write("Enriching algorithm will add information scrapped from the company website to the list.")
        phone_email = st.checkbox("Phone and Email Enrichment")
        pms_gateway = st.checkbox("PMS and Gateway Enrichment")


        # Duplicate merging
        st.divider()
        st.subheader("Duplicate Management")
        merge_duplicates = st.checkbox("Manage Duplicates within the list")

        # Duplicate merge with another dataset
        merge_with_another = st.checkbox("Merge dups with Another Dataset")
        if merge_with_another:
            another_file = st.file_uploader("Upload another CSV file to merge dups with", type="csv")
            if another_file is not None:
                df2 = pd.read_csv(another_file)
        
        # Perform cleaning when button is pressed
        
        if st.button("Refine Data"):
            rename_map = {
                    address_col: 'Address',
                    state_col: 'State',
                    company_phone_col: 'Company phone',
                    company_email_col: 'Email',
                    contact_email_col: 'Contact Email',
                    contact_phone_col: 'Contact Phone',
                    url_column: 'url'
            }
            if clean_email or clean_phone or clean_contact_phone or clean_url or clean_state or clean_address or clean_contact_email:
                df = validate_dataframe(df, rename_map, clean_email, clean_phone, clean_contact_phone, clean_contact_email, clean_url, clean_state, clean_address)
            st.success("Data Cleaning Completed.")

            if phone_email or pms_gateway:
                # Perform data enrichment
                st.write("Enriching data, this may take a while depending on the dataset size...")
                progress_bar = st.progress(0)
                if phone_email:
                    if pms_gateway:
                        df = enrich_full_data(df, company_name, url_column,progress_bar, phone_email= True, pms_gateway = True)
                    else:
                        df = enrich_full_data(df, company_name, url_column,progress_bar, phone_email= True)
                else:
                    if pms_gateway:
                        df = enrich_full_data(df, company_name, url_column, progress_bar, pms_gateway = True)
                progress_bar.progress(100)
                st.success("Data Enrichment Completed.")

            if merge_duplicates:
                # Merge duplicates
                df, original_len, merged_len = dups_manage(df, id_column, company_name, contact_name, url_column)
                st.success("Duplicate Management Completed. Original length: {}, After dedup length: {}, Merged amount: {}".format(original_len, merged_len, original_len-merged_len))

            if merge_with_another and another_file is not None:
                # Merge with another dataset
                df2 = pd.read_csv(another_file)
                df = merge_with_another(df, df2, company_name, contact_name, contact_email_col, company_email_col, company_phone_col, contact_phone_col)
                st.success("Merged with another dataset.")
            
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