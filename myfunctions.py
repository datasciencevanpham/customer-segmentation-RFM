import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime

import scipy
import scipy.stats

from io import StringIO

import pickle
import streamlit as st
import plotly.express as px
import os

import base64
import time
import uuid
from streamlit_extras.dataframe_explorer import dataframe_explorer

def upload_file(uploaded_file):
    if uploaded_file is not None:
        if 'txt' in uploaded_file.name.split('.') or 'csv' in uploaded_file.name.split('.'):
            # file = BytesIO(uploaded_file.read())
            if 'txt' in uploaded_file.name.split('.'):
                # df = pd.read_fwf(uploaded_file)
                with open('CDNOW_master.txt', 'r') as f:
                    raw_data = f.readlines()
                    data = []
                    for line in raw_data:
                        data.append([l for l in line.strip().split(' ') if l !=''])
                df = pd.DataFrame(data)
                df.iloc[:, 0] = df.iloc[:, 0].astype('str')
                df.iloc[:, 1] = df.iloc[:, 1].astype('str')
                df.to_csv("CDNOW_master_new_prediction.csv", index=False)
            else:
                df = pd.read_csv(uploaded_file, dtype={0: 'str', 1: 'str'}) # giữ lại các số 000.. ở đầu
                # https://stackoverflow.com/questions/48903008/how-to-save-a-csv-from-dataframe-to-keep-zeros-left-in-column-with-numbers
                # df.iloc[:, 0] = df.iloc[:, 0].astype('str')
                # df.iloc[:, 1] = df.iloc[:, 1].astype('str')
                df.to_csv("CDNOW_master_new_prediction.csv", index=False)
            # df.to_csv("CDNOW_master_new.csv", index=False)
        else:
            st.warning("File format not supported. Only support for '.txt' and '.csv' file.")

        return df
    
# xử lý tên cột và dữ liệu hoàn chỉnh
def clean_data(df):
    # Đổi tên cột
    df.columns = ['customer_ID', 'purchase_date', 'CD_number', 'total_amount']

    # Chuyển đổi dtype
    df['CD_number'] = df['CD_number'].astype('int32')
    df['total_amount'] = df['total_amount'].astype('float32')
    df['customer_ID'] = df['customer_ID'].astype('str')
    df['purchase_date'] = df['purchase_date'].astype('str')

    string_to_date = lambda x : datetime.strptime(x, "%Y%m%d").date()

    # Convert InvoiceDate from object to datetime format
    df['purchase_date'] = df['purchase_date'].apply(string_to_date)
    df['purchase_date'] = df['purchase_date'].astype('datetime64[ns]')

    df.dropna(inplace=True)
    # df.drop_duplicates(inplace=True) KO XÓA TRÙNG VÌ CÓ THỂ MUA TRÙNG NHIỀU SẢN PHẨM CÙNG GIÁ TIỀN CÙNG NGÀY

    # Reset the index of the dataframe
    df.reset_index(drop=True, inplace=True)

    # Drop rows that contain "?" values from the dataframe
    df = df[~(df == '?').any(axis=1)]

    # Drop rows that contain "negative" values from the dataframe
    df = df[~(df[['CD_number']] < 0).any(axis=1)]
    df = df[~(df[['total_amount']] < 0).any(axis=1)]
    df = df[~(df[['customer_ID']].astype("float") < 0).any(axis=1)]

    return df


# # 3. Build model
# RFM
# Convert string to date, get max date of dataframe

def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))

# Hàm bên dưới sẽ được bạn viết lại tùy bài toán
def rfm_level(df):
    if (df['RFM_Score'] == 12):
        return 'Diamond'

    elif (df['R'] == 1 and df['F'] == 1 and df['M'] == 1):
        return 'Lost'

    else:
        if df['M'] == 3 or df['M'] == 4:
            return 'Gold'

        elif df['M'] == 2:
            return 'Silver'

        elif df['M'] == 1:
            return 'Copper'

# với merged df
def building_model(df_new_clean, df_old):

    df = pd.concat([df_old, df_new_clean], axis=0)
    df = df.reset_index(drop=True)

    max_date = df['purchase_date'].max().date()

    Recency = lambda x : (max_date - x.max().date()).days
    Frequency  = lambda x: len(x)
    Monetary = lambda x : round(sum(x), 2)

    df_RFM = df.groupby('customer_ID').agg({'purchase_date': Recency,
                                            'customer_ID': Frequency,
                                            'total_amount': Monetary })

    # Rename the columns of Dataframe
    df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
    # Descending Sorting
    df_RFM = df_RFM.sort_values('Monetary', ascending=False)

    # Create labels for Recency, Frequency, Monetary
    r_labels = range(4, 0, -1) # số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
    f_labels = range(1, 5)
    m_labels = range(1, 5)

    # Assign these labels to 4 equal percentile groups
    r_groups = pd.qcut(df_RFM['Recency'].rank(method='first'), q=4, labels=r_labels)

    f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=f_labels)

    m_groups = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=4, labels=m_labels)

    # Create new columns R, F, M
    df_RFM = df_RFM.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)


    df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)

    rfm_count_unique = df_RFM.groupby('RFM_Segment')['RFM_Segment'].nunique()
    # The nunique() function returns the number of all unique values.

    # Calculate RFM_Score
    df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)
    df_RFM.head()

    df_RFM['R'] = df_RFM['R'].astype('int')
    df_RFM['F'] = df_RFM['F'].astype('int')
    df_RFM['M'] = df_RFM['M'].astype('int')

    # Create a new column RFM_Level
    df_RFM['RFM_Level'] = df_RFM.apply(rfm_level, axis=1)

    return df_RFM

def descriptives_stas(df_RFM):
    # Calculate average values for each RFM_Level, and return a size of each segment
    rfm_agg = df_RFM.groupby('RFM_Level').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)
    rfm_agg.sort_values(by="MonetaryMean", inplace=True, ascending=False)

    # Reset the index
    rfm_agg = rfm_agg.reset_index()

    return rfm_agg

def download_link_create(df_merged2):
    # Chọn cột để xuất ra file csv - download (không dùng cho Input cuối)
        # # list of available columns to export
        # all_columns2 = df_merged2.columns
        # available_columns2 = ['All columns'] + all_columns2.tolist()

        # # let user select columns to export
        # columns_to_export2 = st.multiselect("Select columns to export", available_columns2)

        # # use the selected columns, or all columns if "All columns" is selected
        # if 'All columns' in columns_to_export2:
        #     filtered_df2 = df_merged2.copy()
        #     columns_to_export2 = all_columns2
        # else:
        #     filtered_df2 = df_merged2.loc[:, columns_to_export2]

        # convert filtered dataframe to CSV and encode as base64
        csv2 = df_merged2.to_csv(index=False).encode()
        b64_csv2 = base64.b64encode(csv2).decode()

        # create a link to download the CSV file
        href_csv2 = f'<a href="data:text/csv;base64,{b64_csv2}" download="exported_data.csv">Download CSV File</a>'
        st.markdown(href_csv2, unsafe_allow_html=True)

        # convert filtered dataframe to text and encode as base64
        txt2 = df_merged2.to_string(index=False).encode()
        b64_txt2 = base64.b64encode(txt2).decode()

        # create a link to download the text file
        href_txt2 = f'<a href="data:text/plain;base64,{b64_txt2}" download="exported_data.txt">Download Text File</a>'
        st.markdown(href_txt2, unsafe_allow_html=True)

        return