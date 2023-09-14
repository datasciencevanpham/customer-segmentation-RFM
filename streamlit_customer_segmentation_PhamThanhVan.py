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

# import pickle
import streamlit as st
import plotly.express as px
# import os

import base64
import time
import uuid
from streamlit_extras.dataframe_explorer import dataframe_explorer

from myfunctions import *

import pdfkit
import tempfile

# https://docs.streamlit.io/library/changelog?highlight=SessionState#version-0-54-0
# import SessionState

# import streamlit_option_menu
# https://github.com/TRGanesh/Customer-Segmentation-Clustering-Analysis/blob/main/customer_segmentation_app.py
# https://trganesh-customer-segmentation-customer-segmentation-app-mklfwl.streamlit.app/


# 1. Read data

with open('CDNOW_master.txt', 'r') as f:
    raw_data = f.readlines()
    data = []
    for line in raw_data:
        data.append([l for l in line.strip().split(' ') if l !=''])
df = pd.DataFrame(data)

#--------------
# GUI
st.title("Data Science - Project 1")
st.write("## Customer Segmentation")
# Upload file

from io import BytesIO
from myfunctions import upload_file

st.write("You can change the dataset in 'Build Project' by uploading your new dataset below.")
uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'],  key="uploaded_file")

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
            df.to_csv("CDNOW_master_new.csv", index=False)
            # st.write("There are {} rows and {} columns in the dataset.".format(df.shape[0], df.shape[1]))
        else:
            df = pd.read_csv(uploaded_file, dtype={0: 'str', 1: 'str'}) # giữ lại các số 000.. ở đầu
            # https://stackoverflow.com/questions/48903008/how-to-save-a-csv-from-dataframe-to-keep-zeros-left-in-column-with-numbers
            # df.iloc[:, 0] = df.iloc[:, 0].astype('str')
            # df.iloc[:, 1] = df.iloc[:, 1].astype('str')
            df.to_csv("CDNOW_master_new.csv", index=False)
            # st.write("There are {} rows and {} columns in the dataset.".format(df.shape[0], df.shape[1]))
     
    else:
        st.warning("File format not supported. Only support for '.txt' and '.csv' file.")


st.write("There are {} rows and {} columns in the dataset.".format(df.shape[0], df.shape[1]))
st.write("(Tất cả nhận xét là tự động, bạn có thể xem report mới trong 'Build Project' khi thêm dataset mới tương tự vào 'Build Project'.)")

# 2. Data pre-processing
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

# Giả sử file upload lên đúng kiểu dữ liệu csv, txt và không có lỗi về cột date, vì nếu excel đưa vào csv sẽ bị lỗi date toàn bộ
# nên lấy file CDmaster cũ và đổi tên để test upload
# Chuyển datetime sai thành NaT (Not a Time), ở bài này không có nên tính sau, giả sử không có NaT
## hàm này sẽ force giá trị theo ý của nó, cần check lại sau
# df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')

# Visualization - Outliers:

# Khởi tạo hàm kiểm tra tỷ lệ % của outliers trong feature.
# round in pandas pyspark: https://spark.apache.org/docs/3.2.1/api/python/reference/pyspark.pandas/api/pyspark.pandas.DataFrame.round.html

def check_outlier_IQR(df, name, lst_outliers_del, lst_outliers_keep):
    st.write("Kiểm tra outliers của feature {}:\n".format(name))
    fig, ax = plt.subplots()
    sns.boxplot(x=df[name], ax=ax)
    st.pyplot(fig)

    Q1 = np.percentile(df[name], 25)
    Q3 = np.percentile(df[name], 75)
    IQR = scipy.stats.iqr(df[name])
    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR
    indexes_outliers = df.index[(df[name] < lower) | (df[name] > upper)].tolist()

    n_outlier_upper = df[df[name] > (Q3 + 1.5*IQR)].shape[0]
    st.write("Số lượng upper outliers:", n_outlier_upper)
    n_outlier_lower = df[df[name] < (Q1 - 1.5*IQR)].shape[0]
    st.write("Số lượng lower outliers:", n_outlier_lower)

    outlier_per = (((n_outlier_upper+n_outlier_lower)/df.shape[0])*100)
    st.write("Tỷ lệ outliers chiếm {}% dữ liệu của feature {}.".format(outlier_per, name))

    if (outlier_per < 3) & (outlier_per > 0):
        st.write("--TH1--: Tỷ lệ outliers < 3%.\n=> Thay thế outliers bằng upper whisker và lower whisker, hoặc xóa đi.")
        lst_outliers_del.append(name)
        st.write("---------\n")

    if outlier_per >= 3:
        st.write("--TH2--: Tỷ lệ outliers >= 3%.\n=> Giữ lại outliers và cân nhắc xem outliers có nằm trong khoảng công ty cho phép hay không.")
        lst_outliers_keep.append(name)
        st.write("---------\n")

    if outlier_per == 0:
        st.write("=> Feature không có outliers.\n---------\n")




# # 3. Build model
# RFM
# Convert string to date, get max date of dataframe
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

def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)

rfm_count_unique = df_RFM.groupby('RFM_Segment')['RFM_Segment'].nunique()
# The nunique() function returns the number of all unique values.

# Calculate RFM_Score
df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)
df_RFM.head()

df_RFM['R'] = df_RFM['R'].astype('int')
df_RFM['F'] = df_RFM['F'].astype('int')
df_RFM['M'] = df_RFM['M'].astype('int')

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

# Create a new column RFM_Level
df_RFM['RFM_Level'] = df_RFM.apply(rfm_level, axis=1)

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



# GUI
menu = ["Business Objectives", "Build Project", 'New Predictions']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objectives':    
    st.subheader("Business Objectives")
    st.write("""
    ###### RFM stands for recency, frequency, and monetary value. The idea is to segment customers based on when their last purchase was, how often they've purchased in the past, and how much they've spent overall.
    """)  
    st.write("""###### => Problem/ Requirement: Use RFM Analysis in Python for Customer Segmentation.""")
    st.write("The file 'CDNOW master.txt' contains the entire purchase history up to the end of June 1998 of the cohort of 23,570 individuals who made their first-ever purchase at CDNOW in the first quarter of 1997. (See Fader and Hardie (2001a) for further details of this dataset.) Each record in this file, 69,659 in total, comprises four fields: the customer’s ID, the date of the transaction, the number of CDs purchased, and the dollar value of the transaction.")
   
    st.image("RFM1.png")
    st.image("RFM2.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("**Upload new dataset if you need a new report**")
    
    st.write("##### 1. Dataset:")
    # st.dataframe(df.head(5))
    # st.dataframe(df.tail(5))
    st.dataframe(dataframe_explorer(df),use_container_width=True)  

    st.write("##### 2. Exploratory data analysis:")

    st.write("**Check outliers**")
    # Các cột kiểu numeric
    num_cols = list(df.select_dtypes(['number']).columns)
    # Check outliers ở các biến số
    lst_outliers_del = []
    lst_outliers_keep = []
    for col in num_cols:
        check_outlier_IQR(df=df, name=col, lst_outliers_del=lst_outliers_del, lst_outliers_keep=lst_outliers_keep)

    st.write("**Nhận xét:**\n* Giữ lại outliers và chạy RFM Analysis, vì có thể những khách hàng VIP nằm trong khoảng outliers (những khách hàng chi nhiều tiền total_amount cho việc mua đĩa CD mỗi năm - dữ liệu ở bài này là hơn 1 năm). Trong kinh doanh, việc giữ chân các khách hàng VIP nằm trong outliers (upper outliers) là vô cùng quan trọng, vì số tiền khách VIP mua hàng cho công ty đem lại nguồn lợi nhuận lớn, do đó, không xóa outliers.")
    st.write("* Dữ liệu đã được xử lý số âm, nên không có outliers nằm trong số âm. Giữ lại outliers để chạy RFM Analysis.")
    st.write("**Some information**")
    # Show transaction timeframe
    st.write("* Transactions timeframe from {} to {}.".format(df['purchase_date'].min(), df['purchase_date'].max()))

    # Display number of transactions without a customer ID
    n_missing_customer_id = df[df.customer_ID.isnull()].shape[0]
    st.write("* {:,} transactions don't have a customer ID.".format(n_missing_customer_id))

    # Display number of unique customer IDs
    n_unique_customer_id = len(df.customer_ID.unique())
    st.write("* {:,} unique customer_ID.".format(n_unique_customer_id))

    n_total_amount_zero = df[df['total_amount']==0].shape[0]
    st.write("* There are {:,} rows where total_amount is equal to 0.".format(n_total_amount_zero))


    st.write("##### 3. Build model:") 
    st.write("(Create RFM analysis for each customers - Manual Segmentation)")
    st.write("**Dataset in RFM table**")
    # st.dataframe(df_RFM.head(5))
    # st.dataframe(df_RFM.tail(5))
    st.dataframe(dataframe_explorer(df_RFM),use_container_width=True)

    st.write("**Descriptive statistics**")
    st.dataframe(df_RFM.describe().T)
    st.dataframe(rfm_agg)

    st.write("**Plot distribution of RFM**")
    # disable warning
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(8,10))
    plt.subplot(3, 1, 1)
    sns.distplot(df_RFM['Recency'])# Plot distribution of R
    plt.subplot(3, 1, 2)
    sns.distplot(df_RFM['Frequency'])# Plot distribution of F
    plt.subplot(3, 1, 3)
    sns.distplot(df_RFM['Monetary']) # Plot distribution of M
    st.pyplot()  

    st.write("**Plot RFM Scores**")
    arr = np.array(df_RFM['RFM_Score'])

    labels, counts = np.unique(arr, return_counts=True)
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.xlabel("RFM_Score")
    plt.ylabel("Number of customers")
    plt.title("RFM Score with number of customers")
    plt.grid(True)
    st.pyplot() 

    # labels, counts = np.unique(df_RFM['RFM_Score'], return_counts=True)
    df_RFMscore = pd.DataFrame({'label':labels, 'count':counts})
    df_RFMscore.sort_values(by="count", ascending=False, inplace=True)
    df_RFMscore['rank'] = df_RFMscore['count'].rank(ascending=False).astype(int)
    df_RFMscore_list = df_RFMscore.loc[df_RFMscore['label'].isin([9,10,11]), 'rank'].tolist()
    first_item = df_RFMscore_list[0]
    second_item = df_RFMscore_list[1]
    third_item = df_RFMscore_list[2]

    st.write("**Nhận xét:**\n* RFM score", df_RFMscore.loc[df_RFMscore['count'] == df_RFMscore['count'].max(), 'label'].values[0], "chiếm tỷ lệ nhiều nhất (có số lượng khách hàng nhiều nhất, khoảng ~", df_RFMscore['count'].max(),"khách).")
    st.write("* RFM score 12 điểm tuyệt đối dành cho khách hàng tiềm năng thì chiếm tỷ lệ nhiều thứ", df_RFMscore.loc[df_RFMscore['label']==12, 'rank'].values[0],"(~", df_RFMscore.loc[df_RFMscore['label']==12, 'count'].values[0] ,"khách).")
    st.write("* RFM score 9-10-11 xấp xỉ nhau chiếm tỷ lệ rank lần lượt là", str(first_item) + ",", str(second_item) + ",", str(third_item) + ",","(khoảng tổng ~", sum(df_RFMscore.loc[df_RFMscore['label'].isin([9,10,11]), 'count'].tolist()),"khách).")
    st.write("Như vậy, số lượng khách tiềm năng khoảng ~", sum(df_RFMscore.loc[df_RFMscore['label'].isin([9,10,11,12]), 'count'].tolist()),"khách, chiếm tỷ lệ khoảng ~", round(sum(df_RFMscore.loc[df_RFMscore['label'].isin([9,10,11,12]), 'count'].tolist())/sum(df_RFMscore['count'])*100,2),"% tổng số lượng khách hàng.")

    st.write("**Plot Recency-Frequency-Monetary**")

    # create scatter plot
    fig1, ax = plt.subplots()
    scat = ax.scatter(df_RFM['Recency'], df_RFM['Frequency'], c=df_RFM['Monetary'], cmap=plt.cm.rainbow)

    # add colorbar to explain the color scheme
    plt.colorbar(scat, label='Monetary')

    # set title and labels
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_title('Recency-Frequency-Monetary Scatter Plot')

    # set grid lines
    ax.grid(True)

    # show plot in Streamlit
    st.pyplot(fig1)

    st.write("Đa phần Khách hàng mua Tổng số tiền (Monetary) trong khoảng trên 0 đến ", round(df_RFM['Monetary'].median(),2) ,"USD.")

    st.write("**Plot R-F-M**")

    # compute the counts of observations
    df_counts = df_RFM.groupby(['R', 'F']).size().reset_index()
    df_counts.columns.values[df_counts.columns == 0] = 'count'

    # compute a size variable for the markers
    scale = 500*df_counts['count'].size
    size = df_counts['count']/df_counts['count'].sum()*scale

    # add Monetary to df_counts and compute Average Monetary per customer
    df_counts_Monetary = df_RFM.groupby(['R', 'F'])['Monetary'].mean().reset_index()
    df_counts['Monetary'] = df_counts_Monetary['Monetary']

    # create scatter plot
    fig, ax = plt.subplots()
    scat = ax.scatter('R', 'F', s=size, data=df_counts, c='Monetary', cmap=plt.cm.rainbow)

    # add colorbar to explain the color scheme
    plt.colorbar(scat, label='Average Spending per customer')

    # set title and labels
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_title('RFM Scatter Plot (with number of customers as marker size)')

    # set grid lines
    ax.grid(True)

    # show plot in Streamlit
    st.pyplot(fig)

    Rrank1 = df_counts.loc[df_counts['count'] == df_counts['count'].max(), 'R'].values[0]
    Frank1 = df_counts.loc[df_counts['count'] == df_counts['count'].max(), 'F'].values[0]
    Mrank1 = df_counts.loc[df_counts['count'] == df_counts['count'].max(), 'Monetary'].values[0]
    df_counts_second_max = df_counts.sort_values(by='count', ascending=False)['count'].iloc[1]  # Tìm giá trị của count lớn thứ hai
    Rrank2 = df_counts.loc[df_counts['count'] == df_counts_second_max, 'R'].values[0]  # Tìm giá trị của cột 'R' ở hàng có giá trị của count bằng df_counts_second_max
    Frank2 = df_counts.loc[df_counts['count'] == df_counts_second_max, 'F'].values[0] 
    Mrank2 = df_counts.loc[df_counts['count'] == df_counts_second_max, 'Monetary'].values[0]

    Rranklast = df_counts.loc[df_counts['count'] == df_counts['count'].min(), 'R'].values[0]
    Franklast = df_counts.loc[df_counts['count'] == df_counts['count'].min(), 'F'].values[0]
    Mranklast = df_counts.loc[df_counts['count'] == df_counts['count'].min(), 'Monetary'].values[0]
    df_count_0 = df_counts.loc[df_counts['count'] == 0, ['R', 'F']].values

    MrankF1 = round(df_counts.loc[df_counts['F'] == 1, 'Monetary'].mean(),1)
    MrankF2 = round(df_counts.loc[df_counts['F'] == 2, 'Monetary'].mean(),1)
    MrankF3 = round(df_counts.loc[df_counts['F'] == 3, 'Monetary'].mean(),1)
    MrankF4 = round(df_counts.loc[df_counts['F'] == 4, 'Monetary'].mean(),1)

    st.write("**Nhận xét:** Nhìn vào biểu đồ ta có thể thấy:")
    st.write("* 1st: R={}-F={} có số lượng khách hàng nhiều nhất và số tiền chi tiêu mỗi khách hàng khoảng ~ {:.1f} USD/ khách).".format(Rrank1, Frank1, Mrank1))
    st.write("* 2nd: R={}-F={} có số lượng khách hàng nhiều nhì và số tiền chi tiêu mỗi khách hàng khoảng ~ {:.1f} USD/ khách.".format(Rrank2, Frank2, Mrank2))
    st.write("* R={}-F={} có số lượng khách hàng thấp nhất và số tiền chi tiêu mỗi khách hàng khoảng ~ {:.1f} USD/ khách.".format(Rranklast, Franklast, Mranklast))
    
    st.write("**Nhìn vào màu sắc Average Spending và Frequency có thể thấy rằng:**")
    st.write("* Tần suất F=1-2 khách mua hàng thấp thì số tiền chi ra thấp (chi khoảng {} USD ~ {} USD/ khách) và mua từ rất lâu rồi (R=1-2).".format(MrankF1, MrankF2))
    st.write("* Tần suất F=3-4 khách mua hàng cao thì số tiền chi ra cao (chi khoảng {} USD ~ {} USD/ khách) và mới mua gần đây (R=3-4).".format(MrankF3, MrankF4))

    st.write("**Quartiles' Range**")

    # st.button("Reset", type="primary")
    # if st.button('Recency'):
    # Tìm range của mỗi quartile
    
    ### quartiles' range
    Rm1 = df_RFM['Recency'][df_RFM['R']==1].min()
    Rm2 = df_RFM['Recency'][df_RFM['R']==2].min()
    Rm3 = df_RFM['Recency'][df_RFM['R']==3].min()
    Rm4 = df_RFM['Recency'][df_RFM['R']==4].min()
    Rmax1 = df_RFM['Recency'][df_RFM['R']==1].max()
    Rmax2 = df_RFM['Recency'][df_RFM['R']==2].max()
    Rmax3 = df_RFM['Recency'][df_RFM['R']==3].max()
    Rmax4 = df_RFM['Recency'][df_RFM['R']==4].max()

    # st.write("Range Recency khi R=1:", "min =", Rm1, "-- max =", Rmax1)
    # st.write("Range Recency khi R=2:", "min =", Rm2, "-- max =", Rmax2)
    # st.write("Range Recency khi R=3:", "min =", Rm3, "-- max =", Rmax3)
    # st.write("Range Recency khi R=4:", "min =", Rm4, "-- max =", Rmax4)
    dataR = {'R=1': [Rm1, Rmax1], 'R=2': [Rm2, Rmax2], 'R=3': [Rm3, Rmax3], 'R=4': [Rm4, Rmax4]}
    dfR = pd.DataFrame(dataR)
    dfR.index = ['Recency Min', 'Recency Max']
    st.dataframe(dfR)

    # if st.button('Frequency'):
    # Tìm range của mỗi quartile
    dataF = {'F=1': [df_RFM['Frequency'][df_RFM['F']==1].min(), df_RFM['Frequency'][df_RFM['F']==1].max()], 
             'F=2': [df_RFM['Frequency'][df_RFM['F']==2].min(), df_RFM['Frequency'][df_RFM['F']==2].max()], 
             'F=3': [df_RFM['Frequency'][df_RFM['F']==3].min(), df_RFM['Frequency'][df_RFM['F']==3].max()], 
             'F=4': [df_RFM['Frequency'][df_RFM['F']==4].min(), df_RFM['Frequency'][df_RFM['F']==4].max()]}
    dfF = pd.DataFrame(dataF)
    dfF.index = ['Frequency Min', 'Frequency Max']
    st.dataframe(dfF)
    
    # if st.button('Monetary'):
    # Tìm range của mỗi quartile
    dataM = {'M=1': [df_RFM['Monetary'][df_RFM['M']==1].min(), df_RFM['Monetary'][df_RFM['M']==1].max()], 
             'M=2': [df_RFM['Monetary'][df_RFM['M']==2].min(), df_RFM['Monetary'][df_RFM['M']==2].max()], 
             'M=3': [df_RFM['Monetary'][df_RFM['M']==3].min(), df_RFM['Monetary'][df_RFM['M']==3].max()], 
             'M=4': [df_RFM['Monetary'][df_RFM['M']==4].min(), df_RFM['Monetary'][df_RFM['M']==4].max()]}
    dfM = pd.DataFrame(dataM)
    dfM.index = ['Monetary Min', 'Monetary Max']
    st.dataframe(dfM)

    # else:
    #     st.write('')
        
    st.write("""**Kết luận chọn Segmentation:**\nCó thể thấy, khách hàng càng chi nhiều tiền mua CD, thì tần suất mua hàng càng nhiều, và số ngày cách lần mua hàng gần nhất càng thấp. Do đó, mà có thể chia nhóm khách hàng dựa trên các mức Monetary, để công ty có chiến dịch marketing tiếp thị mua lại phù hợp cho những khách hàng đem lại nguồn doanh thu khủng cho công ty.\nSau khi xem Range của mỗi quartile trong RFM và biểu đồ RFM, thì chọn được 05 nhóm khách như sau:\n* **Nhóm 1:** Khách hàng Diamond (khách hàng hạng Kim Cương) là khách có RFM: R=4, F=4, M=4.\n* **Nhóm 2:** Khách hàng Gold (khách hàng hạng Vàng) là khách có RFM: R=1-4, F=1-4, M=3-4.\n* **Nhóm 3:** Khách hàng Silver (khách hàng hạng Bạc) là khách có RFM: R=1-4, F=1-4, M=2.\n* **Nhóm 4:** Khách hàng Copper (khách hàng hạng Đồng) là khách có RFM: R=1-4, F=1-4, M=1.\n* **Nhóm 5:** Khách hàng Lost (khách hàng có nguy cơ mất) là khách có RFM: R=1, F=1, M=1.""")

    # st.write("**Some data after RFM analysis**")
    # st.dataframe(df_RFM[::2000])
    
    

    st.write("##### 4. Evaluation:")

    st.write("**Tree Map**")
    
    # assign colors to customer segments
    colors_dict = {'Diamond':'royalblue','Gold':'yellow', 'Silver':'cyan', 'Copper':'purple'}

    # create plot
    fig_2 = plt.figure()
    ax2 = fig_2.add_subplot()
    fig_2.set_size_inches(14, 10)

    # plot treemap using squarify library
    squarify.plot(sizes=rfm_agg['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                        for i in range(0, len(rfm_agg))], alpha=0.5 )

    # set plot title and axis
    plt.title("Customers Segments",fontsize=26,fontweight="bold")
    plt.axis('off')

    # display plot in Streamlit
    st.pyplot(fig_2)

    
    st.write("**RFM Level Scatter Plot**")
    # create scatter plot
    fig3 = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_Level",
                    hover_name="RFM_Level", size_max=100)

    # display plot in Streamlit
    st.plotly_chart(fig3)

    st.write("**Nhận xét:**\n* Tỷ lệ phân cụm khá tốt, không bị chênh lệch quá nhiều.\n* Phân cụm hợp lý: Khách hàng có Monetary càng cao, thì Frequency càng cao và Recency (số ngày cách lần mua hàng gần nhất) càng thấp.")

    st.write("##### 5. Summary:")
    
    st.write("Dữ liệu dựa trên Lịch sử mua hàng từ {} cho đến hết {} của {} khách hàng đã thực hiện, phân cụm như sau:".format(df['purchase_date'].min().date(), df['purchase_date'].max().date(), n_unique_customer_id))

    st.write("* **Diamond:** Nhóm Khách hàng vừa mua hàng gần đây với tần suất nhiều khoảng {} lần và tổng chi tiêu trung bình khủng ~ {} USD/ khách.".format(rfm_agg.loc[rfm_agg['RFM_Level']=="Diamond", 'FrequencyMean'].values[0], rfm_agg.loc[rfm_agg['RFM_Level']=="Diamond", 'MonetaryMean'].values[0]))

    st.write("* **Gold:** Nhóm Khách hàng mua hàng với tần suất khoảng {} lần và khoảng {} USD/khách.".format(rfm_agg.loc[rfm_agg['RFM_Level']=="Gold", 'FrequencyMean'].values[0], rfm_agg.loc[rfm_agg['RFM_Level']=="Gold", 'MonetaryMean'].values[0]))

    st.write("* **Silver:** Nhóm Khách hàng mua hàng với tần suất khoảng {} lần và khoảng {} USD/khách.".format(rfm_agg.loc[rfm_agg['RFM_Level']=="Silver", 'FrequencyMean'].values[0], rfm_agg.loc[rfm_agg['RFM_Level']=="Silver", 'MonetaryMean'].values[0]))

    st.write("* **Copper:** Nhóm Khách hàng mua hàng với tần suất khoảng {} lần và khoảng {} USD/khách.".format(rfm_agg.loc[rfm_agg['RFM_Level']=="Copper", 'FrequencyMean'].values[0], rfm_agg.loc[rfm_agg['RFM_Level']=="Copper", 'MonetaryMean'].values[0]))

    st.write("* **Lost:** Nhóm Khách hàng mua hàng từ rất lâu về trước, chi rất ít hoặc không, tần suất 1 lần.")

    # vì tạo dữ liệu mới dựa trên file cũ CDmaster nên chỉ cần question nhóm Lost có hay không?
    RFMLEVEL_unique = rfm_agg['RFM_Level'].tolist()
    if "Lost" in RFMLEVEL_unique:
        st.write("* Ở tập dữ liệu này có nhóm **Lost**.")
    else:
        st.write("* Ở tập dữ liệu này không có nhóm **Lost**.")

    st.write("**Download RFM table:**")
    # Reset the index and rename the column
    df_RFM_build_project = df_RFM.reset_index().rename(columns={'index': 'customer_ID'})
    df_RFM_build_project = df_RFM_build_project.sort_values(by="customer_ID", ascending=True)

    download_link_create(df_RFM_build_project)


elif choice == 'New Predictions':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        st.write("**Upload new file to append to the dataset in 'Build Project'.**")
        
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'], key="uploaded_file_1")
               
        if uploaded_file_1 is not None:

            uploaded_file_1 = upload_file(uploaded_file_1)
            st.write("There are {} rows and {} columns.".format(uploaded_file_1.shape[0], uploaded_file_1.shape[1]))

            st.write("#### Your sub-dataset:")
            st.dataframe(dataframe_explorer(uploaded_file_1),use_container_width=True) 
            st.write("#### Your new predictions:")

            st.write("##### 1. RFM table with unique customer_ID (Old dataset + Your sub-dataset)")
            st.write("Old dataset is the uploaded dataset in 'Build Project'")
            
            # file new chuẩn chỉnh
            df_new_clean = clean_data(uploaded_file_1)

            # thêm df_new_clean vào df ở buid project để phân cụm khách hàng dựa trên các R,F,M sẵn có
            df_merged_RFM = building_model(df_new_clean, df)
            ######
            df_merged = df_merged_RFM.sort_values(by="customer_ID", ascending=True)

            # Reset the index and rename the column
            df_merged = df_merged.reset_index().rename(columns={'index': 'customer_ID'})
            st.write("There are", df_merged.shape[0], "unique customer_ID.")
            st.dataframe(dataframe_explorer(df_merged),use_container_width=True)

            st.write("Chọn cột để download.")
            # Chọn cột để xuất ra file csv - download
            # list of available columns to export
            all_columns = df_merged.columns
            available_columns = ['All columns'] + all_columns.tolist()

            # let user select columns to export
            columns_to_export = st.multiselect("Select columns to export", available_columns)

            # use the selected columns, or all columns if "All columns" is selected
            if 'All columns' in columns_to_export:
                filtered_df = df_merged.copy()
                columns_to_export = all_columns
            else:
                filtered_df = df_merged.loc[:, columns_to_export]

            # convert filtered dataframe to CSV and encode as base64
            csv = filtered_df.to_csv(index=False).encode()
            b64_csv = base64.b64encode(csv).decode()

            # create a link to download the CSV file
            href_csv = f'<a href="data:text/csv;base64,{b64_csv}" download="exported_data.csv">Download CSV File</a>'
            st.markdown(href_csv, unsafe_allow_html=True)

            # convert filtered dataframe to text and encode as base64
            txt = filtered_df.to_string(index=False).encode()
            b64_txt = base64.b64encode(txt).decode()

            # create a link to download the text file
            href_txt = f'<a href="data:text/plain;base64,{b64_txt}" download="exported_data.txt">Download Text File</a>'
            st.markdown(href_txt, unsafe_allow_html=True)

            #####
            st.write("##### 2. Merge RFM_Level with your dataset (only predictions for your sub-dataset)")
            st.write("Lưu ý: Chỉ xử lý xóa dữ liệu NA/null ở file mới upload.")

            df_sorted = df_merged_RFM.sort_values(by="customer_ID", ascending=True)

            # Reset the index and rename the column
            df_sorted = df_sorted.reset_index().rename(columns={'index': 'customer_ID'})

            # merge df_merged2 with df_new_clean on 'customer_ID'
            # lấy prediction
            df_merged2 = pd.merge(df_new_clean, df_sorted[['customer_ID', 'RFM_Level']], on='customer_ID', how='left')
            st.write("There are", df_merged2.shape[0], "transactions.")
            st.dataframe(dataframe_explorer(df_merged2),use_container_width=True)

            st.write("Chọn cột để download.")
            # Chọn cột để xuất ra file csv - download
            # list of available columns to export
            all_columns2 = df_merged2.columns
            available_columns2 = ['All columns'] + all_columns2.tolist()

            # let user select columns to export
            columns_to_export2 = st.multiselect("Select columns to export", available_columns2)

            # use the selected columns, or all columns if "All columns" is selected
            if 'All columns' in columns_to_export2:
                filtered_df2 = df_merged2.copy()
                columns_to_export2 = all_columns2
            else:
                filtered_df2 = df_merged2.loc[:, columns_to_export2]

            # convert filtered dataframe to CSV and encode as base64
            csv2 = filtered_df2.to_csv(index=False).encode()
            b64_csv2 = base64.b64encode(csv2).decode()

            # create a link to download the CSV file
            href_csv2 = f'<a href="data:text/csv;base64,{b64_csv2}" download="exported_data.csv">Download CSV File</a>'
            st.markdown(href_csv2, unsafe_allow_html=True)

            # convert filtered dataframe to text and encode as base64
            txt2 = filtered_df2.to_string(index=False).encode()
            b64_txt2 = base64.b64encode(txt2).decode()

            # create a link to download the text file
            href_txt2 = f'<a href="data:text/plain;base64,{b64_txt2}" download="exported_data.txt">Download Text File</a>'
            st.markdown(href_txt2, unsafe_allow_html=True)
              
    if type=="Input":        

        # Tạo một empty dataframe để lưu dữ liệu
        # https://docs.streamlit.io/library/advanced-features/dataframes
        data2 = pd.DataFrame(columns=['customer_ID', 'purchase_date', 'CD_number', 'total_amount'])
        config = {
                    'customer_ID' : st.column_config.TextColumn(width='large', required=True),
                    'purchase_date' : st.column_config.TextColumn(width='large', required=True),
                    'CD_number' : st.column_config.NumberColumn(min_value=0),
                    'total_amount' : st.column_config.NumberColumn(min_value=0)
                }
        st.write("You can open widely to add rows quickly (click on Pop-up from the arrow in the upper right of the table).")
        st.write("* 'customer_ID' : number >= 0\n* 'purchased_date' : '%Y%m%d'\n* 'CD_number' : number >= 0\n* 'total_amount' : number >= 0")
        st.write()
        edited_df = st.data_editor(data2, num_rows="dynamic")
        # file new chuẩn chỉnh
        edited_df = pd.DataFrame(edited_df)
        edited_df.iloc[:, 0] = edited_df.iloc[:, 0].astype('str')
        edited_df.iloc[:, 1] = edited_df.iloc[:, 1].astype('str')
        st.write("There are", edited_df.shape[0], "transactions.")
        

        if st.button('Get Results'):

            df_new_clean_input = clean_data(edited_df)

            # thêm df_new_clean_input vào df ở buid project để phân cụm khách hàng dựa trên các R,F,M sẵn có
            df_merged_RFM_input = building_model(df_new_clean_input, df)

            df_sorted_input = df_merged_RFM_input.sort_values(by="customer_ID", ascending=True)

            # Reset the index and rename the column
            df_sorted_input = df_sorted_input.reset_index().rename(columns={'index': 'customer_ID'})

            # merge df_merged2 with df_new_clean on 'customer_ID'
            # lấy prediction
            df_merged2_input = pd.merge(df_new_clean_input, df_sorted_input[['customer_ID', 'RFM_Level']], on='customer_ID', how='left')
                        
            # Xuất ra kết quả dataframe đã nhập
            st.write("*RFM table*")
            st.dataframe(dataframe_explorer(df_sorted_input), use_container_width=True)
            st.write("**Download predictions**")
            st.dataframe(dataframe_explorer(df_merged2_input), use_container_width=True)

            download_link_create(df_merged2_input)