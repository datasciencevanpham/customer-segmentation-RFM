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

uploaded_file = st.file_uploader("Choose a file", type=['txt'])
if uploaded_file is not None:
    file = BytesIO(uploaded_file.read())
    df = pd.read_fwf(file)
    df.to_csv("CDNOW_master_new.csv", index=False)

# 2. Data pre-processing
# Đổi tên cột
df.columns = ['customer_ID', 'purchase_date', 'CD_number', 'total_amount']

# Chuyển đổi dtype
df['CD_number'] = df['CD_number'].astype('int32')
df['total_amount'] = df['total_amount'].astype('float32')

string_to_date = lambda x : datetime.strptime(x, "%Y%m%d").date()

# Convert InvoiceDate from object to datetime format
df['purchase_date'] = df['purchase_date'].apply(string_to_date)
df['purchase_date'] = df['purchase_date'].astype('datetime64[ns]')

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Reset the index of the dataframe
df.reset_index(drop=True, inplace=True)

# Drop rows that contain "?" values from the dataframe
df = df[~(df == '?').any(axis=1)]

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

# Reset the index
rfm_agg = rfm_agg.reset_index()



# GUI
menu = ["Business Objectives", "Build Project"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objectives':    
    st.subheader("Business Objectives")
    st.write("""
    ###### RFM stands for recency, frequency, and monetary value. The idea is to segment customers based on when their last purchase was, how often they've purchased in the past, and how much they've spent overall.
    """)  
    st.write("""###### => Problem/ Requirement: Use RFM Analysis in Python for Customer Segmentation.""")
    st.image("RFM1.png")
    st.image("RFM2.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data:")
    st.dataframe(df.head(5))
    st.dataframe(df.tail(5))  
    st.write("##### 2. Visualization:")

    st.write("**Check outliers**")
    # Các cột kiểu numeric
    num_cols = list(df.select_dtypes(['number']).columns)
    # Check outliers ở các biến số
    lst_outliers_del = []
    lst_outliers_keep = []
    for col in num_cols:
        check_outlier_IQR(df=df, name=col, lst_outliers_del=lst_outliers_del, lst_outliers_keep=lst_outliers_keep)

    st.write("**Nhận xét:**\nGiữ lại outliers và chạy RFM Analysis, vì có thể những khách hàng VIP nằm trong khoảng outliers (những khách hàng chi nhiều tiền total_amount cho việc mua đĩa CD mỗi năm - dữ liệu ở bài này là hơn 1 năm). Trong kinh doanh, việc giữ chân các khách hàng VIP nằm trong outliers (upper outliers) là vô cùng quan trọng, vì số tiền khách VIP mua hàng cho công ty đem lại nguồn lợi nhuận lớn, do đó, không xóa upper outliers.")
    
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
    st.write("**Some data in RFM table**")
    st.dataframe(df_RFM.head(5))
    st.dataframe(df_RFM.tail(5))

    st.write("**Descriptive statistics**")
    st.dataframe(df_RFM.describe().T)

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

    st.write("**Nhận xét:**\n* RFM score 4-5 thấp lại chiếm tỷ lệ nhiều nhất (có số lượng khách hàng nhiều nhất, khoảng tổng trên ~ 9000 khách).\n* Tuy nhiên, RFM score 12 điểm tuyệt đối dành cho khách hàng tiềm năng thì chiếm tỷ lệ nhiều thứ 3 (~ 3500 khách).\n* RFM score 9-10-11 xấp xỉ nhau chiếm tỷ lệ thứ 5 (sau score 6, khoảng tổng ~ trên 6000 khách).\nNhư vậy, số lượng khách tiềm năng chiếm khoảng ~ 9500 khách, gần 50% tổng số lượng khách hàng => Tỷ lệ khá tốt.")

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

    st.write("Đa phần Khách hàng mua Tổng số tiền (Monetary) trong khoảng trên 0 đến dưới 2000 USD.")

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

    st.write("**Nhận xét:** Nhìn vào biểu đồ ta có thể thấy:\n* 1st: R=4-F=4 có số lượng khách hàng nhiều nhất và số tiền chi tiêu mỗi khách hàng cao nhất (~ trên 300 USD/ khách).\n* 2nd: Kế đến là, R=3 F=3 có số lượng khách hàng nhiều nhì và số tiền chi tiêu mỗi khách hàng ~ trên 80 USD/ khách.\n* Thấp nhất là R=1 F=4, nghĩa là các khách hàng mua rất lâu rồi nhưng với số lần nhiều.\n* Không có RF: 4-1, 4-2.\n ")
    st.write("**Nhìn vào màu sắc Average Spending và Frequency có thể thấy rằng:**\n* Tần suất F khách mua hàng thấp thì số tiền chi ra thấp (F=1-2 chi khoảng dưới 50 USD/ khách) và mua từ rất lâu rồi (R=1-2).\n* Tần suất F cao thì số tiền chi ra cao (F=3-4 chi khoảng từ 75 USD đến trên 300 USD/ khách) và mới mua gần đây (R=3-4).")

    st.write("**Quartiles' Range**")

    st.button("Reset", type="primary")
    if st.button('Recency'):
        # Tìm range của mỗi quartile
        st.write("Range Recency khi R=1:", "min =", df_RFM['Recency'][df_RFM['R']==1].min(), "-- max =", df_RFM['Recency'][df_RFM['R']==1].max())
        st.write("Range Recency khi R=2:", "min =", df_RFM['Recency'][df_RFM['R']==2].min(), "-- max =", df_RFM['Recency'][df_RFM['R']==2].max())
        st.write("Range Recency khi R=3:", "min =", df_RFM['Recency'][df_RFM['R']==3].min(), "-- max =", df_RFM['Recency'][df_RFM['R']==3].max())
        st.write("Range Recency khi R=4:", "min =", df_RFM['Recency'][df_RFM['R']==4].min(), "-- max =", df_RFM['Recency'][df_RFM['R']==4].max())

    if st.button('Frequency'):
        # Tìm range của mỗi quartile
        st.write("Range Frequency khi F=1:", "min =", df_RFM['Frequency'][df_RFM['F']==1].min(), "-- max =", df_RFM['Frequency'][df_RFM['F']==1].max())
        st.write("Range Frequency khi F=2:", "min =", df_RFM['Frequency'][df_RFM['F']==2].min(), "-- max =", df_RFM['Frequency'][df_RFM['F']==2].max())
        st.write("Range Frequency khi F=3:", "min =", df_RFM['Frequency'][df_RFM['F']==3].min(), "-- max =", df_RFM['Frequency'][df_RFM['F']==3].max())
        st.write("Range Frequency khi F=4:", "min =", df_RFM['Frequency'][df_RFM['F']==4].min(), "-- max =", df_RFM['Frequency'][df_RFM['F']==4].max())
    
    if st.button('Monetary'):
        # Tìm range của mỗi quartile
        st.write("Range Monetary khi M=1:", "min =", df_RFM['Monetary'][df_RFM['M']==1].min(), "-- max =", df_RFM['Monetary'][df_RFM['M']==1].max())
        st.write("Range Monetary khi M=2:", "min =", df_RFM['Monetary'][df_RFM['M']==2].min(), "-- max =", df_RFM['Monetary'][df_RFM['M']==2].max())
        st.write("Range Monetary khi M=3:", "min =", df_RFM['Monetary'][df_RFM['M']==3].min(), "-- max =", df_RFM['Monetary'][df_RFM['M']==3].max())
        st.write("Range Monetary khi M=4:", "min =", df_RFM['Monetary'][df_RFM['M']==4].min(), "-- max =", df_RFM['Monetary'][df_RFM['M']==4].max())

    else:
        st.write('')
        
    st.write("**Kết luận chọn Segmentation:** Có thể thấy, khách hàng càng chi nhiều tiền mua CD, thì tần suất mua hàng càng nhiều, và số ngày cách lần mua hàng gần nhất càng thấp. Do đó, mà có thể chia nhóm khách hàng dựa trên các mức Monetary, để công ty có chiến dịch marketing tiếp thị mua lại phù hợp cho những khách hàng đem lại nguồn doanh thu khủng cho công ty.\nSau khi xem Range của mỗi quartile trong RFM và biểu đồ RFM, thì chọn được 05 nhóm khách như sau:\n* **Nhóm 1:** Khách hàng Diamond (khách hàng hạng Kim Cương) là khách có RFM: R=4, F=4, M=4.\n* **Nhóm 2:** Khách hàng Gold (khách hàng hạng Vàng) là khách có RFM: R=1-4, F=1-4, M=3-4.\n* **Nhóm 3:** Khách hàng Silver (khách hàng hạng Bạc) là khách có RFM: R=1-4, F=1-4, M=2.\n* **Nhóm 4:** Khách hàng Copper (khách hàng hạng Đồng) là khách có RFM: R=1-4, F=1-4, M=1.\n* **Nhóm 5:** Khách hàng Lost (khách hàng có nguy cơ mất) là khách có RFM: R=1, F=1, M=1 (ở tập dữ liệu này không có nhóm 5 RFM=111).")

    st.write("**Some data after RFM analysis**")
    st.dataframe(df_RFM[::2000])
    
    st.dataframe(rfm_agg)

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
    
    st.write(""" Dữ liệu dựa trên Lịch sử mua hàng từ quý đầu tiên năm 1997 cho đến hết quý thứ hai năm 1998 (cuối tháng 06/1998) (khoảng 1 năm 3 tháng) của 23.570 khách hàng đã thực hiện, phân cụm như sau:

    * **Diamond:** Nhóm Khách hàng vừa mua hàng gần đây với tần suất nhiều khoảng 10 lần và tổng chi tiêu trung bình khủng ~ 387 USD/ khách.

    * **Gold:** Nhóm Khách hàng mua hàng với tần suất khoảng 3 lần và khoảng 110 USD/khách.

    * **Silver:** Nhóm Khách hàng mua hàng với tần suất khoảng 1 lần và khoảng 31 USD/khách.

    * **Copper:** Nhóm Khách hàng mua hàng với tần suất khoảng 1 lần và khoảng 13 USD/khách.

    * **Lost:** Nhóm Khách hàng mua hàng từ rất lâu về trước, chi ít hoặc không, tần suất 1 lần (ở tập dữ liệu này không có nhóm Lost).""")

