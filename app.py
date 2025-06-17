import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title('تحليل بيانات المبيعات - Association Rules')

# تحميل البيانات
uploaded_file = st.file_uploader('ارفع ملف الإكسل الخاص بالمبيعات', type=['xlsx'])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write('عرض أول 5 صفوف من البيانات:')
    st.dataframe(df.head())

    # معالجة البيانات
    df.dropna(axis=0, subset=['Order Reference'], inplace=True)
    df['Order Reference'] = df['Order Reference'].astype(str)
    df['product name'] = df['product name'].str.strip()
    df['product name'] = df['product name'].astype(str)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['YearMonth'] = df['Order Date'].dt.to_period('M')

    # إعادة تسمية الأعمدة إذا لزم الأمر
    df.rename(columns={'Order Lines/Order Reference/Order Date':'Order Date',
                       'Order Lines/Order Reference':'Order Reference',
                       'Order Lines/Barcode':'Barcode',
                       'Order Lines/Product/Display Name':'product name',
                       'Order Lines/Product/Product Category':'Product Category',
                       'Order Lines/Product/Brand':'Brand',
                       'Order Lines/Quantity':'Quantity',
                       'Order Lines/Total':'Total'},inplace=True)

    # استبعاد بعض الفئات
    df = df[~df['Product Category'].isin([
        'Books / Educational / El Moasser',
        'Books / Educational',
        'Books / Educational / El Emtehan',
        'Books / Educational / Selah Eltelmeez'
    ])]

    st.write('عدد الصفوف بعد التنظيف:', df.shape[0])

    # استخراج القيم الفريدة لفئة المنتج
    st.write('الفئات المتوفرة:', pd.unique(df['Product Category']))

    # يمكنك إضافة المزيد من التحليلات أو واجهة تفاعلية هنا
else:
    st.info('يرجى رفع ملف بيانات المبيعات (Excel) لبدء التحليل.')
