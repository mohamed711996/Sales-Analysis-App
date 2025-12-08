import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import io

# ============================
#        دوال أساسية
# ============================

def load_and_prepare_data(uploaded_file):
    """قراءة بيانات Excel بأمان وتجهيزها"""
    if uploaded_file is None:
        return None, "يرجى رفع ملف بيانات أولاً."

    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        return None, f"خطأ عند قراءة الملف: {e}"

    # تحقق من الأعمدة المطلوبة
    required_columns = ['Invoice Date', 'Sales Order Number', 'Item Name']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return None, f"الملف ناقص الأعمدة: {', '.join(missing_cols)}"

    # تنظيف البيانات
    df.dropna(subset=['Sales Order Number', 'Item Name'], inplace=True)
    df['Sales Order Number'] = df['Sales Order Number'].astype(str)
    df['Item Name'] = df['Item Name'].str.strip().astype(str)
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
    df.dropna(subset=['Invoice Date'], inplace=True)
    df['YearMonth'] = df['Invoice Date'].dt.to_period('M').astype(str)

    if df.empty:
        return None, "الملف فارغ بعد تنظيف البيانات."

    return df, None

def run_market_basket_analysis(df, target_months, min_support, min_confidence):
    """تحليل سلة المشتريات وإرجاع النتائج"""
    if not target_months:
        return "اختر فترة للتحليل.", "error"

    all_months_option = "كل الشهور (الفترة الكاملة)"
    if all_months_option in target_months:
        analysis_df = df.copy()
    else:
        analysis_df = df[df['YearMonth'].isin(target_months)]

    if analysis_df.empty:
        return f"لا توجد بيانات للفترة: {', '.join(target_months)}", "error"

    transactions = analysis_df.groupby('Sales Order Number')['Item Name'].apply(list).tolist()
    total_transactions = len(transactions)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot_df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(onehot_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return "لم يتم العثور على أنماط شراء متكررة. قلل قيمة الدعم.", "warning"

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return f"لم يتم العثور على قواعد بالثقة المطلوبة ({min_confidence}).", "info"

    # البحث عن أرقام الفواتير لكل قاعدة
    transactions_by_invoice = analysis_df.groupby('Sales Order Number')['Item Name'].apply(set)
    invoice_lists = []
    for _, rule in rules.iterrows():
        itemset_to_find = rule['antecedents'].union(rule['consequents'])
        matching_invoices = transactions_by_invoice[transactions_by_invoice.apply(lambda products: products.issuperset(itemset_to_find))]
        invoice_lists.append(matching_invoices.index.tolist())

    rules['invoice_numbers'] = invoice_lists
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules['invoice_numbers'] = rules['invoice_numbers'].apply(lambda x: ', '.join(x))
    rules['عدد الفواتير'] = (rules['support'] * total_transactions).astype(int)

    results_df = rules[[
        'antecedents', 'consequents', 'عدد الفواتير', 'invoice_numbers', 'support', 'confidence', 'lift'
    ]].rename(columns={
        'antecedents': 'إذا اشترى العميل (الشرط)',
        'consequents': 'فإنه يشتري أيضًا (النتيجة)',
        'support': 'نسبة الدعم',
        'confidence': 'نسبة الثقة',
        'lift': 'مقياس القوة (Lift)',
        'invoice_numbers': 'أرقام الفواتير'
    })

    return results_df, "success"

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='نتائج التحليل')
        worksheet = writer.sheets['نتائج التحليل']
        for idx, col in enumerate(df):
            series = df[col]
            max_len = max((series.astype(str).map(len).max(), len(str(series.name))))
            worksheet.set_column(idx, idx, min(max_len + 2, 60))
    return output.getvalue()

# ============================
#        واجهة المستخدم
# ============================

st.set_page_config(page_title="محلل سلة المشتريات", layout="wide", page_icon="🛒")
st.title("🛒 مكتشف علاقات المنتجات")
st.markdown("ارفع ملف Excel للمبيعات، ثم اختر الفترة الزمنية لاكتشاف جميع المنتجات التي تُشترى معًا.")

uploaded_file = st.file_uploader("1. ارفع ملف بيانات المبيعات (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    df, error_msg = load_and_prepare_data(uploaded_file)
    if error_msg:
        st.error(error_msg)
    else:
        st.success("تم تحميل وتجهيز البيانات بنجاح!")
        all_months_option = "كل الشهور (الفترة الكاملة)"
        options = [all_months_option] + sorted(df['YearMonth'].unique())
        target_months = st.multiselect("اختر الشهور للتحليل:", options=options)

        min_support_val = st.slider("الحد الأدنى للدعم (Support)", 0.001, 0.1, 0.01, 0.001)
        min_confidence_val = st.slider("الحد الأدنى للثقة (Confidence)", 0.1, 1.0, 0.5, 0.05)

        if st.button("🔍 اكتشف علاقات المنتجات"):
            result, status = run_market_basket_analysis(df, target_months, min_support_val, min_confidence_val)
            if status == "success":
                st.success(f"تم العثور على {len(result)} قاعدة من قواعد الارتباط.")
                st.dataframe(result)
                excel_data = to_excel(result)
                st.download_button("📥 تحميل النتائج كملف Excel", data=excel_data, file_name="تحليل_سلة_المشتريات.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            elif status == "info": st.info(result)
            elif status == "warning": st.warning(result)
            else: st.error(result)
else:
    st.info("في انتظار رفع ملف البيانات...")
