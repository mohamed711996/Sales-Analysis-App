import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import io

# ===================================================================
#               الدوال الأساسية (المنطق الخلفي)
# ===================================================================

@st.cache_data 
def load_and_prepare_data(uploaded_file):
    """
    تقوم هذه الدالة بقراءة البيانات المرفوعة وتجهيزها.
    !!! تم تعديلها لتقبل ملفات Excel و CSV
    """
    if uploaded_file is None:
        return None, "يرجى رفع ملف بيانات أولاً."

    try:
        # التحقق من نوع الملف واستخدام الدالة المناسبة
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            return None, f"نوع الملف '{file_extension}' غير مدعوم. يرجى استخدام CSV أو XLSX."
            
    except Exception as e:
        return None, f"حدث خطأ أثناء قراءة الملف: {e}"

    # التحقق من وجود الأعمدة المطلوبة
    required_columns = ['Order Reference', 'product name', 'Order Date']
    if not all(col in df.columns for col in required_columns):
        return None, f"الملف يجب أن يحتوي على الأعمدة التالية: {', '.join(required_columns)}"

    # ... باقي الكود يبقى كما هو ...
    df.dropna(axis=0, subset=['Order Reference', 'product name'], inplace=True)
    df['Order Reference'] = df['Order Reference'].astype(str)
    df['product name'] = df['product name'].str.strip().astype(str)
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)
    df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
    
    return df, None

# ... باقي دوال التحليل تبقى كما هي بدون تغيير ...
def run_market_basket_analysis(df, target_month, min_support, min_confidence):
    monthly_df = df[df['YearMonth'] == target_month]
    if monthly_df.empty: return f"لا توجد بيانات مبيعات للشهر المحدد: {target_month}", "error"
    transactions = monthly_df.groupby('Order Reference')['product name'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot_df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(onehot_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty: return "لم يتم العثور على أنماط شراء متكررة. حاول تقليل قيمة 'الحد الأدنى للدعم'.", "warning"
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty: return f"لم يتم العثور على قواعد ربط قوية بالمعايير المحددة (Confidence > {min_confidence:.0%}). حاول تقليل 'الحد الأدنى للثقة'.", "info"
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    results_df = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    results_df = results_df.sort_values(by='lift', ascending=False).reset_index(drop=True)
    results_df.rename(columns={'antecedents': 'إذا اشترى العميل (الشرط)','consequents': 'فإنه يشتري أيضًا (النتيجة)','support': 'نسبة الدعم','confidence': 'نسبة الثقة','lift': 'مقياس القوة (Lift)'}, inplace=True)
    return results_df, "success"

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='نتائج التحليل')
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column))
            writer.sheets['نتائج التحليل'].set_column(df.columns.get_loc(column), df.columns.get_loc(column), column_width + 2)
    return output.getvalue()


# ===================================================================
#                      واجهة المستخدم (Streamlit UI)
# ===================================================================

st.set_page_config(page_title="محلل سلة المشتريات", layout="wide", page_icon="🛒")
st.title("🛒 مكتشف علاقات المنتجات")
st.markdown("ارفع ملف المبيعات، ثم اختر الشهر لاكتشاف جميع المنتجات التي تُشترى معًا.")

# !!! هذا هو السطر الذي تم تعديله
uploaded_file = st.file_uploader(
    "1. ارفع ملف بيانات المبيعات (CSV أو Excel)", 
    type=["csv", "xlsx"]  # <-- أضفنا "xlsx" هنا
)

# ... باقي واجهة المستخدم تبقى كما هي ...
if uploaded_file is not None:
    df, error_msg = load_and_prepare_data(uploaded_file)
    if error_msg: st.error(error_msg)
    else:
        st.success("تم تحميل وتجهيز البيانات بنجاح!")
        st.header("2. اختر مدخلات التحليل")
        available_months = sorted(df['YearMonth'].unique())
        target_month = st.selectbox("اختر الشهر للتحليل:", options=available_months)
        with st.expander("إعدادات التحليل (متقدم)"):
            col1, col2 = st.columns(2)
            with col1: min_support_val = st.slider("الحد الأدنى للدعم (Support)", 0.001, 0.1, 0.01, 0.001, format="%.3f", help="يمثل نسبة تكرار ظهور المنتج. قيمة أصغر للبيانات الكبيرة.")
            with col2: min_confidence_val = st.slider("الحد الأدنى للثقة (Confidence)", 0.1, 1.0, 0.5, 0.05, format="%.2f", help="يمثل مدى قوة العلاقة (إذا اشترى A، فما هي احتمالية شراء B؟)")
        if st.button("🔍 اكتشف علاقات المنتجات", type="primary"):
            with st.spinner("جاري التحليل..."):
                result, status = run_market_basket_analysis(df, target_month, min_support_val, min_confidence_val)
            st.header(f"3. نتائج تحليل شهر: {target_month}")
            if status == "success":
                st.success(f"تم العثور على {len(result)} قاعدة من قواعد الارتباط القوية.")
                st.dataframe(result)
                excel_data = to_excel(result)
                st.download_button(label="📥 تحميل النتائج كملف Excel", data=excel_data, file_name=f"تحليل_سلة_المشتريات_{target_month}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else: st.info(result) if status == "info" else (st.warning(result) if status == "warning" else st.error(result))
else:
    st.info("في انتظار رفع ملف البيانات...")
