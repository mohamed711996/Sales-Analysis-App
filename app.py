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
    تقبل ملفات Excel و CSV.
    """
    if uploaded_file is None:
        return None, "يرجى رفع ملف بيانات أولاً."

    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            return None, f"نوع الملف '{file_extension}' غير مدعوم. يرجى استخدام CSV أو XLSX."
            
    except Exception as e:
        return None, f"حدث خطأ أثناء قراءة الملف: {e}"

    required_columns = ['Order Reference', 'product name', 'Order Date']
    if not all(col in df.columns for col in required_columns):
        return None, f"الملف يجب أن يحتوي على الأعمدة التالية: {', '.join(required_columns)}"

    df.dropna(axis=0, subset=['Order Reference', 'product name'], inplace=True)
    df['Order Reference'] = df['Order Reference'].astype(str)
    df['product name'] = df['product name'].str.strip().astype(str)
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)
    df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
    
    return df, None

# !!! تم تعديل هذه الدالة لإضافة أرقام الفواتير
def run_market_basket_analysis(df, target_months, min_support, min_confidence):
    """
    تحليل سلة المشتريات مع إضافة أرقام الفواتير لكل قاعدة.
    """
    if not target_months:
        return "يرجى اختيار شهر واحد على الأقل أو 'كل الشهور' للتحليل.", "error"

    all_months_option = "كل الشهور (الفترة الكاملة)"
    if all_months_option in target_months:
        analysis_df = df.copy()
    else:
        analysis_df = df[df['YearMonth'].isin(target_months)]
    
    if analysis_df.empty:
        return f"لا توجد بيانات مبيعات للفترة المحددة: {', '.join(target_months)}", "error"
    
    transactions = analysis_df.groupby('Order Reference')['product name'].apply(list).tolist()
    total_transactions = len(transactions)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot_df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(onehot_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty: 
        return "لم يتم العثور على أنماط شراء متكررة. حاول تقليل قيمة 'الحد الأدنى للدعم'.", "warning"
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty: 
        return f"لم يتم العثور على قواعد ربط قوية بالمعايير المحددة (Confidence > {min_confidence:.0%}). حاول تقليل 'الحد الأدنى للثقة'.", "info"

    # !!! بداية الجزء الجديد: البحث عن أرقام الفواتير
    st.info("جاري البحث عن أرقام الفواتير المطابقة لكل قاعدة...") # رسالة للمستخدم
    
    # تجهيز بيانات الفواتير للبحث السريع
    transactions_by_invoice = analysis_df.groupby('Order Reference')['product name'].apply(set)

    invoice_lists = []
    for _, rule in rules.iterrows():
        # دمج الشرط والنتيجة في مجموعة واحدة
        itemset_to_find = rule['antecedents'].union(rule['consequents'])
        
        # البحث عن الفواتير التي تحتوي على كل عناصر المجموعة
        # .issuperset() تتحقق مما إذا كانت مجموعة منتجات الفاتورة تحتوي على كل عناصر القاعدة
        matching_invoices = transactions_by_invoice[transactions_by_invoice.apply(lambda products: products.issuperset(itemset_to_find))]
        
        invoice_lists.append(matching_invoices.index.tolist())
    
    # إضافة القائمة كعمود جديد
    rules['invoice_numbers'] = invoice_lists
    # !!! نهاية الجزء الجديد

    # تجهيز النتائج النهائية للعرض
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    # تحويل قائمة الفواتير إلى نص للعرض بشكل أفضل
    rules['invoice_numbers'] = rules['invoice_numbers'].apply(lambda x: ', '.join(x))

    results_df = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)

    # إعادة تسمية الأعمدة للغة العربية
    results_df.rename(columns={
        'antecedents': 'إذا اشترى العميل (الشرط)',
        'consequents': 'فإنه يشتري أيضًا (النتيجة)',
        'support': 'نسبة الدعم',
        'confidence': 'نسبة الثقة',
        'lift': 'مقياس القوة (Lift)',
        'invoice_numbers': 'أرقام الفواتير' # <-- العمود الجديد
    }, inplace=True)
    
    results_df['عدد الفواتير (الشرط والنتيجة معًا)'] = (results_df['نسبة الدعم'] * total_transactions).astype(int)

    # إعادة ترتيب الأعمدة لتكون أكثر منطقية
    final_columns_order = [
        'إذا اشترى العميل (الشرط)',
        'فإنه يشتري أيضًا (النتيجة)',
        'عدد الفواتير (الشرط والنتيجة معًا)', 
        'أرقام الفواتير', # <-- وضع العمود الجديد هنا
        'نسبة الدعم',
        'نسبة الثقة',
        'مقياس القوة (Lift)'
    ]
    # اختيار الأعمدة النهائية بالترتيب المطلوب
    results_df = results_df[final_columns_order]

    return results_df, "success"

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='نتائج التحليل')
        worksheet = writer.sheets['نتائج التحليل']
        for idx, col in enumerate(df):
            series = df[col]
            # تعديل بسيط للتعامل مع الأعمدة الطويلة جدًا مثل أرقام الفواتير
            max_len = max((series.astype(str).map(len).max(), len(str(series.name))))
            # وضع حد أقصى لعرض العمود حتى لا يصبح الملف ضخمًا جدًا
            worksheet.set_column(idx, idx, min(max_len + 2, 60)) 
    return output.getvalue()


# ===================================================================
#                      واجهة المستخدم (Streamlit UI)
# ===================================================================

st.set_page_config(page_title="محلل سلة المشتريات", layout="wide", page_icon="🛒")
st.title("🛒 مكتشف علاقات المنتجات")
st.markdown("ارفع ملف المبيعات، ثم اختر الفترة الزمنية لاكتشاف جميع المنتجات التي تُشترى معًا.")

uploaded_file = st.file_uploader(
    "1. ارفع ملف بيانات المبيعات (CSV أو Excel)", 
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    df, error_msg = load_and_prepare_data(uploaded_file)
    if error_msg:
        st.error(error_msg)
    else:
        st.success("تم تحميل وتجهيز البيانات بنجاح!")
        st.header("2. اختر مدخلات التحليل")
        
        available_months = sorted(df['YearMonth'].unique())
        all_months_option = "كل الشهور (الفترة الكاملة)"
        options = [all_months_option] + available_months
        
        target_months = st.multiselect(
            "اختر الشهور للتحليل (يمكن اختيار أكثر من شهر أو الفترة الكاملة):", 
            options=options,
            help="اختر 'كل الشهور' لتحليل كامل البيانات، أو اختر الشهور التي تريدها."
        )

        with st.expander("إعدادات التحليل (متقدم)"):
            col1, col2 = st.columns(2)
            with col1: 
                min_support_val = st.slider("الحد الأدنى للدعم (Support)", 0.001, 0.1, 0.01, 0.001, format="%.3f", help="يمثل نسبة تكرار ظهور المنتج. قيمة أصغر للبيانات الكبيرة.")
            with col2: 
                min_confidence_val = st.slider("الحد الأدنى للثقة (Confidence)", 0.1, 1.0, 0.5, 0.05, format="%.2f", help="يمثل مدى قوة العلاقة (إذا اشترى A، فما هي احتمالية شراء B؟)")

        if target_months:
            if st.button("🔍 اكتشف علاقات المنتجات", type="primary"):
                # استخدام spinner لعرض رسائل متعددة للمستخدم
                with st.spinner("جاري تحليل الأنماط المتكررة..."):
                    result, status = run_market_basket_analysis(df, target_months, min_support_val, min_confidence_val)
                
                period_str = "الفترة الكاملة" if all_months_option in target_months else ", ".join(target_months)
                st.header(f"3. نتائج تحليل الفترة: {period_str}")

                if status == "success":
                    st.success(f"تم العثور على {len(result)} قاعدة من قواعد الارتباط القوية.")
                    # عرض النتائج في جدول يمكن التحكم فيه
                    st.dataframe(result)
                    
                    excel_data = to_excel(result)
                    file_name_period = "الفترة_الكاملة" if all_months_option in target_months else "_".join(target_months)
                    st.download_button(
                        label="📥 تحميل النتائج كملف Excel", 
                        data=excel_data, 
                        file_name=f"تحليل_سلة_المشتريات_{file_name_period}.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif status == "info": st.info(result)
                elif status == "warning": st.warning(result)
                else: st.error(result)
        else:
            st.info("يرجى اختيار فترة التحليل من القائمة أعلاه للبدء.")

else:
    st.info("في انتظار رفع ملف البيانات...")
