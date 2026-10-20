import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import io
import warnings

# تجاهل التحذيرات من openpyxl
warnings.simplefilter("ignore", UserWarning)

# ============================
#        دوال أساسية
# ============================

def load_and_prepare_data(uploaded_file):
    """قراءة بيانات Excel من Odoo وتجهيزها واستبعاد المصروفات والعهود"""
    if uploaded_file is None:
        return None, "يرجى رفع ملف بيانات أولاً."

    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        return None, f"خطأ عند قراءة الملف: {e}"

    required_columns = ['Order Reference', 'Order Date', 'Invoices/Invoice lines/Product', 'Invoices/Invoice lines/Quantity']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return None, f"الملف ناقص الأعمدة التالية: {', '.join(missing_cols)}"

    # ملء القيم الفارغة في أرقام الطلبات والتاريخ (لأن Odoo يتركها فارغة للأسطر اللاحقة في نفس الفاتورة)
    df['Order Reference'] = df['Order Reference'].ffill()
    df['Order Date'] = df['Order Date'].ffill()

    # تنظيف البيانات الأساسية
    df.dropna(subset=['Order Reference', 'Invoices/Invoice lines/Product', 'Invoices/Invoice lines/Quantity'], inplace=True)
    df['Order Reference'] = df['Order Reference'].astype(str).str.strip()
    df['Invoices/Invoice lines/Product'] = df['Invoices/Invoice lines/Product'].astype(str).str.strip()
    df['Invoices/Invoice lines/Quantity'] = pd.to_numeric(df['Invoices/Invoice lines/Quantity'], errors='coerce').fillna(0)

    # استبعاد المنتجات الخاصة بـ "employee custody" أو "Shipping fees" (أو أي بنود خدمة/مصروفات مشابهة)
    exclude_keywords = ['employee custody', 'shipping fees', 'shipping', 'custody']
    pattern = '|'.join(exclude_keywords)
    df = df[~df['Invoices/Invoice lines/Product'].str.lower().str.contains(pattern, na=False)].copy()

    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)
    df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)

    # تجميع الكميات لكل منتج داخل نفس الطلب (للتعامل مع المرتجعات والكميات السالبة)
    grouped_qty = df.groupby(['Order Reference', 'Invoices/Invoice lines/Product', 'YearMonth'], as_index=False)['Invoices/Invoice lines/Quantity'].sum()

    # استبعاد المنتجات التي كميتها الصافية <= 0 (مرتجعات / ألغيت)
    valid_df = grouped_qty[grouped_qty['Invoices/Invoice lines/Quantity'] > 0].copy()

    if valid_df.empty:
        return None, "الملف فارغ بعد تصفية المرتجعات والمصروفات وتجهيز البيانات."

    return valid_df, None

def run_market_basket_analysis(df, target_months, min_support, min_confidence, max_len):
    """تحليل سلة المشتريات وإرجاع النتائج بناءً على سلوك العملاء الحقيقي"""
    if not target_months:
        return "اختر فترة للتحليل.", "error"

    all_months_option = "كل الشهور (الفترة الكاملة)"
    if all_months_option in target_months:
        analysis_df = df.copy()
    else:
        analysis_df = df[df['YearMonth'].isin(target_months)]

    if analysis_df.empty:
        return f"لا توجد بيانات للفترة: {', '.join(target_months)}", "error"

    # تجهيز المعاملات (كل Order Reference يمثل سلة مشتريات العميل الفعلية)
    transactions = analysis_df.groupby('Order Reference')['Invoices/Invoice lines/Product'].apply(list).tolist()
    total_transactions = len(transactions)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot_df = pd.DataFrame(te_ary, columns=te.columns_)

    # تشغيل Apriori مع التحكم في عدد المنتجات في البندل (max_len) لاكتشاف العروض بناءً على سلوك الشراء
    frequent_itemsets = apriori(onehot_df, min_support=min_support, use_colnames=True, max_len=max_len)
    if frequent_itemsets.empty:
        return "لم يتم العثور على أنماط شراء متكررة بهذا العدد من المنتجات. جرب خفض الحد الأدنى للدعم (Support) أو زيادة عدد منتجات البندل.", "warning"

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return f"لم يتم العثور على قواعد بالثقة المطلوبة ({min_confidence}). جرب خفض قيمة الثقة أو الدعم.", "info"

    # ترتيب القواعد حسب قوة الارتباط (Lift) ثم الثقة (Confidence) لتقديم أفضل اقتراحات للبندل بناءً على سلوك العملاء
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=False).reset_index(drop=True)

    # البحث عن أرقام الفواتير لكل قاعدة
    transactions_by_invoice = analysis_df.groupby('Order Reference')['Invoices/Invoice lines/Product'].apply(set).to_dict()
    invoice_lists = []
    for _, rule in rules.iterrows():
        itemset_to_find = rule['antecedents'].union(rule['consequents'])
        matching_invoices = [invoice for invoice, items in transactions_by_invoice.items() if items.issuperset(itemset_to_find)]
        invoice_lists.append(matching_invoices)

    rules['invoice_numbers'] = invoice_lists
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules['invoice_numbers'] = rules['invoice_numbers'].apply(lambda x: ', '.join(x))
    rules['عدد الفواتير'] = (rules['support'] * total_transactions).astype(int)

    results_df = rules[[
        'antecedents', 'consequents', 'عدد الفواتير', 'invoice_numbers', 'support', 'confidence', 'lift'
    ]].rename(columns={
        'antecedents': 'إذا اشترى العميل (الشرط / المنتج الأساسي)',
        'consequents': 'فإنه يشتري أيضًا (النتيجة / المقترح للبندل)',
        'support': 'نسبة الدعم (انتشار البند)',
        'confidence': 'نسبة الثقة (احتمالية الشراء معاً)',
        'lift': 'مقياس قوة الارتباط (Lift)',
        'invoice_numbers': 'أرقام الطلبات'
    })

    return results_df, "success"

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='نتائج البندلز المقترحة')
        worksheet = writer.sheets['نتائج البندلز المقترحة']
        for idx, col in enumerate(df):
            series = df[col]
            max_len = max((series.astype(str).map(len).max(), len(str(series.name))))
            worksheet.set_column(idx, idx, min(max_len + 2, 60))
    return output.getvalue()

# ============================
#        واجهة المستخدم
# ============================

st.set_page_config(page_title="مكتشف البندلز وعلاقات المنتجات", layout="wide", page_icon="🛒")
st.title("🛒 مكتشف البندلز الذكي (بناءً على سلوك العملاء)")
st.markdown("ارفع ملف Excel المصدّر من Odoo لاكتشاف أفضل مجموعات المنتجات (Bundles) التي يشتريها العملاء معًا بشكل متكرر، مع استبعاد المصروفات والعهود والمرتجعات تلقائياً.")

uploaded_file = st.file_uploader("1. ارفع ملف بيانات المبيعات (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    df, error_msg = load_and_prepare_data(uploaded_file)
    if error_msg:
        st.error(error_msg)
    else:
        st.success("✅ تم تحميل البيانات وتصفية المرتجعات والبنود غير المرغوبة بنجاح!")

        st.markdown("---")
        st.subheader("⚙️ إعدادات وتخصيص التحليل")

        # اختيار الفترة الزمنية مع شرح توضيحي
        all_months_option = "كل الشهور (الفترة الكاملة)"
        options = [all_months_option] + sorted(df['YearMonth'].unique())
        target_months = st.multiselect(
            "اختر الشهور للتحليل:",
            options=options,
            default=[all_months_option],
            help="حدد الشهور التي تريد دراسة سلوك العملاء خلالها. يمكنك اختيار شهر واحد أو أكثر، أو اختيار الفترة الكاملة."
        )
        st.caption("💡 **فائدة هذا الفلتر:** يساعدك على معرفة ما إذا كانت هناك منتجات معينة ترتبط ببعضها في مواسم أو شهور محددة.")

        # اختيار عدد المنتجات في البندل مع شرح توضيحي
        max_items_in_bundle = st.slider(
            "عدد المنتجات في البندل (Max Bundle Size):",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="يحدد أقصى عدد من المنتجات التي يتم ربطها معاً في العرض (مثلاً: منتجين معاً، أو 3 منتجات معاً)."
        )
        st.caption("💡 **فائدة هذا الفلتر:** التحكم في حجم العبوة أو العرض المشترك (بندل ثنائي، ثلاثي، إلخ) بناءً على استراتيجية التسويق لديك.")

        col1, col2 = st.columns(2)
        with col1:
            min_support_val = st.slider(
                "الحد الأدنى للدعم (Support):",
                0.001, 0.1, 0.005, 0.001,
                help="نسبة تكرار ظهور هذه المنتجات معاً مقارنة بإجمالي الفواتير. كلما قللت الرقم، كلما ظهرت منتجات أكثر ندرة."
            )
            st.caption("💡 يمثل مدى شيوع هذا البندل بين إجمالي عملاء المتجر.")

        with col2:
            min_confidence_val = st.slider(
                "الحد الأدنى للثقة (Confidence):",
                0.1, 1.0, 0.3, 0.05,
                help="مدى اليقين بأن العميل إذا اشترى المنتج الأول، فسوف يشتري معه المنتج الثاني (مثلاً 0.5 تعني 50%)."
            )
            st.caption("💡 يمثل مدى قوة توقع شراء المنتج الثاني عند شراء الأول.")

        st.markdown("---")

        if st.button("🚀 تحليل واقتراح البندلز بناءً على سلوك العملاء", type="primary"):
            with st.spinner("جاري تحليل سلال المشتريات واستخراج أفضل البندلز..."):
                result, status = run_market_basket_analysis(df, target_months, min_support_val, min_confidence_val, max_items_in_bundle)
                if status == "success":
                    st.success(f"🎉 تم العثور على {len(result)} بندل وعلاقة ارتباط ناجحة!")

                    # عرض النتائج في جدول مرتب
                    st.dataframe(result, use_container_width=True)

                    excel_data = to_excel(result)
                    st.download_button(
                        "📥 تحميل تقزارير البندلز المقترحة كملف Excel",
                        data=excel_data,
                        file_name="بندلز_المشتريات_المقترحة.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif status == "info":
                    st.info(result)
                elif status == "warning":
                    st.warning(result)
                else:
                    st.error(result)
else:
    st.info("📂 في انتظار رفع ملف بيانات المبيعات البدء في التحليل...")
