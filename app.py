# -*- coding: utf-8 -*-
# v2.5.8: Final fixes including pricing ceiling rounding, chart rewrite/cleanup, and welcome screen redesign.
# v2.5.9: Implemented session state to prevent rerun loops on file load errors.
# v2.5.9-mobile-logo-center-v2: Centered logo in sidebar using base64 data URI and HTML/CSS.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties # 中文字体
import io
import os
import numpy as np
from datetime import datetime, timedelta
import pytz # Using pytz for timezone handling
import streamlit.components.v1 as components
import traceback # For detailed error logging if needed
import math # Added for ceiling rounding
import base64 # Added for image encoding
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from openai import OpenAI
import requests # Added for API calls
import time # Added for retry logic



# --- Constants ---
TOP_N_DISPLAY = 150 # Max rows to display in tables (performance)
APP_TIMEZONE_STR = 'Europe/Athens' # Define standard timezone for calculations
API_MAX_RETRIES = 3 # Max retries for API calls
API_TIMEOUT = 10 # Timeout in seconds for API calls
LOW_STOCK_THRESHOLD = 10 # Threshold for low stock warning
INVENTORY_API_URL = "https://api.example.com/inventory" # Default inventory API URL

# Use current date for versioning
try:
    # Attempt to get current time in the specified timezone for versioning
    APP_VERSION = f"v2.5.9-{datetime.now(pytz.timezone(APP_TIMEZONE_STR)).strftime('%Y%m%d')}"
except Exception:
    # Fallback if timezone setup fails early
    APP_VERSION = "v2.5.9-unknown_date"


# --- Timezone Setup ---
try:
    APP_TIMEZONE = pytz.timezone(APP_TIMEZONE_STR)
except pytz.exceptions.UnknownTimeZoneError:
    st.warning(f"无法识别的时区 '{APP_TIMEZONE_STR}'，将回退到 UTC。")
    APP_TIMEZONE = pytz.utc # Fallback
    APP_TIMEZONE_STR = 'UTC'

# --- Update version string again now that timezone is confirmed ---
APP_VERSION = f"v2.5.9-{datetime.now(APP_TIMEZONE).strftime('%Y%m%d')}"

# --- Matplotlib 中文显示设置 ---
# Try to find the font in common locations or the script directory
font_paths_to_check = [
    'SimHei.ttf', # Check local directory first
    '/usr/share/fonts/truetype/simhei/SimHei.ttf', # Linux common path
    '/Library/Fonts/SimHei.ttf', # macOS common path
    'C:/Windows/Fonts/simhei.ttf' # Windows common path
]
font_path = None
for p in font_paths_to_check:
    if os.path.exists(p):
        font_path = p
        break

FONT_AVAILABLE = False
chinese_font = None
if font_path:
    try:
        chinese_font = FontProperties(fname=font_path, size=10)
        FONT_AVAILABLE = True
        print(f"成功加载字体: {font_path}") # Log success
    except Exception as font_err:
        print(f"加载字体文件 '{font_path}' 时出错: {font_err}") # Log error to console
else:
    print(f"警告：在常见位置及脚本目录未找到中文字体文件 (如 SimHei.ttf)。") # Log warning

# Ensure negative signs display correctly even with CJK fonts
plt.rcParams['axes.unicode_minus'] = False

# --- Initialize Session State (For Preventing Rerun Loops) ---
if 'main_load_error' not in st.session_state:
    st.session_state.main_load_error = None
if 'last_main_file_id' not in st.session_state:
    st.session_state.last_main_file_id = None


# ======== Helper Functions ========

# --- Function to encode image to base64 ---
def get_image_as_base64(path):
    """Reads an image file and returns its base64 encoded data URI."""
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        # Basic image type detection from extension
        image_type = os.path.splitext(path)[1].lower().replace('.', '')
        # Add more types if needed, provide fallback
        supported_types = ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp']
        if image_type not in supported_types:
            image_type = 'png' # Default fallback if extension is unknown/unsupported
        return f"data:image/{image_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Logo file not found at {path}")
        return None
    except Exception as e:
        print(f"Error encoding image {path}: {e}")
        return None

# --- AI Response Generation Function ---
@st.cache_data(ttl=timedelta(minutes=10), show_spinner=False)
def generate_ai_response(prompt: str, analysis_data: dict) -> str:
    """生成AI分析回复(仅回答用户问题，不重复展示完整分析)"""
    # 获取用户选择的API
    api_provider = st.session_state.get("api_provider", "deepseek")
    
    # 预加载客户端减少延迟
    if api_provider == "deepseek":
        from deepseek_api import DeepSeekClient
        try:
            client = DeepSeekClient(api_key=st.session_state.get("deepseek_api_key") or st.secrets.get("DEEPSEEK_API_KEY"))
        except:
            client = DeepSeekClient()  # 使用内置密钥
    else:
        client = OpenAI(api_key=st.session_state.get("openai_api_key") or st.secrets["OPENAI_API_KEY"])
    
    # 专业系统提示
    system_prompt = """你是希腊雅典百货城专业批发市场分析师，精通中文、希腊语、英语。
    专长领域:
    - 希腊及欧洲批发市场行情
    - 零售百货趋势分析
    - 职业经理人视角
    
    当前分析数据:
    洞察: {insights}
    风险: {alerts}
    建议: {recommendations}
    
    工作要求:
    1. 使用欧元(€)计算(汇率:1€=1.08USD)
    2. 根据用户语言自动切换响应语言
    3. 结合希腊市场特点分析
    4. 提供专业批发市场见解
    5. 回答简明扼要"""
    
    messages = [
        {"role": "system", "content": system_prompt.format(
            insights=", ".join(str(i) for i in analysis_data.get("top_insights", [])[:3]),
            alerts=", ".join(str(a) for a in analysis_data.get("risk_alerts", [])[:3]),
            recommendations=", ".join(str(r) for r in analysis_data.get("recommendations", [])[:3])
        )},
        {"role": "user", "content": prompt}
    ]
    
    # 优化API调用参数
    params = {
        "model": "deepseek-chat" if api_provider == "deepseek" else "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 800  # 减少token数量
    }
    
    # 仅OpenAI支持stream参数
    if api_provider != "deepseek":
        params["stream"] = False
    
    response = client.chat.completions.create(**params)
    
    return response.choices[0].message.content

# --- load_data 函数 (For Main Analysis) ---
@st.cache_data(ttl=timedelta(minutes=10))
def load_data(uploaded_file_content, uploaded_file_name):
    """
    Loads data from uploaded file content (Excel or CSV) for main analysis.
    Returns (sales_df, stock_df, purchase_df, error_message)
    On success, error_message is None.
    On failure, dataframes are None and error_message contains the error string.
    """
    try:
        file_ext = os.path.splitext(uploaded_file_name)[1].lower()
        file_buffer = io.BytesIO(uploaded_file_content)
        sales_df, stock_df, purchase_df = None, None, None

        if file_ext == '.csv':
            df = pd.read_csv(file_buffer)
            if 'DataType' not in df.columns: raise ValueError("CSV文件必须包含 'DataType' 列以区分 '订单数据', '库存数据', '采购数据'.")
            sales_df = df[df['DataType'] == '订单数据'].copy()
            stock_df = df[df['DataType'] == '库存数据'].copy()
            purchase_df = df[df['DataType'] == '采购数据'].copy()
            if sales_df.empty: raise ValueError("在CSV中未找到 DataType 为 '订单数据' 的数据.")
            # Handle potentially empty but present stock/purchase data in CSV
            if stock_df.empty and '库存数据' in df['DataType'].unique():
                st.warning("警告：CSV中DataType为'库存数据'的部分为空。将使用空结构。")
                stock_df = pd.DataFrame(columns=["产品ID", "当前库存", "产品名称", "采购价", "产品分类"])
            elif not stock_df.empty and '库存数据' not in df['DataType'].unique():
                 # Should not happen if logic above is right, but as safety
                 st.warning("警告：发现库存数据，但未标记为 '库存数据' DataType。")

            if purchase_df.empty and '采购数据' in df['DataType'].unique():
                st.caption("注意：CSV中DataType为'采购数据'的部分为空。将使用空结构。")
                purchase_df = pd.DataFrame(columns=["采购日期", "产品ID", "采购数量", "产品分类"])
            elif not purchase_df.empty and '采购数据' not in df['DataType'].unique():
                 st.warning("警告：发现采购数据，但未标记为 '采购数据' DataType。")

        elif file_ext in ['.xlsx', '.xls']:
            try:
                xls = pd.ExcelFile(file_buffer)
                required_sheets = ["订单数据", "库存数据"]
                if not all(sheet in xls.sheet_names for sheet in required_sheets):
                    raise ValueError(f"Excel文件必须至少包含以下工作表: {', '.join(required_sheets)}")

                sales_df = pd.read_excel(xls, sheet_name="订单数据")
                stock_df = pd.read_excel(xls, sheet_name="库存数据")

                if "采购数据" in xls.sheet_names:
                    purchase_df = pd.read_excel(xls, sheet_name="采购数据")
                    if purchase_df.empty: st.caption("注意：'采购数据' 工作表为空。将使用空结构。")
                else:
                    st.warning("警告：Excel文件中未找到 '采购数据' 工作表。采购相关分析将受限。将使用空结构。")
                    purchase_df = pd.DataFrame(columns=["采购日期", "产品ID", "采购数量", "产品分类"]) # Ensure it exists even if sheet missing

            except Exception as e:
                raise ValueError(f"读取Excel文件结构或内容时出错: {e}")
        else:
            raise ValueError("不支持的文件类型。请上传 .csv, .xlsx, 或 .xls 文件。")

        # Ensure dataframes are not None before proceeding
        if sales_df is None: sales_df = pd.DataFrame() # Should have been caught earlier
        if stock_df is None: stock_df = pd.DataFrame(columns=["产品ID", "当前库存", "产品名称", "采购价", "产品分类"])
        if purchase_df is None: purchase_df = pd.DataFrame(columns=["采购日期", "产品ID", "采购数量", "产品分类"])


        # --- Date Conversions ---
        date_cols_map = {'sales': (sales_df, '订单日期'), 'purchase': (purchase_df, '采购日期')}
        for key, (df_loop, date_col) in date_cols_map.items():
             if isinstance(df_loop, pd.DataFrame) and not df_loop.empty and date_col in df_loop.columns:
                 original_count = len(df_loop)
                 # Coerce errors first, then handle NaT if needed, then convert valid to datetime
                 df_loop[date_col] = pd.to_datetime(df_loop[date_col], errors='coerce')
                 failed_count = df_loop[date_col].isna().sum() # Count NaNs/NaTs directly
                 if failed_count > 0:
                     st.warning(f"警告：在 '{key}' 数据的 '{date_col}' 列中发现 {failed_count} 个无效日期格式，这些值已被置空。后续分析可能忽略这些行。")
                     df_loop.dropna(subset=[date_col], inplace=True) # Drop rows where date conversion failed

        # --- Column Validation ---
        # 添加 "销售额" 到必需列
        required_sales_cols = ["订单日期", "产品ID", "销售数量", "产品名称", "销售额"]
        required_stock_cols = ["产品ID", "当前库存", "产品名称"] # Keep '采购价' optional here, validate numeric later
        required_purchase_cols = ["采购日期", "产品ID", "采购数量"] # Optional '产品分类'和'销售数量'

        if not all(col in sales_df.columns for col in required_sales_cols):
            raise ValueError(f"'订单数据' 缺少必需列: {', '.join([c for c in required_sales_cols if c not in sales_df.columns])}")
        if not stock_df.empty and not all(col in stock_df.columns for col in required_stock_cols):
             raise ValueError(f"'库存数据' 缺少必需列: {', '.join([c for c in required_stock_cols if c not in stock_df.columns])}")
        # Only validate purchase columns if the purchase_df is not empty
        if purchase_df is not None and not purchase_df.empty and not all(col in purchase_df.columns for col in required_purchase_cols):
            raise ValueError(f"'采购数据' 工作表存在但缺少必需列: {', '.join([c for c in required_purchase_cols if c not in purchase_df.columns])}")

        # --- Numeric Conversion ---
        num_cols_map = {
            # 添加 "销售额" 到需要转换的数值列
            'sales': (sales_df, ["销售数量", "销售额"]),
            'stock': (stock_df, ["当前库存", "采购价"]), #采购价 optional but convert if present
            'purchase': (purchase_df, ["采购数量"])
        }

        return sales_df, stock_df, purchase_df, None
    except Exception as e:
        error_msg = f"加载数据时出错: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return None, None, None, error_msg

# --- 库存API接口功能 ---
def get_inventory_data():
    """从API获取库存数据"""
    for attempt in range(API_MAX_RETRIES):
        try:
            response = requests.get(
                INVENTORY_API_URL,
                timeout=API_TIMEOUT,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            # 验证返回数据格式
            data = response.json()
            if not isinstance(data, list):
                raise ValueError("API返回数据格式不正确，应为列表")
                
            return pd.DataFrame(data)
            
        except requests.exceptions.RequestException as e:
            if attempt == API_MAX_RETRIES - 1:  # 最后一次尝试失败
                st.error(f"获取库存数据失败: {str(e)}")
                return pd.DataFrame()
            time.sleep(1)  # 等待1秒后重试
            
        except ValueError as e:
            st.error(f"数据格式错误: {str(e)}")
            return pd.DataFrame()
            
    return pd.DataFrame()  # 默认返回空DataFrame

def show_inventory_warning():
    """显示库存预警"""
    st.header("📊 库存预警")
    inventory_df = get_inventory_data()
    if not inventory_df.empty:
        low_stock = inventory_df[inventory_df['quantity'] < LOW_STOCK_THRESHOLD]
        if not low_stock.empty:
            st.warning(f"⚠️ 有 {len(low_stock)} 个商品库存低于阈值 {LOW_STOCK_THRESHOLD}")
            st.dataframe(low_stock[['product_name', 'quantity']])
        else:
            st.success("✅ 所有商品库存充足")
    else:
        st.info("ℹ️ 未获取到库存数据")

# --- 数值转换处理 ---
def process_numeric_data(num_cols_map):
    """处理数值列转换"""
    for key, (df_loop, cols) in num_cols_map.items():
        if isinstance(df_loop, pd.DataFrame) and not df_loop.empty:
            for col in cols:
                if col in df_loop.columns: # Check if column exists before conversion
                    # Store original dtype to check if conversion actually happens
                    # original_dtype = df_loop[col].dtype
                    initial_na_count = df_loop[col].isna().sum()
                    # Force conversion to numeric, coercing errors to NaN
                    df_loop[col] = pd.to_numeric(df_loop[col], errors='coerce')
                    final_na_count = df_loop[col].isna().sum()
                    newly_failed_count = final_na_count - initial_na_count
                    if newly_failed_count > 0:
                        st.warning(f"警告：在 '{key}' 数据的 '{col}' 列中发现 {newly_failed_count} 个无法解析的非数值，已替换为空值 (NaN)。")
                    # Fill NaN resulting from coercion or original NaNs with 0
                    df_loop[col] = df_loop[col].fillna(0)
                    # Attempt integer conversion only if feasible
                    try:
                        # Check if all non-zero values are whole numbers after filling NaNs
                        non_zero_mask = df_loop[col] != 0
                        if not non_zero_mask.any() or (df_loop.loc[non_zero_mask, col] % 1 == 0).all():
                            # Convert to largest integer type to avoid overflow
                            df_loop[col] = df_loop[col].astype(np.int64)
                        # Otherwise, keep as float if there are decimals
                    except Exception:
                        # If int conversion fails for any reason, keep as float
                        pass
                elif col in ["采购价"] and key == 'stock':
                    # If optional '采购价' is missing in stock, add it as 0 float
                    df_loop[col] = 0.0
                    st.caption(f"注意：库存数据中未找到 '{col}' 列，将假设其值为 0。")

    return num_cols_map
# --- calculate_metrics 函数 ---
def calculate_metrics(sales_df, stock_df, purchase_df, start_date, end_date, dead_stock_threshold_days=90, dead_stock_sales_threshold=0.1, ai_model_version="deepseek-v3"):
    """Calculates key metrics including dead stock detection and AI analysis.
    Args:
        ai_model_version: AI模型版本，默认为deepseek-v3
    Args:
        dead_stock_threshold_days: Number of days without sales to consider as dead stock
        dead_stock_sales_threshold: Daily sales threshold below which stock is considered slow-moving
    Returns:
        metrics_results: Dictionary containing calculated metrics
        stock_analysis: DataFrame with detailed stock analysis
        has_category_in_analysis: Boolean indicating if category data exists
        ai_analysis: Dictionary containing AI-generated insights and recommendations
    """
    try:
        # Ensure dates are Timestamps for comparison
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        now_ts = pd.Timestamp.now(tz=APP_TIMEZONE).tz_localize(None)
    except Exception as date_err:
        st.error(f"无法解析日期范围: {date_err}")
        return {}, pd.DataFrame(), False # Return empty results

    metrics_results = {}
    stock_analysis = pd.DataFrame()
    has_category_in_analysis = False
    # 初始化毛利相关指标
    metrics_results["total_gross_profit_period"] = 0
    metrics_results["overall_gross_margin_period"] = 0.0

    try:
        # --- Sales Metrics ---
        if not isinstance(sales_df, pd.DataFrame) or '订单日期' not in sales_df.columns or not pd.api.types.is_datetime64_any_dtype(sales_df['订单日期']):
            st.error("计算指标错误：'订单数据'无效或缺少正确的'订单日期'列。")
            return metrics_results, stock_analysis, has_category_in_analysis # Return empty

        # Filter sales data for the period
        # Ensure end_ts includes the whole day if it doesn't have time component
        end_ts_inclusive = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1) if end_ts.time() == datetime.min.time() else end_ts
        sales_filtered = sales_df[(sales_df["订单日期"] >= start_ts) & (sales_df["订单日期"] <= end_ts_inclusive)].copy()

        # Calculate total sales quantity
        metrics_results["total_sales_period"] = 0
        if '销售数量' in sales_filtered.columns:
            # Ensure the column is numeric before summing
            # 同时处理销售额
            sales_filtered['销售数量_num_calc'] = pd.to_numeric(sales_filtered['销售数量'], errors='coerce').fillna(0)
            sales_filtered['销售额_num_calc'] = pd.to_numeric(sales_filtered['销售额'], errors='coerce').fillna(0)
            metrics_results["total_sales_period"] = int(sales_filtered["销售数量_num_calc"].sum())
            metrics_results["total_revenue_period"] = float(sales_filtered['销售额_num_calc'].sum()) # 计算总销售额
        else:
             st.warning("销售数据缺少 '销售数量' 或 '销售额' 列，无法计算完整销售指标。")
        # Calculate average daily sales
        num_days_period = max(1, (end_ts - start_ts).days + 1) # Add 1 to include both start and end date
        metrics_results["avg_daily_sales_period"] = round((metrics_results["total_sales_period"] / num_days_period), 1)

        # Calculate active selling days
        metrics_results["active_days_period"] = sales_filtered["订单日期"].nunique() if not sales_filtered.empty else 0

        # Find top selling product
        top_product_period = "无"
        top_selling_data = pd.Series(dtype=float)
        if '产品名称' in sales_filtered.columns and '销售数量_num_calc' in sales_filtered.columns and not sales_filtered.empty:
            try:
                # Group by product name and sum the numeric quantity
                top_selling_data = sales_filtered.groupby("产品名称")["销售数量_num_calc"].sum().sort_values(ascending=False)
                if not top_selling_data.empty:
                    top_product_period = str(top_selling_data.index[0]) # Get name of top product
            except Exception as e:
                st.warning(f"计算热销产品时出错: {e}")
        metrics_results["top_product_period"] = top_product_period
        metrics_results["top_selling_period_chart_data"] = top_selling_data.head(10) # Data for pie chart

        # --- 销售趋势与占比分析 ---
        # 1. 月度趋势数据
        monthly_trend_data = pd.Series(dtype=float)
        # 2. 产品类别销售占比
        category_sales_data = pd.Series(dtype=float)
        # 3. 渠道销售占比
        channel_sales_data = pd.Series(dtype=float)
        
        if not sales_filtered.empty and '订单日期' in sales_filtered.columns and '销售数量_num_calc' in sales_filtered.columns:
            try:
                # 1. 月度趋势数据
                monthly_trend_data = sales_filtered.groupby(sales_filtered["订单日期"].dt.to_period("M"))['销售数量_num_calc'].sum()
                
                # 2. 产品类别销售占比 (如果存在类别数据)
                if '产品分类' in sales_filtered.columns and not sales_filtered['产品分类'].isnull().all():
                    category_sales_data = sales_filtered.groupby("产品分类")['销售数量_num_calc'].sum().sort_values(ascending=False)
                
                # 3. 渠道销售占比 (如果存在渠道数据)
                if '销售渠道' in sales_filtered.columns and not sales_filtered['销售渠道'].isnull().all():
                    channel_sales_data = sales_filtered.groupby("销售渠道")['销售数量_num_calc'].sum().sort_values(ascending=False)
                    
            except Exception as e:
                st.warning(f"计算销售趋势与占比时出错: {e}")
                
        metrics_results.update({
            "monthly_trend_chart_data": monthly_trend_data,
            "category_sales_distribution": category_sales_data,
            "channel_sales_distribution": channel_sales_data
        })

        # --- Stock Analysis ---
        if not isinstance(stock_df, pd.DataFrame) or stock_df.empty or "产品ID" not in stock_df.columns or "当前库存" not in stock_df.columns:
             st.warning("库存数据无效或缺少必需列 ('产品ID', '当前库存')，无法进行详细库存分析。")
             # Define empty dataframe with expected columns for consistency downstream (添加毛利列)
             stock_analysis = pd.DataFrame(columns=["产品ID", "产品名称", "当前库存", "最后销售日期", "压货时间_天", "最后采购日期", "最后采购数量", "天数自上次采购", "期间销售量", "期间销售额", "期间销售成本", "期间毛利", "毛利率", "期间日均销量", "预计可用天数", "产品分类"])
             metrics_results["total_stock_units"] = 0
             return metrics_results, stock_analysis, False # Return empty results but defined structure

        # Calculate total stock units
        stock_df['当前库存_num_calc'] = pd.to_numeric(stock_df['当前库存'], errors='coerce').fillna(0)
        metrics_results["total_stock_units"] = int(stock_df['当前库存_num_calc'].sum())

        # Check if category data exists and is usable
        has_category = "产品分类" in stock_df.columns and not stock_df["产品分类"].isnull().all()

        # Base stock analysis dataframe from unique product IDs in stock data
        stock_analysis_cols = ["产品ID", "产品名称", "当前库存"] # Start with essential cols
        if has_category: stock_analysis_cols.append("产品分类") # Add category if available
        # Ensure only columns that actually exist in stock_df are selected
        stock_analysis_cols_present = [col for col in stock_analysis_cols if col in stock_df.columns]
        # Ensure '采购价' is included if available, as it's needed for margin calculation
        if '采购价' in stock_df.columns:
            stock_analysis_cols_present.append('采购价')
        else:
            # 如果库存数据中没有采购价，则无法计算毛利
            st.warning("警告：库存数据缺少 '采购价' 列，无法计算毛利率。")
            # 添加一个空的采购价列以便后续流程不报错，但毛利会是 NaN
            stock_df['采购价'] = np.nan
            if '采购价' not in stock_analysis_cols_present:
                stock_analysis_cols_present.append('采购价')
        # Drop duplicates based on Product ID, keeping the first occurrence
        stock_analysis_base = stock_df[stock_analysis_cols_present].drop_duplicates(subset=["产品ID"], keep='first').copy()


        # --- Merge Last Sale Date ---
        last_sale_overall = pd.Series(dtype='datetime64[ns]')
        if isinstance(sales_df, pd.DataFrame) and not sales_df.empty and "产品ID" in sales_df.columns and "订单日期" in sales_df.columns:
            try:
                # Use only valid sales records (where date is not NaT)
                valid_sales = sales_df.dropna(subset=['订单日期', '产品ID'])
                if not valid_sales.empty:
                    # Find the index of the latest sale date for each product ID
                    last_sale_idx = valid_sales.groupby("产品ID")["订单日期"].idxmax()
                    # Create a mapping Series: Product ID -> Last Sale Date
                    last_sale_overall = valid_sales.loc[last_sale_idx].set_index('产品ID')['订单日期'].rename("最后销售日期")
            except Exception as e:
                st.warning(f"聚合最后销售日期时出错: {e}")

        # Merge last sale date into the stock analysis table
        stock_analysis = stock_analysis_base.merge(last_sale_overall, on="产品ID", how="left")


        # --- Calculate Stock Aging (Days Since Last Sale) ---
        now_ts_aware = pd.Timestamp.now(tz=APP_TIMEZONE)
        now_ts_naive = now_ts_aware.tz_localize(None) # Use naive timestamp for calculations with naive dates from files

        if '最后销售日期' in stock_analysis.columns:
             # Ensure the merged date is datetime, handle potential errors post-merge
             stock_analysis["最后销售日期"] = pd.to_datetime(stock_analysis["最后销售日期"], errors='coerce').dt.tz_localize(None) # Make naive
             # Calculate difference only where last sale date is valid
             valid_last_sale_mask = stock_analysis["最后销售日期"].notna()
             stock_analysis.loc[valid_last_sale_mask, "压货时间_天"] = (now_ts_naive - stock_analysis.loc[valid_last_sale_mask, "最后销售日期"]).dt.days
        else:
            # If column doesn't exist after merge (shouldn't happen if logic above is right)
            stock_analysis["压货时间_天"] = np.nan

        # Fill NaN (never sold or error) with 9999, convert to int, clip max value
        stock_analysis["压货时间_天"] = stock_analysis["压货时间_天"].fillna(9999).astype(int).clip(upper=9999)
        
        # 计算总销量: 按产品ID聚合销售数据
        if not sales_df.empty:
            sales_by_product = sales_df.groupby("产品ID")["销售数量"].sum().reset_index()
            sales_by_product.rename(columns={"销售数量": "总销量"}, inplace=True)
            stock_analysis = stock_analysis.merge(
                sales_by_product,
                on="产品ID",
                how="left"
            )
            stock_analysis["总销量"] = stock_analysis["总销量"].fillna(0)
        else:
            stock_analysis["总销量"] = 0
            
        # 计算期间日均销量 = 总销量 / 分析天数
        analysis_days = (end_ts - start_ts).days + 1  # +1 to include both start and end dates
        stock_analysis["期间日均销量"] = stock_analysis["总销量"] / analysis_days
        stock_analysis["期间日均销量"] = stock_analysis["期间日均销量"].fillna(0)
        
        # --- 动销分析 ---
        # 是否动销 (90天内是否有销售)
        stock_analysis["是否动销"] = stock_analysis["压货时间_天"] <= dead_stock_threshold_days
        
        # 滞销等级 (高/中/低)
        conditions = [
            (stock_analysis["压货时间_天"] > dead_stock_threshold_days) &
            (stock_analysis["期间日均销量"] < dead_stock_sales_threshold),
            (stock_analysis["压货时间_天"] > dead_stock_threshold_days//2) &
            (stock_analysis["期间日均销量"] < dead_stock_sales_threshold*2),
            True  # 默认值
        ]
        choices = ["高", "中", "低"]
        stock_analysis["滞销等级"] = np.select(conditions, choices, default="低")


        # --- Merge Last Purchase Info ---
        # Initialize columns to default values
        stock_analysis["最后采购日期"] = pd.NaT
        stock_analysis["最后采购数量"] = 0
        stock_analysis["天数自上次采购"] = 9999

        if isinstance(purchase_df, pd.DataFrame) and not purchase_df.empty and all(col in purchase_df.columns for col in ["产品ID", "采购日期", "采购数量"]):
            # Ensure purchase date is datetime and drop invalid rows
            purchase_df['采购日期'] = pd.to_datetime(purchase_df['采购日期'], errors='coerce')
            purchase_df_valid = purchase_df.dropna(subset=['采购日期', '产品ID', '采购数量'])

            if not purchase_df_valid.empty:
                 try:
                      # Ensure purchase quantity is numeric for potential use later (though not directly used for mapping date)
                      purchase_df_valid['采购数量'] = pd.to_numeric(purchase_df_valid['采购数量'], errors='coerce').fillna(0)
                      # Find index of the latest purchase date per product ID
                      last_purchase_idx = purchase_df_valid.groupby('产品ID')['采购日期'].idxmax()
                      # Create mapping dataframe: Product ID -> Last Purchase Date, Last Purchase Qty
                      last_purchase_map = purchase_df_valid.loc[last_purchase_idx].set_index('产品ID')

                      # Map last purchase date and quantity to stock analysis table
                      stock_analysis['最后采购日期'] = stock_analysis['产品ID'].map(last_purchase_map['采购日期'])
                      stock_analysis['最后采购数量'] = stock_analysis['产品ID'].map(last_purchase_map['采购数量'])

                      # Calculate days since last purchase
                      if '最后采购日期' in stock_analysis.columns:
                          # Ensure date is datetime and naive
                          stock_analysis["最后采购日期"] = pd.to_datetime(stock_analysis["最后采购日期"], errors='coerce').dt.tz_localize(None)
                          valid_purchase_dates_mask = stock_analysis['最后采购日期'].notna()
                          stock_analysis.loc[valid_purchase_dates_mask, "天数自上次采购"] = (now_ts_naive - stock_analysis.loc[valid_purchase_dates_mask, "最后采购日期"]).dt.days

                      # Fill NaN (no purchase record or error) with 9999, convert to int, clip max
                      stock_analysis["天数自上次采购"] = stock_analysis["天数自上次采购"].fillna(9999).astype(int).clip(upper=9999)

                 except Exception as merge_err:
                      st.warning(f"合并采购数据时出错: {merge_err}")
            else:
                 st.caption("无有效的采购记录行可供合并。")


        # --- Calculate Sales & Revenue Within Period ---
        stock_analysis["期间销售量"] = 0 # Initialize
        stock_analysis["期间销售额"] = 0.0 # Initialize
        qty_col_sales_filtered = '销售数量_num_calc'
        revenue_col_sales_filtered = '销售额_num_calc'

        if not sales_filtered.empty and "产品ID" in sales_filtered.columns and qty_col_sales_filtered in sales_filtered.columns and revenue_col_sales_filtered in sales_filtered.columns:
             try:
                 # Aggregate sales quantity and revenue within the filtered period by product ID
                 sales_in_period_agg = sales_filtered.groupby("产品ID").agg(
                     期间销售量_agg=(qty_col_sales_filtered, 'sum'),
                     期间销售额_agg=(revenue_col_sales_filtered, 'sum')
                 )
                 # Map the aggregated sales to the stock analysis table
                 if "产品ID" in stock_analysis.columns:
                     stock_analysis['期间销售量'] = stock_analysis['产品ID'].map(sales_in_period_agg['期间销售量_agg'])
                     stock_analysis['期间销售额'] = stock_analysis['产品ID'].map(sales_in_period_agg['期间销售额_agg'])
                     # Fill products with no sales in the period with 0
                     stock_analysis["期间销售量"] = stock_analysis["期间销售量"].fillna(0).astype(int)
                     stock_analysis["期间销售额"] = stock_analysis["期间销售额"].fillna(0.0).astype(float)
                 else:
                     st.warning("库存分析缺少'产品ID'列，无法合并期间销售数据。")
             except Exception as e:
                 st.error(f"计算或合并期间销售数据错误: {e}")
        # --- Calculate Average Daily Sales (Period) ---
        if "期间销售量" in stock_analysis.columns:
             stock_analysis["期间日均销量"] = (stock_analysis["期间销售量"] / num_days_period).round(2)
        else:
             stock_analysis["期间日均销量"] = 0.0 # Default if calculation failed


        # --- Calculate Gross Margin ---
        stock_analysis['期间销售成本'] = 0.0
        stock_analysis['期间毛利'] = 0.0
        stock_analysis['毛利率'] = 0.0

        if '期间销售量' in stock_analysis.columns and '采购价' in stock_analysis.columns:
            # Ensure '采购价' is numeric before calculation
            stock_analysis['采购价_num_calc'] = pd.to_numeric(stock_analysis['采购价'], errors='coerce').fillna(0)
            stock_analysis['期间销售成本'] = stock_analysis['期间销售量'] * stock_analysis['采购价_num_calc']
            stock_analysis['期间销售成本'] = stock_analysis['期间销售成本'].fillna(0.0) # Ensure no NaNs

            if '期间销售额' in stock_analysis.columns:
                stock_analysis['期间毛利'] = stock_analysis['期间销售额'] - stock_analysis['期间销售成本']
                stock_analysis['期间毛利'] = stock_analysis['期间毛利'].fillna(0.0) # Ensure no NaNs

                # Calculate margin rate, handle division by zero
                stock_analysis['毛利率'] = np.where(
                    stock_analysis['期间销售额'] != 0, # Condition: sales amount is not zero
                    stock_analysis['期间毛利'] / stock_analysis['期间销售额'], # Value if true
                    0.0 # Value if false (sales amount is zero)
                )
                stock_analysis['毛利率'] = stock_analysis['毛利率'].fillna(0.0) # Ensure no NaNs from other sources
            else:
                st.warning("无法计算毛利和毛利率，缺少 '期间销售额' 数据。")
        else:
            st.warning("无法计算销售成本和毛利，缺少 '期间销售量' 或 '采购价' 数据。")

        # --- Calculate Estimated Stock Days ---
        stock_analysis['预计可用天数'] = 9999 # Initialize with default (infinite/unknown)
        if "期间日均销量" in stock_analysis.columns and "当前库存" in stock_analysis.columns:
             # Use the numeric stock column created earlier
             stock_analysis['当前库存_num_calc'] = pd.to_numeric(stock_analysis['当前库存'], errors='coerce').fillna(0)
             # Calculate only where average daily sales is positive
             mask_positive_sales = stock_analysis['期间日均销量'] > 0
             stock_analysis.loc[mask_positive_sales, '预计可用天数'] = \
                 stock_analysis.loc[mask_positive_sales, '当前库存_num_calc'] / stock_analysis.loc[mask_positive_sales, '期间日均销量']

             # Fill NaN (e.g., from 0 sales) with 9999, round result, convert to int, clip max
             stock_analysis['预计可用天数'] = stock_analysis['预计可用天数'].fillna(9999).round().astype(int).clip(upper=9999)
        # --- Handle Product Category ---
        if has_category and "产品分类" not in stock_analysis.columns and "产品分类" in stock_df.columns:
             # If category exists in original stock but not merged (e.g., due to drop_duplicates issue), try re-mapping
             try:
                 # Create map from original stock data (ensure duplicates removed first)
                 category_map = stock_df[['产品ID', '产品分类']].drop_duplicates(subset=['产品ID']).set_index('产品ID')['产品分类']
                 stock_analysis['产品分类'] = stock_analysis['产品ID'].map(category_map)
                 stock_analysis['产品分类'] = stock_analysis['产品分类'].fillna("未分类") # Fill missing categories
             except KeyError:
                 st.warning("重新合并产品分类时遇到KeyError，跳过。")
             except Exception as cat_err:
                 st.warning(f"重新合并产品分类时出错: {cat_err}")
        elif not has_category and "产品分类" in stock_analysis.columns:
             # If category column ended up in analysis but wasn't expected, drop it
             stock_analysis = stock_analysis.drop(columns=["产品分类"])
        elif has_category and "产品分类" in stock_analysis.columns:
             # If category is present as expected, ensure NaNs are filled
             stock_analysis['产品分类'] = stock_analysis['产品分类'].fillna("未分类")

        # Final check if category column ended up in the result dataframe
        has_category_in_analysis = "产品分类" in stock_analysis.columns if not stock_analysis.empty else False


        # --- Calculate Overall Gross Margin Metrics ---
        if '期间毛利' in stock_analysis.columns and '期间销售额' in stock_analysis.columns:
            total_gross_profit = stock_analysis['期间毛利'].sum()
            total_revenue = stock_analysis['期间销售额'].sum() # Use already calculated sum if available, or sum here
            metrics_results["total_gross_profit_period"] = float(total_gross_profit)
            if total_revenue != 0:
                metrics_results["overall_gross_margin_period"] = float(total_gross_profit / total_revenue)
            else:
                metrics_results["overall_gross_margin_period"] = 0.0
        else:
            # Set defaults if calculation wasn't possible
            metrics_results["total_gross_profit_period"] = 0.0
            metrics_results["overall_gross_margin_period"] = 0.0


        # --- Cleanup Temporary Columns ---
        temp_cols_to_drop = ['销售数量_num_calc', '销售额_num_calc', '当前库存_num_calc', '采购价_num_calc']
        stock_analysis = stock_analysis.drop(columns=[col for col in temp_cols_to_drop if col in stock_analysis.columns], errors='ignore')

        # --- Return results ---
        # --- AI Analysis Section ---
        ai_analysis = {
            "top_insights": [],
            "recommendations": [],
            "risk_alerts": []
        }

        # 1. 识别滞销商品
        dead_stock = stock_analysis[stock_analysis["压货时间_天"] > dead_stock_threshold_days]
        if not dead_stock.empty:
            ai_analysis["risk_alerts"].append({
                "type": "dead_stock",
                "count": len(dead_stock),
                "total_value": (dead_stock["当前库存"] * dead_stock["采购价"]).sum(),
                "items": dead_stock[["产品ID", "产品名称", "当前库存", "压货时间_天"]].to_dict('records')
            })
            ai_analysis["recommendations"].append("考虑对{}种滞销商品进行促销或清仓处理".format(len(dead_stock)))

        # 2. 识别热销商品库存不足
        fast_moving = stock_analysis[
            (stock_analysis["期间日均销量"] > dead_stock_sales_threshold * 3) &
            (stock_analysis["预计可用天数"] < 7)
        ]
        if not fast_moving.empty:
            ai_analysis["risk_alerts"].append({
                "type": "low_stock_fast_moving",
                "count": len(fast_moving),
                "items": fast_moving[["产品ID", "产品名称", "当前库存", "期间日均销量", "预计可用天数"]].to_dict('records')
            })
            ai_analysis["recommendations"].append("建议优先补货{}种热销且库存不足的商品".format(len(fast_moving)))

        # 3. 销售趋势分析
        if "monthly_trend_chart_data" in metrics_results and not metrics_results["monthly_trend_chart_data"].empty:
            trend_data = metrics_results["monthly_trend_chart_data"]
            last_month = trend_data.index[-1]
            growth_rate = (trend_data[last_month] - trend_data[last_month-1]) / trend_data[last_month-1] * 100
            ai_analysis["top_insights"].append({
                "type": "sales_trend",
                "current_month": str(last_month),
                "growth_rate": round(growth_rate, 1),
                "trend": "上升" if growth_rate > 0 else "下降"
            })

        return metrics_results, stock_analysis, has_category_in_analysis, ai_analysis

    except Exception as e:
        st.error(f"在 calculate_metrics 中发生未预料的错误: {e.__class__.__name__}: {e}")
        return {}, pd.DataFrame(), False, {}  # Return empty results including ai_analysis
        traceback.print_exc() # Log detailed error to console
        # Return empty but defined structures (add margin columns)
        return {}, pd.DataFrame(columns=["产品ID", "产品名称", "当前库存", "最后销售日期", "压货时间_天", "最后采购日期", "最后采购数量", "天数自上次采购", "期间销售量", "期间销售额", "期间销售成本", "期间毛利", "毛利率", "期间日均销量", "预计可用天数", "产品分类"]), False


# --- calculate_purchase_suggestions 函数 ---
def calculate_purchase_suggestions(stock_analysis_df, target_days, safety_days,
                                   seasonal_factors=None, promotion_plans=None,
                                   supplier_lead_times=None):
    """计算智能采购建议，考虑多种因素：
    - 基础库存水平
    - 季节性因素 (可选)
    - 促销计划 (可选)
    - 供应商交货周期 (可选)
    """
    if not isinstance(stock_analysis_df, pd.DataFrame) or stock_analysis_df.empty:
        return pd.DataFrame() # Return empty if no input data

    df = stock_analysis_df.copy()
    required_cols = ["期间日均销量", "当前库存", "产品ID", "产品名称"] # 基础计算必需列

    # Check if required columns exist
    if not all(col in df.columns for col in required_cols):
        st.warning("库存分析数据缺少计算采购建议的必要列 ('期间日均销量', '当前库存', '产品ID', '产品名称')。")
        return pd.DataFrame() # Return empty

    try:
        # --- Ensure Numeric Types ---
        # Convert relevant columns to numeric, coercing errors and filling NaNs with 0
        df['期间日均销量_num'] = pd.to_numeric(df['期间日均销量'], errors='coerce').fillna(0)
        df['当前库存_num'] = pd.to_numeric(df['当前库存'], errors='coerce').fillna(0)

        # --- 计算基础目标库存水平 ---
        df["基础目标库存"] = df["期间日均销量_num"] * (target_days + safety_days)

        # --- 应用智能调整因子 ---
        # 1. 季节性因素调整
        if seasonal_factors and "产品ID" in seasonal_factors.columns and "季节系数" in seasonal_factors.columns:
            df = df.merge(seasonal_factors[["产品ID", "季节系数"]], on="产品ID", how="left")
            df["季节系数"] = df["季节系数"].fillna(1.0) # 默认无季节影响
            df["季节调整销量"] = df["期间日均销量_num"] * df["季节系数"]
        else:
            df["季节调整销量"] = df["期间日均销量_num"]

        # 2. 促销计划调整
        if promotion_plans and "产品ID" in promotion_plans.columns and "促销系数" in promotion_plans.columns:
            df = df.merge(promotion_plans[["产品ID", "促销系数"]], on="产品ID", how="left")
            df["促销系数"] = df["促销系数"].fillna(1.0) # 默认无促销影响
            df["促销调整销量"] = df["季节调整销量"] * df["促销系数"]
        else:
            df["促销调整销量"] = df["季节调整销量"]

        # 3. 供应商交货周期调整
        if supplier_lead_times and "产品ID" in supplier_lead_times.columns and "交货周期" in supplier_lead_times.columns:
            df = df.merge(supplier_lead_times[["产品ID", "交货周期"]], on="产品ID", how="left")
            df["交货周期"] = df["交货周期"].fillna(7) # 默认7天交货周期
            adjusted_target_days = target_days + df["交货周期"]
        else:
            adjusted_target_days = target_days

        # --- 计算最终目标库存水平 ---
        df["目标库存水平"] = df["促销调整销量"] * (adjusted_target_days + safety_days)

        # --- 计算原始建议量 ---
        df["建议采购量_raw"] = df["目标库存水平"] - df["当前库存_num"]

        # --- Final Suggestion (Round Up, Non-Negative Integer) ---
        # Apply math.ceil to round up, ensure it's at least 0, then convert to integer
        df["建议采购量"] = df["建议采购量_raw"].apply(
            lambda x: max(0, math.ceil(x)) if pd.notnull(x) and x > -float('inf') else 0 # Handle potential NaN/inf from raw calc
        ).astype(int)

        # --- Filter and Select Display Columns ---
        purchase_suggestions = df[df["建议采购量"] > 0].copy() # Only show suggestions > 0

        # Define columns to display in the final suggestion table
        display_cols = ["产品名称"] # 始终显示产品名称
        # 添加分类信息（如果存在）
        if "产品分类" in purchase_suggestions.columns:
            display_cols.append("产品分类")

        # 添加标准指标列（如果存在）
        standard_cols = ["当前库存", "预计可用天数", "期间日均销量", "季节调整销量",
                        "促销调整销量", "目标库存水平", "建议采购量"]
        for col in standard_cols:
            if col in purchase_suggestions.columns:
                display_cols.append(col)

        # Ensure all selected display columns actually exist (safety check)
        final_display_cols = [col for col in display_cols if col in purchase_suggestions.columns]

        # Create the final dataframe with selected columns
        purchase_suggestions_final = purchase_suggestions[final_display_cols]

        # Sort by suggested quantity descending
        purchase_suggestions_final = purchase_suggestions_final.sort_values("建议采购量", ascending=False)

        return purchase_suggestions_final

    except Exception as e:
        st.error(f"计算采购建议时出错: {e}")
        traceback.print_exc() # Log detailed error
        return pd.DataFrame() # Return empty on error


# ======== Streamlit App Layout ========
import requests
import io
import json

st.set_page_config(page_title=f"TP.STER 智能数据平台 {APP_VERSION}", layout="wide", page_icon="📊")

# Font Missing Hint
if not FONT_AVAILABLE:
    st.warning("""⚠️ **未找到中文字体...** 图表中的中文标签可能无法正确显示。请尝试安装 'SimHei' 或类似的中文字体，或将字体文件 (如 `SimHei.ttf`) 放置于脚本相同目录下。""", icon="ℹ️")

# --- Sidebar ---
with st.sidebar:
    st.markdown(" ")
    logo_path = "tpster_logo.png" # Ensure your logo file is named this and in the same directory

    # --- MODIFIED LOGO DISPLAY (v2 - Base64) ---
    logo_data_uri = None
    placeholder_needed = True
    placeholder_url = "https://via.placeholder.com/180x80?text=TP.STER+LOGO"
    error_msg_logo = None

    if os.path.exists(logo_path):
        logo_data_uri = get_image_as_base64(logo_path)
        if logo_data_uri:
            placeholder_needed = False
        else:
            # File exists but encoding failed
            error_msg_logo = "Error loading logo file."
            placeholder_needed = True # Fallback to placeholder
    else:
        # File does not exist
        placeholder_needed = True


    image_source = logo_data_uri if not placeholder_needed else placeholder_url

    # Use markdown with centered div and img tag
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 5px;">
            <img src="{image_source}" alt="Logo" width="180" style="display: block; margin-left: auto; margin-right: auto;">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display caption or error centered below the image
    if placeholder_needed and not error_msg_logo:
        st.markdown("<div style='text-align: center; margin-top: 0px;'><p style='font-size: 0.8em; color: grey;'>Logo Placeholder</p></div>", unsafe_allow_html=True)
    elif error_msg_logo:
         st.markdown(f"<div style='text-align: center; margin-top: 0px;'><p style='font-size: 0.8em; color: red;'>{error_msg_logo}</p></div>", unsafe_allow_html=True)
    # --- END OF MODIFIED LOGO DISPLAY ---

    # Initialize variables
    uploaded_main_file = None
    uploaded_financial_file = None
    uploaded_crm_file = None
    main_analysis_ready = False
    financial_data_ready = False
    crm_data_ready = False

    # AI Configuration Section
    with st.expander("🤖 AI功能配置"):
        ai_enabled = st.checkbox("启用AI分析功能", value=False)
        if ai_enabled:
            st.info("AI功能将提供智能分析建议")
            ai_model = st.selectbox(
                "选择AI模型",
                ["销售预测", "库存优化", "采购建议"],
                index=0
            )
            # 内置DeepSeek API密钥
            deepseek_key = "sk-28e9f3ab9805477c8996c6919b109e1f"

    st.markdown(" ") # Add space after logo/caption
    st.markdown(f"<div style='text-align: center; font-size: 12px; color: gray;'>版本: {APP_VERSION}</div>", unsafe_allow_html=True)
    # Example User Info - Replace with actual authentication if needed
    st.markdown("<div style='text-align: center; font-size: 14px; color: gray; margin-bottom: 10px; margin-top: 5px;'>当前身份：<strong>管理员</strong> (示例)</div>", unsafe_allow_html=True)

    # --- Real-time Clock Display using HTML/JS Component ---
    html_code = """
    <div style='text-align: center; font-size: 12px; color: gray; margin-top: 8px;'>
        🇬🇷 希腊: <span id="athens-time">--:--:--</span>
    </div>
    <div style='text-align: center; font-size: 12px; color: gray; margin-top: 2px; margin-bottom: 8px;'>
        🇨🇳 北京: <span id="beijing-time">--:--:--</span>
    </div>

    <script>
    function updateClocks() {
      const now = new Date();
      const options = {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false // Use 24-hour format
      };

      try {
        // Athens Time (using Europe/Athens)
        const athensTimeStr = now.toLocaleString('sv-SE', { ...options, timeZone: 'Europe/Athens' }); // sv-SE gives YYYY-MM-DD format
        document.getElementById('athens-time').innerText = athensTimeStr;
      } catch (e) {
        document.getElementById('athens-time').innerText = '时间加载错误';
        console.error("Error getting Athens time:", e);
      }

      try {
        // Beijing Time (using Asia/Shanghai)
        const beijingTimeStr = now.toLocaleString('sv-SE', { ...options, timeZone: 'Asia/Shanghai' }); // sv-SE gives YYYY-MM-DD format
        document.getElementById('beijing-time').innerText = beijingTimeStr;
      } catch (e) {
        document.getElementById('beijing-time').innerText = '时间加载错误';
        console.error("Error getting Beijing time:", e);
      }
    }

    // Initial call to display time immediately
    updateClocks();

    // Update every second
    setInterval(updateClocks, 1000);
    </script>
    """
    components.html(html_code, height=60) # Adjust height as needed

    st.markdown("---")

    # --- File Uploaders ---
    st.markdown("#### 📂 百货城数据 (采购建议)")
    
    # 数据源选择
    data_source = st.radio(
        "选择数据来源",
        ["上传文件", "ERP系统API"],
        horizontal=True,
        key="main_data_source"
    )
    
    if data_source == "上传文件":
        uploaded_main_file = st.file_uploader(
            label="上传主数据文件 (Excel/CSV)",
            type=["xlsx", "xls", "csv"],
            key="main_data_uploader",
            help="上传包含销售(计算需求)、库存(检查低库存)及可选采购数据的文件。Excel需含'订单数据', '库存数据'表。CSV需含'DataType'列。"
        )
    else:
        # ERP API 配置
        with st.expander("ERP API 配置"):
            api_url = st.text_input("API地址", "https://erp.example.com/api")
            api_key = st.text_input("API密钥", type="password")
            endpoint = st.selectbox(
                "数据端点",
                ["/inventory", "/sales", "/purchase"]
            )
        
        if st.button("从ERP系统获取数据"):
            try:
                with st.spinner("正在从ERP系统获取数据..."):
                    # 调用ERP API获取数据
                    headers = {"Authorization": f"Bearer {api_key}"}
                    params = {
                        "start_date": start_date_dt.strftime("%Y-%m-%d"),
                        "end_date": end_date_dt.strftime("%Y-%m-%d")
                    }
                    response = requests.get(
                        url=f"{api_url}{endpoint}",
                        headers=headers,
                        params=params,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    # 转换API响应为DataFrame
                    data = response.json()
                    uploaded_main_file = io.StringIO(json.dumps(data))
                    st.success("数据获取成功!")
            except requests.exceptions.RequestException as e:
                st.error(f"从ERP系统获取数据失败: {str(e)}")
                st.exception(e)
    st.divider()
    # --- NEW: Additional Data Uploaders ---
    st.markdown("#### 📊 财务数据 (可选)")
    
    # 财务数据源选择
    fin_data_source = st.radio(
        "选择财务数据来源",
        ["上传文件", "ERP系统API"],
        horizontal=True,
        key="fin_data_source"
    )
    
    if fin_data_source == "上传文件":
        uploaded_financial_file = st.file_uploader(
            label="上传财务数据文件 (Excel/CSV)",
            type=["xlsx", "xls", "csv"],
            key="financial_data_uploader",
            help="上传包含收入、支出、利润等财务指标的文件。"
        )
    else:
        # ERP财务API配置
        with st.expander("ERP财务API配置"):
            fin_api_url = st.text_input("财务API地址", "https://erp.example.com/finance/api")
            fin_api_key = st.text_input("财务API密钥", type="password", key="fin_api_key")
            fin_endpoint = st.selectbox(
                "财务数据端点",
                ["/balance_sheet", "/income_statement", "/cash_flow"],
                key="fin_endpoint"
            )
        
        if st.button("从ERP系统获取财务数据"):
            try:
                with st.spinner("正在从ERP系统获取财务数据..."):
                    # 调用ERP财务API获取数据
                    headers = {"Authorization": f"Bearer {fin_api_key}"}
                    params = {"start_date": start_date_dt, "end_date": end_date_dt}
                    response = requests.get(
                        url=f"{fin_api_url}{fin_endpoint}",
                        headers=headers,
                        params=params,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    # 转换API响应为DataFrame
                    data = response.json()
                    uploaded_financial_file = io.StringIO(json.dumps(data))
                    st.success("财务数据获取成功!")
            except requests.exceptions.RequestException as e:
                st.error(f"从ERP系统获取财务数据失败: {str(e)}")
                st.exception(e)
    st.divider()

    st.markdown("#### 👥 CRM 数据 (可选)")
    uploaded_crm_file = st.file_uploader(
        label="上传CRM数据文件 (Excel/CSV)",
        type=["xlsx", "xls", "csv"],
        key="crm_data_uploader",
        help="上传包含客户联系信息、购买历史等CRM数据的文件。"
    )
    st.divider()


    # --- Data Loading and State Management ---
    main_sales_data, main_stock_data, main_purchase_data = None, None, None
    
    # AI Analysis Section - 更醒目的展示位置
    if ai_enabled and main_analysis_ready:
        st.markdown("---")
        st.markdown("## 🚀 AI智能分析报告")
        st.markdown("---")
        with st.spinner("🤖 正在使用DeepSeek AI分析数据..."):
            try:
                if ai_model == "销售预测":
                    # Call DeepSeek API for sales prediction
                    try:
                        headers = {
                            "Authorization": f"Bearer {deepseek_key}",  # 使用内置API密钥
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": "deepseek-sales-forecast",
                            "data": main_sales_data.to_dict(orient="records")
                        }
                        response = requests.post(
                            "https://api.deepseek.com/v1/sales/forecast",
                            headers=headers,
                            json=payload,
                            timeout=30
                        )
                        response.raise_for_status()
                        forecast = response.json()
                        st.success("✅ 销售预测分析完成")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("预计下月销售额增长",
                                    f"{forecast['growth_rate']}%",
                                    delta=f"{forecast['growth_rate']}%")
                            st.write("📈 趋势分析:", forecast['trend_analysis'])
                        
                        with col2:
                            st.write("🔥 热门商品预测:")
                            for product in forecast['top_products']:
                                st.markdown(f"- {product}")
                        
                        # 添加可视化图表
                        st.line_chart(pd.DataFrame({
                            '月份': forecast['monthly_trend']['months'],
                            '销售额': forecast['monthly_trend']['values']
                        }).set_index('月份'))
                    except Exception as e:
                        st.error(f"销售预测失败: {str(e)}")
                elif ai_model == "库存优化":
                    # Placeholder for inventory optimization logic
                    st.success("库存优化建议生成")
                    st.write("建议减少以下商品库存: 商品A, 商品B")
                elif ai_model == "采购建议":
                    # Placeholder for purchase recommendation logic
                    st.success("采购建议生成")
                    st.write("建议优先采购: 商品C, 商品D")
            except Exception as e:
                st.error(f"AI分析失败: {str(e)}")
    main_analysis_ready = False
    financial_data_ready = False # New state
    crm_data_ready = False       # New state
    has_category_column_main = False
    # -- Process Main Data File --
    if uploaded_main_file:
        current_main_file_id = uploaded_main_file.file_id
        # Check if it's the same file that previously caused an error
        if current_main_file_id == st.session_state.last_main_file_id and st.session_state.main_load_error:
            st.error(st.session_state.main_load_error) # Show the stored error
            main_analysis_ready = False
        # Otherwise, try to load (new file or previously successful one)
        elif current_main_file_id != st.session_state.last_main_file_id or not st.session_state.main_load_error:
            st.session_state.last_main_file_id = current_main_file_id # Store current file ID
            with st.spinner("⏳ 正在加载分析数据..."):
                uploaded_main_content = uploaded_main_file.getvalue()
                # Call modified load_data which returns error message
                sales_df, stock_df, purchase_df, error_msg = load_data(uploaded_main_content, uploaded_main_file.name)

            if error_msg:
                st.error(error_msg) # Show error returned by loading function
                st.session_state.main_load_error = error_msg # Store the error
                main_analysis_ready = False
            elif isinstance(sales_df, pd.DataFrame) and isinstance(stock_df, pd.DataFrame):
                st.session_state.main_load_error = None # Clear any previous error
                main_sales_data, main_stock_data, main_purchase_data = sales_df, stock_df, purchase_df
                main_analysis_ready = True
                # Check for category column after successful load
                if not main_stock_data.empty:
                    has_category_column_main = ("产品分类" in main_stock_data.columns and not main_stock_data["产品分类"].isnull().all())
            else:
                 # Handle unexpected case where load_data returned no error but invalid data
                unknown_error = "加载分析数据时发生未知问题，未收到有效数据。"
                st.error(unknown_error)
                st.session_state.main_load_error = unknown_error
                main_analysis_ready = False

    # If user clears the file uploader, reset the state
    elif not uploaded_main_file and st.session_state.last_main_file_id is not None:
        st.session_state.last_main_file_id = None
        st.session_state.main_load_error = None
        main_analysis_ready = False # Ensure state is reset
    # --- NEW: Process Additional Data Files (Placeholder Logic) ---
    # Initialize session state for new file types if they don't exist
    if 'financial_load_error' not in st.session_state: st.session_state.financial_load_error = None
    if 'last_financial_file_id' not in st.session_state: st.session_state.last_financial_file_id = None
    if 'crm_load_error' not in st.session_state: st.session_state.crm_load_error = None
    if 'last_crm_file_id' not in st.session_state: st.session_state.last_crm_file_id = None
    if 'production_load_error' not in st.session_state: st.session_state.production_load_error = None
    if 'last_production_file_id' not in st.session_state: st.session_state.last_production_file_id = None
    if 'hr_load_error' not in st.session_state: st.session_state.hr_load_error = None
    if 'last_hr_file_id' not in st.session_state: st.session_state.last_hr_file_id = None

    # Placeholder processing logic for Financial Data
    financial_data_loaded = None
    if uploaded_financial_file:
        current_financial_file_id = uploaded_financial_file.file_id
        if current_financial_file_id == st.session_state.last_financial_file_id and st.session_state.financial_load_error:
            st.error(f"财务数据加载错误: {st.session_state.financial_load_error}")
            financial_data_ready = False
        elif current_financial_file_id != st.session_state.last_financial_file_id or not st.session_state.financial_load_error:
            st.session_state.last_financial_file_id = current_financial_file_id
            try:
                # Placeholder: In a real scenario, call a load_financial_data function here
                financial_data_loaded = pd.read_excel(io.BytesIO(uploaded_financial_file.getvalue())) # Basic load example
                st.success("财务数据文件已加载 (占位符)。")
                st.session_state.financial_load_error = None
                financial_data_ready = True
            except Exception as e:
                error_msg = f"加载财务数据失败: {e}"
                st.error(error_msg)
                st.session_state.financial_load_error = error_msg
                financial_data_ready = False
    elif not uploaded_financial_file and st.session_state.last_financial_file_id is not None:
        st.session_state.last_financial_file_id = None
        st.session_state.financial_load_error = None
        financial_data_ready = False

    # Placeholder processing logic for CRM Data (similar structure)
    crm_data_loaded = None
    if uploaded_crm_file:
        current_crm_file_id = uploaded_crm_file.file_id
        if current_crm_file_id == st.session_state.last_crm_file_id and st.session_state.crm_load_error:
            st.error(f"CRM数据加载错误: {st.session_state.crm_load_error}")
            crm_data_ready = False
        elif current_crm_file_id != st.session_state.last_crm_file_id or not st.session_state.crm_load_error:
            st.session_state.last_crm_file_id = current_crm_file_id
            try:
                crm_data_loaded = pd.read_excel(io.BytesIO(uploaded_crm_file.getvalue()))
                st.success("CRM数据文件已加载 (占位符)。")
                st.session_state.crm_load_error = None
                crm_data_ready = True
            except Exception as e:
                error_msg = f"加载CRM数据失败: {e}"
                st.error(error_msg)
                st.session_state.crm_load_error = error_msg
                crm_data_ready = False
    elif not uploaded_crm_file and st.session_state.last_crm_file_id is not None:
        st.session_state.last_crm_file_id = None
        st.session_state.crm_load_error = None
        crm_data_ready = False

    # --- END NEW: Process Additional Data Files ---

    # --- Analysis Parameters (Only show if main data loaded successfully) ---
    selected_category = "全部" # Default value
    try:
        default_end_date = datetime.now(APP_TIMEZONE).date()
    except Exception as e:
        st.warning(f"获取当前日期时出错({APP_TIMEZONE_STR}): {e}，使用 UTC。")
        default_end_date = datetime.utcnow().date()
    default_start_date = default_end_date - timedelta(days=89) # Default 90 day period
    date_range = (default_start_date, default_end_date) # Default range tuple
    target_days_input = 30 # Default target days
    safety_days_input = 7 # Default safety days

    if main_analysis_ready:
        with st.expander("⚙️ 分析参数设置", expanded=True): # Default expanded

            # Category Filter (Indented)
            if has_category_column_main:
                try:
                    # Get unique categories, convert to string, sort, handle potential NaN/None
                    all_categories = sorted([str(cat) for cat in main_stock_data["产品分类"].dropna().unique()])
                    options = ["全部"] + [cat for cat in all_categories if cat != "全部"] # Ensure "全部" is first
                    selected_category = st.selectbox(
                        "🗂️ 产品分类筛选",
                        options=options,
                        index=0, # Default to "全部"
                        key="category_select_key"
                    )
                except Exception as cat_err:
                    st.warning(f"加载产品分类选项时出错: {cat_err}")
                    selected_category = "全部" # Fallback
                    has_category_column_main = False # Disable filtering if error occurs
            else:
                selected_category = "全部" # Set default if no category data

            # Date Range Selector (Indented)
            st.markdown("##### 🗓️ 销售分析周期")
            min_date_allowed = None
            max_date_allowed = None
            if main_sales_data is not None and not main_sales_data.empty and '订单日期' in main_sales_data.columns:
                 # Use already converted and cleaned dates
                 valid_dates = main_sales_data['订单日期'].dropna()
                 if not valid_dates.empty:
                     try:
                          min_date_allowed = valid_dates.min().date()
                          max_date_allowed = valid_dates.max().date()
                     except Exception as date_parse_err:
                          st.warning(f"无法解析销售数据中的日期范围: {date_parse_err}")

            # Determine final min/max for the date picker, considering data range and default range
            final_min_date = min(min_date_allowed, default_start_date) if min_date_allowed else default_start_date
            final_max_date = max(max_date_allowed, default_end_date) if max_date_allowed else default_end_date
            # Ensure min is not after max
            if final_min_date > final_max_date: final_min_date = final_max_date

            # Adjust default start/end to be within the allowed range
            actual_default_start = max(final_min_date, default_start_date)
            actual_default_end = min(final_max_date, default_end_date)
            # Ensure start is not after end in default values
            if actual_default_start > actual_default_end: actual_default_start = actual_default_end

            try:
                date_range_input = st.date_input(
                    "选择周期",
                    value=(actual_default_start, actual_default_end), # Use adjusted defaults
                    min_value=final_min_date,
                    max_value=final_max_date,
                    key="date_range_selector",
                    help="选择用于计算期间日均销量等指标的时间范围。"
                )
                # Validate the input tuple/list
                if isinstance(date_range_input, (tuple, list)) and len(date_range_input) == 2:
                    start_d, end_d = date_range_input
                    if start_d <= end_d:
                        date_range = (start_d, end_d) # Update date_range if valid
                    else:
                        st.warning("开始日期不能晚于结束日期，使用上次有效或默认范围。")
                        # Keep previous date_range value
                else:
                    # Handle cases where date_input might return a single date if max_value = min_value
                     if isinstance(date_range_input, datetime.date):
                         date_range = (date_range_input, date_range_input)
                     else:
                         st.warning("日期范围选择无效，将使用上次有效或默认范围。")
                         # Keep previous date_range value
            except Exception as date_err:
                st.warning(f"日期范围设置时出错: {date_err}，将使用默认范围。")
                date_range = (actual_default_start, actual_default_end) # Fallback to defaults on error

            # Inventory/Purchase Parameters (Indented)
            st.markdown("##### ⚙️ 库存与采购参数")
            target_days_input = st.number_input(
                "目标库存天数",
                min_value=1, max_value=180, value=30, step=1,
                key="target_days_key",
                help="期望库存能满足多少天的销售"
            )
            safety_days_input = st.number_input(
                "安全库存天数",
                min_value=0, max_value=90, value=7, step=1,
                key="safety_days_key",
                help="额外的缓冲天数"
            )

            # Prediction Method Selection
            st.markdown("##### 📊 销量预测方法")
            predict_method = st.selectbox(
                "选择预测算法",
                ["moving_average", "exponential_smoothing"],
                index=1,
                key="predict_method_key",
                help="移动平均适合稳定产品，指数平滑适合有季节性或趋势的产品"
            )

# --- Main Area ---
st.markdown(f"""<div style='text-align: center; padding: 15px 0 10px 0;'><h1 style='margin-bottom: 5px; color: #262730;'>📊 TP.STER 智能数据平台 {APP_VERSION}</h1><p style='color: #5C5C5C; font-size: 18px; font-weight: 300; margin-top: 5px;'>洞察数据价值 · 驱动智能决策 · 优化供应链管理</p></div>""", unsafe_allow_html=True)
st.divider()

# --- Main Content ---
# Display Welcome Message if no files are uploaded at all (Check all potential uploaders now)
if not any([uploaded_main_file, uploaded_financial_file, uploaded_crm_file]): # Adjusted condition
    # Use the Centered and refined welcome message
    st.markdown(
        f"""
        <div style='text-align: center; max-width: 800px; margin: auto; padding-top: 20px; padding-bottom: 30px;'>

        本平台通过集成化分析，助您轻松管理业务数据，核心功能包括：

        <div style='text-align: left; display: inline-block; margin-top: 15px; margin-bottom: 20px;'>
        <ul style="list-style-type: none; padding-left: 0; font-size: 16px;">
            <li style="margin-bottom: 10px;">📊 &nbsp; <strong>销售分析:</strong> 追踪销售趋势，识别核心产品，分析季节性波动</li>
            <li style="margin-bottom: 10px;">📦 &nbsp; <strong>库存分析:</strong> 评估库存健康度，识别滞销商品，优化周转效率</li>
            <li style="margin-bottom: 10px;">🛒 &nbsp; <strong>采购建议:</strong> 基于AI预测，提供精准补货建议，避免断货或积压</li>
            <li style="margin-bottom: 10px;">💰 &nbsp; <strong>财务指标:</strong> 计算毛利率、ROI等关键财务指标，监控业务健康度</li>
            <li style="margin-bottom: 10px;">👥 &nbsp; <strong>CRM摘要:</strong> 分析客户购买行为，识别高价值客户，优化客户关系</li>
        </ul>
        </div>

        <hr style='margin-top: 15px; margin-bottom: 25px; border-top: 1px solid #eee;'>

        #### **开始使用**

        <p style="font-size: 16px; margin-bottom: 20px;">请在左侧边栏上传您的数据文件 (支持 <code>.xlsx</code>, <code>.xls</code>, <code>.csv</code>):</p>

        <div style='text-align: left; max-width: 600px; margin: auto; background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #e9ecef;'>
        <ol style="padding-left: 25px; margin-bottom: 0;">
            <li style="margin-bottom: 15px;">
                <strong>核心数据 (用于分析与建议):</strong><br>
                上传包含 <code>订单数据</code> 和 <code>库存数据</code> 的文件。<br>
                <span style="font-size: 0.9em; color: #6c757d;"><em>(Excel文件需包含名为 "订单数据" 和 "库存数据" 的工作表)</em></span><br>
                <span style="font-size: 0.9em; color: #6c757d;"><em>(CSV文件需包含 'DataType' 列区分数据类型)</em></span>
            </li>
        </ol>
        </div>

        </div>
        """, unsafe_allow_html=True)

# Display content if at least one file was uploaded and processed (or attempted) - Check all potential uploaders
elif any([uploaded_main_file, uploaded_financial_file, uploaded_crm_file]): # Adjusted condition

    metrics = {}
    stock_analysis = pd.DataFrame()
    purchase_suggestions = pd.DataFrame()
    has_category_data = False # Will be set by calculate_metrics
    data_filtered_message = ""

    # Perform main analysis only if data is ready
    if main_analysis_ready:
        st.markdown(f"### 📈 **主数据分析** ({date_range[0].strftime('%Y-%m-%d')} 到 {date_range[1].strftime('%Y-%m-%d')})")

        # --- Apply Category Filtering if selected ---
        sales_calc = main_sales_data.copy()
        stock_calc = main_stock_data.copy()
        # Ensure purchase_calc is a DataFrame even if main_purchase_data was None initially
        purchase_calc = main_purchase_data.copy() if main_purchase_data is not None else pd.DataFrame()

        if selected_category != "全部" and has_category_column_main:
            data_filtered_message = f"分析已筛选，仅包含分类：**{selected_category}**"
            st.info(data_filtered_message) # Show filter info early

            # Filter stock data first
            if "产品分类" in stock_calc.columns and not stock_calc.empty:
                 category_product_ids = stock_calc[stock_calc["产品分类"] == selected_category]["产品ID"].unique()
                 stock_calc = stock_calc[stock_calc["产品ID"].isin(category_product_ids)] # Filter stock

                 # Filter sales and purchase based on the product IDs from the filtered stock
                 if not sales_calc.empty and "产品ID" in sales_calc.columns:
                     sales_calc = sales_calc[sales_calc["产品ID"].isin(category_product_ids)]
                 if purchase_calc is not None and not purchase_calc.empty and "产品ID" in purchase_calc.columns:
                     purchase_calc = purchase_calc[purchase_calc["产品ID"].isin(category_product_ids)]
            else:
                 st.warning("库存数据中无'产品分类'列或数据为空，无法按分类筛选。将分析全部数据。")
                 data_filtered_message = "" # Clear filter message if filtering failed

        # --- Run Calculations ---
        start_date_dt, end_date_dt = date_range
        try:
            with st.spinner('⏳ 正在分析主数据...'):
                 # Pass the potentially filtered dataframes to calculation functions
                 metrics, stock_analysis, has_category_data, ai_analysis = calculate_metrics(
                     sales_calc, stock_calc, purchase_calc, start_date_dt, end_date_dt,
                     ai_model_version="deepseek-v3"
                 )
                 # Ensure AI analysis is always generated
                 if not ai_analysis:
                     ai_analysis = {
                         "top_insights": [],
                         "risk_alerts": [],
                         "recommendations": ["⚠️ AI分析未生成具体结果，但数据已处理完成"]
                     }
                 # Calculate suggestions based on the result of calculate_metrics
                 purchase_suggestions = calculate_purchase_suggestions(
                     stock_analysis, target_days_input, safety_days_input
                 )
        except Exception as calc_error:
            st.error(f"❌ 在执行主数据计算时发生严重错误: {calc_error}")
            traceback.print_exc()
            # Reset results to prevent downstream errors
            metrics = {}
            stock_analysis = pd.DataFrame()
            purchase_suggestions = pd.DataFrame()
            # Keep main_analysis_ready as True if initial load was ok, but show error

        # --- Display KPIs (if calculation succeeded) ---
        if main_analysis_ready and metrics: # Check if metrics dictionary was populated
            if data_filtered_message:
                st.markdown(f"**筛选分类:** `{selected_category}`") # Show filter again if applied

            # 扩展 KPI 列数以容纳毛利指标
            kpi_cols = st.columns(7)
            # Use .get() with default values for robustness
            kpi_cols[0].metric("期间总销量", f"{metrics.get('total_sales_period', 0):,} 个")
            kpi_cols[1].metric("期间总销售额", f"€ {metrics.get('total_revenue_period', 0):,.2f}") # 新增总销售额
            kpi_cols[2].metric("期间总毛利", f"€ {metrics.get('total_gross_profit_period', 0):,.2f}", help="期间总销售额 - 期间总销售成本") # 新增总毛利
            kpi_cols[3].metric("期间毛利率", f"{metrics.get('overall_gross_margin_period', 0):.1%}", help="期间总毛利 / 期间总销售额") # 新增毛利率

            # Calculate SKU count from the final stock_analysis dataframe
            sku_count = stock_analysis['产品ID'].nunique() if isinstance(stock_analysis, pd.DataFrame) and not stock_analysis.empty else 0
            kpi_cols[4].metric("分析产品 SKU 数", f"{sku_count:,}")

            # Calculate total stock from the final stock_analysis dataframe
            total_stock_kpi = 0
            if isinstance(stock_analysis, pd.DataFrame) and '当前库存' in stock_analysis.columns:
                 total_stock_kpi = int(pd.to_numeric(stock_analysis['当前库存'], errors='coerce').fillna(0).sum())
            kpi_cols[5].metric("当前总库存", f"{total_stock_kpi:,} 个")

            kpi_cols[6].metric("期间热销产品", metrics.get('top_product_period', '无'))
            st.divider()
        elif main_analysis_ready and not metrics:
             # This case might happen if calculate_metrics itself failed internally but didn't raise exception caught above
             st.warning("主数据分析计算完成，但未能生成关键指标。请检查数据或计算逻辑。")
             st.divider()
        # If main_analysis_ready is False (loading failed), KPIs won't show. Error is shown in sidebar.

    # --- Display Tabs ---
    # --- Define Tabs (Including New Modules) ---
    tab_list = [
        "📊 销售分析", "📦 库存分析", "🛒 采购建议",
        "🤖 AI分析", # 新增AI分析标签
        "💰 财务指标", "👥 CRM摘要",
        "🔔 待办提醒", "📈 自定义分析"
    ]
    tabs = st.tabs(tab_list)

    # Assign tabs to variables for clarity
    tab_sales = tabs[0]
    tab_inventory = tabs[1]
    tab_purchase = tabs[2]
    tab_ai = tabs[3]
    tab_financial = tabs[4]
    tab_crm = tabs[5]
    tab_alerts = tabs[6]
    tab_custom_analysis = tabs[7]
    # --- End Define Tabs ---

    # --- AI Analysis Tab ---
    with tab_ai:
        # API选择器
        col1, col2 = st.columns([1, 2])
        with col1:
            st.selectbox(
                "选择AI服务提供商",
                ["deepseek", "openai"],
                index=0,
                key="api_provider",
                help="选择要使用的AI API服务"
            )
        
        with col2:
            if st.session_state.get("api_provider", "deepseek") == "deepseek":
                st.text_input(
                    "DeepSeek API密钥 (可选)",
                    type="password",
                    help="留空将使用内置密钥",
                    key="deepseek_api_key"
                )
            else:
                st.text_input(
                    "OpenAI API密钥",
                    type="password",
                    key="openai_api_key"
                )
        if main_analysis_ready:
            # Initialize ai_analysis if not exists
            if 'ai_analysis' not in st.session_state:
                st.session_state.ai_analysis = None
            
            # AI分析结果展示
            st.subheader("AI数据分析摘要")
            if ai_analysis:
                # 直接展示关键数据和优化建议
                st.markdown("### 📊 核心指标")
                if ai_analysis["top_insights"]:
                    for insight in ai_analysis["top_insights"]:
                        if insight["type"] == "sales_trend":
                            trend_icon = "📈" if insight["trend"] == "上升" else "📉"
                            st.metric(
                                f"{trend_icon} {insight['current_month']}月销售趋势",
                                f"{insight['trend']} {abs(insight['growth_rate'])}%"
                            )
                
                st.markdown("### ⚠️ 风险概览")
                if ai_analysis["risk_alerts"]:
                    for alert in ai_analysis["risk_alerts"]:
                        if alert["type"] == "dead_stock":
                            st.error(f"滞销商品: {alert['count']}种")
                        elif alert["type"] == "low_stock_fast_moving":
                            st.warning(f"缺货风险: {alert['count']}种")
                
                # 直接展示优化建议
                if ai_analysis["recommendations"]:
                    st.markdown("### 💡 优化建议")
                    for rec in ai_analysis["recommendations"]:
                        st.success(f"✅ {rec}")
            else:
                st.info("AI分析数据为空，请检查输入数据。")
            
            # 简洁AI对话功能
            st.divider()
            st.subheader("AI数据分析助手")
            
            # 创建容器用于聊天记录
            chat_container = st.container()
            
            # 用户输入框固定在顶部
            if prompt := st.chat_input("请输入您关于数据分析的问题...", key="chat_input"):
                st.session_state.ai_messages = st.session_state.get("ai_messages", [])
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                
                # AI回复
                response = generate_ai_response(prompt, ai_analysis)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                
                # 重新渲染整个聊天记录
                st.rerun()
            
            # 在容器中显示聊天记录
            with chat_container:
                for msg in st.session_state.get("ai_messages", []):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
        elif main_analysis_ready and not ai_analysis:
            st.info("AI分析数据为空，请检查输入数据。")
    # --- End Define Tabs ---

    # --- Sales Analysis Tab ---
    with tab_sales:
        if main_analysis_ready and metrics: # Only show content if main analysis ran successfully
             st.subheader("销售趋势与占比分析")
             chart_cols = st.columns([2, 1])

             # --- Monthly Sales Trend (Line Chart) ---
             with chart_cols[0]:
                 st.markdown("###### 月度销售趋势 (按销售量)")
                 monthly_data = metrics.get('monthly_trend_chart_data')
                 # Check if data is valid Series and contains meaningful data
                 if isinstance(monthly_data, pd.Series) and not monthly_data.empty and not (monthly_data.isnull().all() or (monthly_data == 0).all()):
                     fig_line = None # Initialize figure variable
                     try:
                          # Convert PeriodIndex to Timestamp for plotting
                          plot_index = monthly_data.index.to_timestamp()
                          fig_line, ax_line = plt.subplots(figsize=(8, 3))
                          ax_line.plot(plot_index, monthly_data.values, marker='o', linestyle='-', linewidth=1.5, markersize=4, color='#1f77b4')

                          xlabel, ylabel = "月份", "销售数量 (个)"
                          # Apply Chinese font if available
                          if FONT_AVAILABLE and chinese_font:
                              ax_line.set_xlabel(xlabel, fontproperties=chinese_font, fontsize=9)
                              ax_line.set_ylabel(ylabel, fontproperties=chinese_font, fontsize=9)
                              # Set font for tick labels
                              for label in ax_line.get_xticklabels() + ax_line.get_yticklabels():
                                   label.set_fontproperties(chinese_font)
                                   label.set_fontsize(8)
                          else:
                              ax_line.set_xlabel(xlabel, fontsize=9)
                              ax_line.set_ylabel(ylabel, fontsize=9)
                              for label in ax_line.get_xticklabels() + ax_line.get_yticklabels():
                                   label.set_fontsize(8)

                          # Format x-axis dates
                          ax_line.xaxis.set_major_formatter(mdates.DateFormatter("%Y年%m月"))

                          # --- Dynamic Date Locator Interval ---
                          # Calculate the number of months in the selected range
                          start_date_dt, end_date_dt = date_range # Get dates from sidebar selection
                          num_months_in_range = (end_date_dt.year - start_date_dt.year) * 12 + end_date_dt.month - start_date_dt.month + 1

                          # Determine interval based on the number of months in the selected range
                          if num_months_in_range <= 1: # Very short range (e.g., within a month)
                              locator_interval = 1 # Show the month(s) involved
                          elif num_months_in_range <= 12: # Up to a year
                              locator_interval = 1 # Show every month
                          elif num_months_in_range <= 24: # Up to two years
                              locator_interval = 2 # Show every 2 months
                          elif num_months_in_range <= 48: # Up to four years
                              locator_interval = 3 # Show every 3 months
                          else: # Very long range
                              locator_interval = 6 # Show every 6 months

                          # Apply the determined locator interval
                          ax_line.xaxis.set_major_locator(mdates.MonthLocator(interval=locator_interval))
                          # --- End Dynamic Date Locator Interval ---

                          ax_line.grid(axis='y', linestyle=':', alpha=0.7)
                          fig_line.autofmt_xdate(rotation=30, ha='right') # Auto format date labels
                          plt.tight_layout() # Adjust layout
                          st.pyplot(fig_line, clear_figure=True) # Display plot

                     except AttributeError as attr_err:
                          st.warning(f"月度趋势图的日期格式无法被正确处理以进行绘图: {attr_err}")
                     except Exception as e:
                          st.error(f"绘制月度趋势图时出错: {e}")
                          if fig_line is not None: plt.close(fig_line) # Close plot if error occurred during processing
                 else:
                     st.caption("无足够数据绘制月度销售趋势。")

             # --- Sales Share (Pie Chart) ---
             with chart_cols[1]:
                 st.markdown("###### 期间销量占比 Top 10")
                 top_data = metrics.get('top_selling_period_chart_data')
                 # Check if data is valid Series and has a positive sum
                 if isinstance(top_data, pd.Series) and not top_data.empty and top_data.sum() > 0:
                     fig_pie = None # Initialize figure variable
                     try:
                          pie_labels = [str(label) for label in top_data.index] # Ensure labels are strings
                          pie_values = top_data.values
                          fig_pie, ax_pie = plt.subplots(figsize=(5, 3)) # Adjust figure size if needed

                          # Define text properties, considering font availability
                          text_props = {'fontproperties': chinese_font, 'size': 8} if FONT_AVAILABLE and chinese_font else {'size': 8}
                          legend_props = {'prop': chinese_font} if FONT_AVAILABLE and chinese_font else {}
                          title_fontprop = chinese_font if FONT_AVAILABLE and chinese_font else None

                          # Use a suitable colormap
                          colors = plt.get_cmap('Pastel1').colors

                          wedges, texts, autotexts = ax_pie.pie(
                              pie_values,
                              autopct='%1.1f%%', # Format percentage
                              startangle=90,
                              pctdistance=0.85, # Distance of percentage text from center
                              colors=colors,
                              textprops=text_props # Apply font props to percentage labels
                          )

                          # Set font for autotexts (percentages) explicitly if font is available
                          if FONT_AVAILABLE and chinese_font:
                             plt.setp(autotexts, fontproperties=chinese_font)

                          ax_pie.axis('equal') # Equal aspect ratio ensures pie is drawn as a circle.

                          # Add legend
                          legend_title = "产品名称"
                          # Position legend outside the pie
                          ax_pie.legend(wedges, pie_labels,
                                        title=legend_title,
                                        loc="center left",
                                        bbox_to_anchor=(1.05, 0, 0.5, 1), # Adjust anchor to position legend
                                        fontsize=8,
                                        prop=legend_props.get('prop'), # Font for legend items
                                        title_fontproperties=title_fontprop) # Font for legend title

                          plt.subplots_adjust(left=0.1, right=0.65) # Adjust subplot to make space for legend
                          st.pyplot(fig_pie, clear_figure=True) # Display plot

                     except Exception as e:
                          st.error(f"绘制销量占比图时出错: {e}")
                          if fig_pie is not None: plt.close(fig_pie) # Close plot on error
                 else:
                     st.caption("无足够数据绘制销量占比图。")
        elif not main_analysis_ready:
            st.info("请在左侧上传有效的 **百货城数据** 文件以查看销售分析。")
            if st.session_state.main_load_error:
                st.error(f"文件加载失败: {st.session_state.main_load_error}")


    # --- Inventory Analysis Tab ---
    with tab_inventory:
        if main_analysis_ready and isinstance(stock_analysis, pd.DataFrame) and not stock_analysis.empty:
             st.subheader("库存健康度与账龄分析")
             
             # 滞销筛选开关
             col1, col2 = st.columns(2)
             with col1:
                 show_dead_stock = st.checkbox("仅显示滞销产品", value=False)
             with col2:
                 dead_stock_level = st.selectbox(
                     "滞销等级筛选",
                     ["全部", "高", "中", "低"],
                     index=0
                 )
             
             # 应用筛选
             if show_dead_stock or dead_stock_level != "全部":
                 stock_analysis = stock_analysis.copy()  # Avoid modifying original
                 if show_dead_stock:
                     stock_analysis = stock_analysis[stock_analysis["滞销等级"] != "低"]
                 if dead_stock_level != "全部":
                     stock_analysis = stock_analysis[stock_analysis["滞销等级"] == dead_stock_level]
             # --- 多维度筛选器 ---
             st.markdown("### 🔍 多维度筛选")
             filter_col1, filter_col2, filter_col3 = st.columns(3)
             
             with filter_col1:
                 # 产品分类筛选
                 if "产品分类" in stock_analysis.columns:
                     categories = ["全部"] + sorted(stock_analysis["产品分类"].dropna().unique().tolist())
                     selected_category = st.selectbox("产品分类", categories, index=0)
                     if selected_category != "全部":
                         stock_analysis = stock_analysis[stock_analysis["产品分类"] == selected_category]
             
             with filter_col2:
                 # 压货时间段筛选
                 age_ranges = ["全部", "0-30天", "31-60天", "61-90天", "91-180天", "181+天"]
                 selected_age = st.selectbox("压货时间段", age_ranges, index=0)
                 if selected_age != "全部":
                     if "+" in selected_age:
                         min_days = int(selected_age.split("+")[0])
                         stock_analysis = stock_analysis[stock_analysis["压货时间_天"] >= min_days]
                     else:
                         min_days, max_days = map(int, selected_age.split("天")[0].split("-"))
                         stock_analysis = stock_analysis[
                             (stock_analysis["压货时间_天"] >= min_days) &
                             (stock_analysis["压货时间_天"] <= max_days)
                     ]
             
             with filter_col3:
                 # 库存天数区间筛选
                 stock_days_ranges = ["全部", "0-7天", "7-30天", "30+天"]
                 selected_stock_days = st.selectbox("库存天数区间", stock_days_ranges, index=0)
                 if selected_stock_days != "全部":
                     if "+" in selected_stock_days:
                         min_days = int(selected_stock_days.split("+")[0])
                         stock_analysis = stock_analysis[stock_analysis["当前库存"] >= min_days]
                     else:
                         min_days, max_days = map(int, selected_stock_days.split("-")[0].split("天")[0].split("-"))
                         stock_analysis = stock_analysis[
                             (stock_analysis["当前库存"] >= min_days) &
                             (stock_analysis["当前库存"] <= max_days)
                         ]
             
             st.markdown("---")
             # --- Stock Aging Distribution (Bar Chart) ---
             st.markdown("###### 库存账龄分布 (按 SKU 数)")
             # Helper function for bucketing (can be defined inside or outside)
             def get_age_bucket(days):
                 try:
                      if pd.isna(days): return "未知"
                      days_int = int(float(days)) # Ensure it's treated as number
                 except (ValueError, TypeError, OverflowError):
                      return "未知" # Handle non-numeric gracefully

                 if days_int == 9999: return "从未售出" # Special code for never sold / no last sale date
                 elif days_int <= 30: return "0-30 天"
                 elif days_int <= 60: return "31-60 天"
                 elif days_int <= 90: return "61-90 天"
                 elif days_int <= 180: return "91-180 天"
                 else: return "181+ 天"

             try:
                 if '压货时间_天' in stock_analysis.columns:
                     # Apply the bucketing function
                     stock_analysis['库存账龄分组'] = stock_analysis['压货时间_天'].apply(get_age_bucket)
                     # Define the desired order for the categories (从小到大)
                     age_order = ["0-30 天", "31-60 天", "61-90 天", "91-180 天", "181+ 天", "从未售出", "未知"]
                     # Sort the data before plotting
                     aging_data_sku = stock_analysis.groupby('库存账龄分组', observed=False).size()
                     aging_data_sku = aging_data_sku.reindex(age_order).fillna(0)
                     aging_data_sku = aging_data_sku[aging_data_sku > 0]
                     # Group by the new category, count SKUs, reindex to enforce order, fill missing groups with 0
                     aging_data_sku = stock_analysis.groupby('库存账龄分组', observed=False).size().reindex(age_order).fillna(0)
                     # Filter out categories with zero count for cleaner chart
                     aging_data_sku = aging_data_sku[aging_data_sku > 0]

                     if not aging_data_sku.empty:
                          # Use Streamlit's built-in bar chart
                          st.bar_chart(aging_data_sku, use_container_width=True)
                          st.caption("库存账龄根据产品最后销售日期计算。'从未售出'表示无销售记录或无法确定最后销售日期。")
                     else:
                          st.caption("无有效的库存账龄数据可供绘制。")
                 else:
                     st.warning("无法计算库存账龄分布，缺少 '压货时间_天' 数据。")
             except Exception as e:
                 st.error(f"生成库存账龄图表时出错: {e}")
                 traceback.print_exc()

             # --- Inventory Details Table ---
             st.markdown("---")
             st.markdown("###### 库存明细与状态")
             # Sort the full dataframe (e.g., by stock age descending)
             stock_analysis_sorted = stock_analysis.sort_values("压货时间_天", ascending=False) if '压货时间_天' in stock_analysis.columns else stock_analysis
             # Limit display rows
             stock_analysis_display_limited = stock_analysis_sorted.head(TOP_N_DISPLAY)
             st.info(f"💡 下表显示库存详细信息（最多显示前 {TOP_N_DISPLAY} 条记录，按压货天数降序排列）。完整数据可下载。")

             # Configure columns for st.data_editor
             dynamic_stock_config = {}
             base_stock_configs = {
                 "产品名称": st.column_config.TextColumn("产品名称", width="medium", help="产品名称"),
                 "产品分类": st.column_config.TextColumn("分类", width="small", help="产品所属分类"),
                 "当前库存": st.column_config.NumberColumn("当前库存", format="%d 个", help="当前实际库存数量"),
                 "期间销售量": st.column_config.NumberColumn("期间销售量", format="%d 个", help="所选分析周期内的总销售数量"),
                 "期间销售额": st.column_config.NumberColumn("期间销售额 (€)", format="%.2f", help="所选分析周期内的总销售额"), # 新增
                 "期间销售成本": st.column_config.NumberColumn("期间销售成本 (€)", format="%.2f", help="期间销售量 * 采购价"), # 新增
                 "期间毛利": st.column_config.NumberColumn("期间毛利 (€)", format="%.2f", help="期间销售额 - 期间销售成本"), # 新增
                 "毛利率": st.column_config.NumberColumn("毛利率", format="%.1f%%", help="(期间毛利 / 期间销售额) * 100"), # 新增
                 "期间日均销量": st.column_config.NumberColumn("期间日均销售", format="%.2f 个/天", help="所选分析周期内的平均每日销售数量"),
                 "预计可用天数": st.column_config.NumberColumn("预计可用天数", help="当前库存预计可维持天数 (9999代表>9999天或无近期销量)", format="%d 天"),
                 "压货时间_天": st.column_config.NumberColumn("压货天数", help="自上次售出至今的天数 (9999代表从未售出或无记录)", format="%d 天"),
                 "最后销售日期": st.column_config.DateColumn("最后销售日期", format="YYYY-MM-DD", help="该产品最后一次有销售记录的日期"),
                 "天数自上次采购": st.column_config.NumberColumn("距上次采购", help="自上次采购至今的天数 (9999代表无采购记录)", format="%d 天"),
                 "最后采购日期": st.column_config.DateColumn("最后采购日期", format="YYYY-MM-DD", help="该产品最后一次有采购记录的日期"),
                 "最后采购数量": st.column_config.NumberColumn("最后采购数量", format="%d 个", help="最后一次采购的数量"),
                 "采购价": st.column_config.NumberColumn("采购价 (€)", format="%.2f", help="库存数据中记录的采购单价"),
             }

             # Define which columns to show and in what order
             stock_cols_to_show_final = ["产品名称"]
             if has_category_data: stock_cols_to_show_final.append("产品分类")
             # 添加毛利相关列到显示列表
             stock_cols_to_show_final.extend([
                 "当前库存", "期间销售量", "期间销售额", "期间销售成本", "期间毛利", "毛利率",
                 "期间日均销量", "预计可用天数", "压货时间_天",
                 "最后销售日期", "天数自上次采购", "最后采购日期", "最后采购数量", "采购价"
             ])

             # Filter columns to only those present in the dataframe and configure them
             final_stock_cols = []
             for col_name in stock_cols_to_show_final:
                  if col_name in stock_analysis_display_limited.columns:
                       final_stock_cols.append(col_name)
                       if col_name in base_stock_configs:
                           dynamic_stock_config[col_name] = base_stock_configs[col_name]
                       #else: # Add default config for columns not explicitly defined? Optional.
                           #dynamic_stock_config[col_name] = st.column_config.TextColumn(col_name)


             # Display the data editor table
             st.data_editor(
                 stock_analysis_display_limited[final_stock_cols], # Use filtered columns
                 use_container_width=True,
                 num_rows="fixed", # Fixed height based on TOP_N_DISPLAY
                 disabled=True, # Make table read-only
                 key="stock_data_editor",
                 # 格式化毛利率为百分比
                 column_config={**dynamic_stock_config, "毛利率": st.column_config.NumberColumn("毛利率", format="%.1f%%")},
                 hide_index=True
             )
             st.caption(f"注：表格仅显示排序后的前 {len(stock_analysis_display_limited)} 条记录 (共 {len(stock_analysis_sorted)} 条)。")

             # --- Download Button for Full Inventory Analysis ---
             try:
                 # Select columns for download (exclude temporary/grouping cols)
                 stock_cols_download = [col for col in stock_analysis_sorted.columns if col not in ['库存账龄分组']]
                 # Ensure columns exist
                 stock_cols_download = [col for col in stock_cols_download if col in stock_analysis_sorted.columns]

                 df_to_download_stock = stock_analysis_sorted[stock_cols_download]
                 # Prepare Excel file in memory
                 excel_buffer_stock = io.BytesIO()
                 with pd.ExcelWriter(excel_buffer_stock, engine='openpyxl') as writer:
                     df_to_download_stock.to_excel(writer, index=False, sheet_name='库存分析详细数据')
                 excel_buffer_stock.seek(0) # Rewind buffer

                 # Generate filename
                 download_filename_stock = f"库存分析_{selected_category}_{start_date_dt.strftime('%Y%m%d')}-{end_date_dt.strftime('%Y%m%d')}.xlsx"

                 # Create download button
                 st.download_button(
                     label=f"📥 下载完整库存分析表 ({len(stock_analysis_sorted)}条)",
                     data=excel_buffer_stock,
                     file_name=download_filename_stock,
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     key="download_stock_analysis"
                 )
             except Exception as e:
                 st.error(f"生成库存分析 Excel 下载文件时出错: {e}")
                 traceback.print_exc()

        elif not main_analysis_ready:
             st.info("请在左侧上传有效的 **百货城数据** 文件以查看库存分析。")
             if st.session_state.main_load_error:
                  st.error(f"文件加载失败: {st.session_state.main_load_error}")
        else: # main_analysis_ready is True, but stock_analysis is empty or invalid
             st.info("无详细库存数据可供分析。请检查上传的“百货城数据”文件中的 '库存数据' 或筛选条件。")


    # --- Purchase Suggestions Tab ---
    with tab_purchase:
         st.subheader("智能采购建议")
         if main_analysis_ready and isinstance(purchase_suggestions, pd.DataFrame) and not purchase_suggestions.empty:
             # Get selected prediction method from session state
             predict_method = st.session_state.get("predict_method_key", "exponential_smoothing")
             # Data is ready and suggestions exist
             purchase_suggestions_full = purchase_suggestions # Keep the full dataframe
             purchase_suggestions_display_limited = purchase_suggestions_full.head(TOP_N_DISPLAY) # Limit for display

             st.info(f"💡 下表显示建议采购的产品（最多显示前 {TOP_N_DISPLAY} 条，按建议采购量降序排列）。完整建议可下载。")

             # Configure columns for display
             dynamic_purchase_config = {}
             base_purchase_configs = {
                  "产品名称": st.column_config.TextColumn("产品名称", width="medium"),
                  "产品分类": st.column_config.TextColumn("分类", width="small"),
                  "当前库存": st.column_config.NumberColumn("当前库存", format="%d 个"),
                  "期间日均销量": st.column_config.NumberColumn("期间日均销售", format="%.2f 个/天"),
                  "预计可用天数": st.column_config.NumberColumn("当前可用天数", format="%d 天", help="基于当前库存和期间日均销量的估算"),
                  "目标库存水平": st.column_config.NumberColumn("目标库存", help=f"目标({target_days_input}天)+安全({safety_days_input}天)所需库存量", format="%.0f 个"),
                  "建议采购量": st.column_config.NumberColumn("建议采购量", format="%d 个", width="large", help="建议补充的数量 (已向上取整)")
             }

             # Determine columns to show based on what's available in purchase_suggestions_display_limited
             purchase_cols_to_show = []
             for col_name in purchase_suggestions_display_limited.columns:
                  if col_name in base_purchase_configs:
                       purchase_cols_to_show.append(col_name)
                       dynamic_purchase_config[col_name] = base_purchase_configs[col_name]
                  # Add handling for unexpected columns if necessary

             # Display the data editor table
             st.data_editor(
                 purchase_suggestions_display_limited[purchase_cols_to_show],
                 use_container_width=True,
                 num_rows="fixed",
                 disabled=True, # Read-only
                 key="purchase_data_editor",
                 column_config=dynamic_purchase_config,
                 hide_index=True
             )
             st.caption(f"注：表格仅显示建议采购量最多的前 {len(purchase_suggestions_display_limited)} 条建议 (共 {len(purchase_suggestions_full)} 条)。")
             
             # --- Download Button for Full Purchase Suggestions ---
             try:
                 df_to_download_purchase = purchase_suggestions_full # Use the full dataframe
                 excel_buffer_purchase = io.BytesIO()
                 with pd.ExcelWriter(excel_buffer_purchase, engine='openpyxl') as writer:
                     df_to_download_purchase.to_excel(writer, index=False, sheet_name='采购建议清单')
                 excel_buffer_purchase.seek(0)

                 # Generate filename including parameters
                 download_filename_purchase = f"采购建议_{selected_category}_{start_date_dt.strftime('%Y%m%d')}-{end_date_dt.strftime('%Y%m%d')}_T{target_days_input}S{safety_days_input}.xlsx"

                 st.download_button(
                     label=f"📤 下载完整采购建议 ({len(purchase_suggestions_full)}条)",
                     data=excel_buffer_purchase,
                     file_name=download_filename_purchase,
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     key="download_purchase_suggestions"
                 )
             except Exception as e:
                 st.error(f"生成采购建议 Excel 下载文件时出错: {e}")
                 traceback.print_exc()

         elif main_analysis_ready and isinstance(purchase_suggestions, pd.DataFrame) and purchase_suggestions.empty:
             # Analysis ran, but no suggestions were generated
             st.success("✅ 根据当前库存和所选参数，暂无产品需要立即采购。")
         elif not main_analysis_ready:
             # Main data wasn't loaded successfully
             pass

             # --- 财务对账 Tab ---
             if 'tab_reconciliation' in locals():
                 with tab_reconciliation:
                     st.subheader("财务对账")
                
                     if 'order_df' in locals() and not order_df.empty:
                         # 检查财务数据是否上传
                         if 'finance_df' not in locals() or finance_df.empty:
                             st.warning("请先上传财务数据文件")
                         else:
                             # 核心对账逻辑
                             try:
                                 # 1. 按订单编号精确匹配
                                 reconciled = order_df.merge(
                                     finance_df,
                                     on='订单编号',
                                     how='left',
                                     suffixes=('_销售', '_财务')
                                 )
                                 
                                 # 2. 对未匹配的记录尝试客户+金额近似匹配
                                 unmatched = reconciled[reconciled['实收金额'].isna()]
                                 if not unmatched.empty:
                                     for idx, row in unmatched.iterrows():
                                         match = finance_df[
                                             (finance_df['客户名称'] == row['客户']) &
                                             (abs(finance_df['金额'] - row['销售额']) < 0.01)
                                         ].head(1)
                                         if not match.empty:
                                             reconciled.loc[idx, '实收金额'] = match.iloc[0]['金额']
                                             
                                 # 计算统计指标
                                 total_sales = reconciled['销售额'].sum()
                                 total_received = reconciled['实收金额'].sum()
                                 payment_rate = (total_received / total_sales) * 100 if total_sales > 0 else 0
                                 
                                 # 显示结果
                                 col1, col2, col3 = st.columns(3)
                                 col1.metric("总销售额", f"{total_sales:,.2f}元")
                                 col2.metric("已回款", f"{total_received:,.2f}元")
                                 col3.metric("回款率", f"{payment_rate:.1f}%")
                                 
                                 # 应收账款明细
                                 with st.expander("应收账款明细"):
                                     overdue = reconciled[reconciled['实收金额'].isna()]
                                     st.dataframe(overdue)
                                     
                             except Exception as e:
                                 st.error(f"对账过程中发生错误: {str(e)}")
             st.info("请先在左侧上传有效的 **百货城数据** 文件以生成采购建议。")
             if st.session_state.main_load_error:
                  st.error(f"文件加载失败: {st.session_state.main_load_error}")
         else:
             # Catchall for other states (e.g., purchase suggestion calculation failed)
             st.warning("无法生成采购建议。请检查库存分析数据和参数。")


    # --- NEW: Financial Metrics Tab ---
    with tab_financial:
        st.subheader("💰 财务指标概览")
        if financial_data_ready and isinstance(financial_data_loaded, pd.DataFrame):
            st.success("财务数据已加载。")
            st.markdown("_(此处将显示关键财务指标，例如总收入、总支出、净利润、应收/应付账款等)_")
            st.dataframe(financial_data_loaded.head(), use_container_width=True) # Display sample data
            # TODO: Implement actual financial metric calculations and display
        elif uploaded_financial_file and not financial_data_ready:
            st.warning("财务数据文件已上传，但加载或处理失败。请检查侧边栏错误信息。")
        else:
            st.info("请在左侧上传 **财务数据** 文件以查看此模块。")

    # --- NEW: CRM Summary Tab ---
    with tab_crm:
        st.subheader("👥 CRM 摘要")
        if crm_data_ready and isinstance(crm_data_loaded, pd.DataFrame):
            st.success("CRM 数据已加载。")
            st.markdown("_(此处将显示 CRM 相关摘要，例如新增潜在客户、客户活动概览等)_")
            st.dataframe(crm_data_loaded.head(), use_container_width=True) # Display sample data
            # TODO: Implement actual CRM metric calculations and display
        elif uploaded_crm_file and not crm_data_ready:
            st.warning("CRM 数据文件已上传，但加载或处理失败。请检查侧边栏错误信息。")
        else:
            st.info("请在左侧上传 **CRM 数据** 文件以查看此模块。")


    # --- NEW: Alerts & Tasks Tab ---
    with tab_alerts:
        st.subheader("🔔 待办事项与提醒")
        st.markdown("_(此区域将整合关键提醒，例如低库存预警、建议采购项目等)_")
        # Example: Display low stock items (requires modification in calculate_metrics or here)
        if main_analysis_ready and isinstance(stock_analysis, pd.DataFrame) and not stock_analysis.empty and '预计可用天数' in stock_analysis.columns:
            low_stock_threshold_days = 7 # Example threshold
            low_stock_items = stock_analysis[stock_analysis['预计可用天数'] <= low_stock_threshold_days]
            if not low_stock_items.empty:
                st.warning(f"⚠️ **低库存预警** (预计可用天数 <= {low_stock_threshold_days} 天):")
                st.dataframe(low_stock_items[['产品名称', '当前库存', '预计可用天数']].head(10), use_container_width=True, hide_index=True)
            else:
                st.success("✅ 当前无明显低库存风险。")
        else:
            st.info("需要加载有效的 **百货城数据** 以生成库存预警。")

        # Example: Display purchase suggestions summary
        if main_analysis_ready and isinstance(purchase_suggestions, pd.DataFrame) and not purchase_suggestions.empty:
             st.info(f"🛒 **采购建议提醒**: {len(purchase_suggestions)} 个产品建议采购。详情请见 '采购建议' 标签页。")
        # TODO: Add other potential alerts (e.g., overdue tasks if CRM data is available)

    # --- NEW: Custom Analysis Tab ---
    with tab_custom_analysis:
        st.subheader("📈 自定义分析 (占位符)")
        st.info("此功能正在开发中。未来将允许您基于已加载的数据进行更灵活的探索和报表生成。")
        # Placeholder for future features like:
        # - Selecting data source (Sales, Stock, Finance, etc.)
        # - Choosing columns for grouping/aggregation
        # - Selecting chart types
        # - Saving custom views


# Fallback message if file upload was attempted but processing failed for *both* types (or only one was attempted and failed)
# This might be redundant now with errors shown in sidebar/tabs, but can be a final catch-all. Check all potential uploads.
elif any([(uploaded_main_file and not main_analysis_ready),
          (uploaded_financial_file and not financial_data_ready), # Check new states
          (uploaded_crm_file and not crm_data_ready)]):             # Check new states
    st.error("❌ 部分上传的文件处理失败或包含无效数据。请检查侧边栏的错误信息，并根据提示检查文件格式和内容。")


# Footer (appears only if any file was uploaded or attempted)
if any([uploaded_main_file, uploaded_financial_file, uploaded_crm_file]): # Ensure footer shows if any file is uploaded
    st.markdown("---")
    try:
        current_year = datetime.now(APP_TIMEZONE).year
    except NameError: # Fallback if APP_TIMEZONE wasn't defined early
        current_year = datetime.utcnow().year
    st.markdown( f"""<div style='text-align: center; font-size: 14px; color: gray;'> TP.STER 智能数据平台 {APP_VERSION} @ {current_year}</div>""", unsafe_allow_html=True)


# --- 销量预测功能 ---
def predict_sales(sales_df, forecast_days=30, method="exponential_smoothing"):
    """预测未来销量
    
    参数:
        sales_df (DataFrame): 历史销售数据
        forecast_days (int): 预测天数
        method (str): 预测方法 ('moving_average' 或 'exponential_smoothing')
    
    返回:
        DataFrame: 包含预测结果的DataFrame
    """
    try:
        # 根据选择的方法调用不同的预测逻辑
        if method == "moving_average":
            # 简单移动平均法
            window_size = 7  # 7天移动平均
            sales_df['预测销量'] = sales_df['销量'].rolling(window=window_size).mean()
            sales_df['30天预测总销量'] = sales_df['预测销量'] * forecast_days
        else:
            # 默认使用指数平滑法
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # 按产品和日期聚合销量
        daily_sales = sales_df.groupby(['产品ID', '订单日期'])['销售数量'].sum().reset_index()
        
        predictions = []
        for product_id, group in daily_sales.groupby('产品ID'):
            # 准备时间序列数据
            ts_data = group.set_index('订单日期')['销售数量'].sort_index()
            
            # 填充缺失日期为0
            idx = pd.date_range(ts_data.index.min(), ts_data.index.max())
            ts_data = ts_data.reindex(idx, fill_value=0)
            
            # 训练模型
            model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=7).fit()
            
            # 预测未来销量
            forecast = model.forecast(forecast_days)
            
            predictions.append({
                '产品ID': product_id,
                '产品名称': group['产品名称'].iloc[0],
                '日均预测销量': round(forecast.mean(), 2),
                '30天预测总销量': round(forecast.sum(), 2)
            })
            
        return pd.DataFrame(predictions)
        
    except Exception as e:
        st.error(f"销量预测失败: {str(e)}")
        return None

# --- END OF SCRIPT ---