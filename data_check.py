import pandas as pd
import streamlit as st
from typing import Optional

def fix_column_names(df: pd.DataFrame, expected_columns: dict) -> pd.DataFrame:
    """
    自动修复数据框中的列名不一致问题
    参数:
        df: 要修复的数据框
        expected_columns: 期望列名映射，如 {'销售数量': ['购买数量', 'quantity']}
    返回:
        修复后的数据框
    """
    for target_col, alt_names in expected_columns.items():
        for alt in alt_names:
            if alt in df.columns and target_col not in df.columns:
                df = df.rename(columns={alt: target_col})
                st.success(f"✅ 自动将列名 '{alt}' 修正为 '{target_col}'")
                break
    return df

def fill_missing_categories(
    target_df: pd.DataFrame, 
    source_df: pd.DataFrame,
    product_id_col: str = "产品ID",
    category_col: str = "产品分类"
) -> tuple[pd.DataFrame, int]:
    """
    从源数据框补全目标数据框缺失的产品分类
    参数:
        target_df: 需要补全分类的目标数据框
        source_df: 包含完整分类信息的源数据框
        product_id_col: 产品ID列名
        category_col: 产品分类列名
    返回:
        (修复后的数据框, 补全的分类数量)
    """
    if category_col not in target_df.columns:
        target_df[category_col] = None
    
    # 创建产品ID到分类的映射
    category_map = source_df.drop_duplicates(product_id_col).set_index(product_id_col)[category_col].to_dict()
    
    # 找出需要补全的行
    missing_mask = target_df[category_col].isna() & target_df[product_id_col].notna()
    fixed_count = missing_mask.sum()
    
    if fixed_count > 0:
        # 补全分类
        target_df.loc[missing_mask, category_col] = (
            target_df.loc[missing_mask, product_id_col].map(category_map)
        )
        st.success(f"✅ 自动补全了 {fixed_count} 条缺失的产品分类")
    
    return target_df, fixed_count

def calculate_missing_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算缺失的销售额(单价*数量)
    参数:
        df: 要处理的数据框
    返回:
        处理后的数据框
    """
    if "销售额" not in df.columns and all(col in df.columns for col in ["单价", "销售数量"]):
        df["销售额"] = df["单价"] * df["销售数量"]
        st.success("✅ 自动计算并添加了销售额(单价×数量)")
    return df

def run_data_checks(
    main_stock_data: pd.DataFrame,
    main_sales_data: pd.DataFrame,
    financial_data: Optional[pd.DataFrame] = None,
    crm_data: Optional[pd.DataFrame] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    执行所有数据检查和修复
    返回修复后的数据框(顺序与输入一致)
    """
    # 定义期望的列名映射
    expected_columns = {
        "销售数量": ["购买数量", "quantity", "销售件数"],
        "产品分类": ["分类", "category", "产品类别"],
        "产品ID": ["SKU", "商品ID", "product_id"]
    }
    
    # 修复列名
    main_stock_data = fix_column_names(main_stock_data, expected_columns)
    main_sales_data = fix_column_names(main_sales_data, expected_columns)
    
    if financial_data is not None:
        financial_data = fix_column_names(financial_data, expected_columns)
    if crm_data is not None:
        crm_data = fix_column_names(crm_data, expected_columns)
    
    # 补全产品分类(从库存表补全到销售表)
    if "产品ID" in main_sales_data.columns and "产品ID" in main_stock_data.columns:
        main_sales_data, _ = fill_missing_categories(main_sales_data, main_stock_data)
    
    # 计算缺失的销售额
    main_sales_data = calculate_missing_sales(main_sales_data)
    
    return main_stock_data, main_sales_data, financial_data, crm_data