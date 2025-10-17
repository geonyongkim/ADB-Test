import streamlit as st
import pandas as pd
import numpy as np
from nptdms import TdmsFile
import plotly.express as px
import plotly.graph_objects as go
import io
import re
import os

# --- 1. Configuration ---
# VBScript에서 추출한 코스별 조도계 원점 좌표
ZERO_MAP = {
    "100L": (357.220, 0.864),
    "230L": (357.645, 1.076),
    "230R": (360.405, 1.031),
    "370L": (357.850, 1.174),
    "370R": (360.202, 1.110),
    "230LP": (357.645, 1.076), # Preceding
    "ST": (None),  # Oncoming - PosLocalX 기준
    "STP": None, # Preceding - PosLocalX 기준
}

ILLUMINANCE_SAMPLING_RATE = 201  # Hz
DISTANCE_GRID_INTERVAL = 0.1  # meters

# 고정 시간 오프셋
TIME_OFFSET_SECONDS = 32381

# --- 2. Modular Functions ---

def find_and_rename_columns(df, column_map):
    """DataFrame에서 지정된 패턴의 컬럼을 찾아 표준 이름으로 변경합니다."""
    rename_dict = {}
    for standard_name, pattern in column_map.items():
        for col in df.columns:
            if re.search(pattern, col, re.IGNORECASE):
                rename_dict[col] = standard_name
                break
    return df.rename(columns=rename_dict)

def load_vehicle_csv(uploaded_file):
    """주행 데이터 CSV 파일을 로드하고 Timestamp 생성 및 컬럼 표준화를 수행합니다."""
    if uploaded_file is None: return None
    try:
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp949')
            st.info("CSV 파일을 'cp949' 인코딩으로 다시 읽었습니다.")

        df_processed = df.copy()

        # 주요 컬럼 표준화
        column_map = {
            "Date": r"Date.*",
            "Time": r"Time.*\(UTC",
            "TimeFromStart": r"TimeFromStart.*",
            "PosLocalX": r"PosLocalX.*",
            "PosLocalY": r"PosLocalY.*",
            "Pitch": r"AnglePitch.*", # Pitch 데이터 추가
        }
        df_processed = find_and_rename_columns(df_processed, column_map)

        # Timestamp 생성
        if "Date" in df_processed.columns and "Time" in df_processed.columns:
            df_processed["Timestamp"] = pd.to_datetime(df_processed["Date"] + " " + df_processed["Time"], errors='coerce')
        else:
            st.warning("CSV 파일에 날짜/시간 정보가 없어 Timestamp를 생성할 수 없습니다.")
            return None
        
        st.success("주행 데이터(CSV) 로드 및 전처리 성공")
        return df_processed

    except Exception as e:
        st.error(f"CSV 파일 처리 오류: {e}")
        return None

def load_illuminance_tdms(uploaded_file):
    """조도 TDMS 파일을 로드하고 채널 표준화 및 Timestamp를 생성합니다."""
    if uploaded_file is None: return None
    try:
        with io.BytesIO(uploaded_file.getvalue()) as buffer:
            tdms_file = TdmsFile(buffer)
        
        all_channels = tdms_file.as_dataframe()

        # 표준 조도 채널 DataFrame 생성
        standard_channels = [f"Middle Gain {i}" for i in range(1, 11)]
        df_illuminance = pd.DataFrame()

        # 시간 데이터 추출
        time_data = None
        timestamp_col_name = None

        # 1. 'timestamp' 이름의 채널을 먼저 찾아봅니다.
        for col in all_channels.columns:
            if 'timestamp' in col.lower():
                timestamp_col_name = col
                break
        
        if timestamp_col_name:
            st.info(f"'{timestamp_col_name}' 채널을 시간 정보로 사용합니다.")
            time_data = all_channels[timestamp_col_name]
        else:
            # 2. 'timestamp' 채널이 없으면 기존의 time_track() 방식을 시도합니다.
            st.info("'timestamp' 채널을 찾지 못했습니다. time_track() 메서드를 시도합니다.")
            for group in tdms_file.groups():
                for channel in group.channels():
                    try:
                        time_data = channel.time_track()
                        break
                    except KeyError:
                        continue
                if time_data is not None:
                    break
        
        if time_data is None:
            st.error("TDMS 파일에서 시간 채널을 찾을 수 없습니다.")
            return None
            
        df_illuminance['Timestamp'] = pd.to_datetime(time_data)

        # Middle Gain 채널 탐지 및 표준화
        found_channels = 0
        for i in range(1, 11):
            standard_name = f"Middle Gain {i}"
            # 다양한 채널 이름 형식에 대응하기 위한 정규식
            pattern = re.compile(f".*mid(dle)?.*gain.*{i}.*", re.IGNORECASE)
            
            found = False
            for col in all_channels.columns:
                if pattern.match(col):
                    df_illuminance[standard_name] = all_channels[col]
                    found = True
                    found_channels += 1
                    break
            if not found:
                df_illuminance[standard_name] = np.nan # 채널이 없으면 NaN으로 채움

        if found_channels == 0:
            st.warning("TDMS 파일에서 'Middle Gain' 채널을 찾지 못했습니다.")
        else:
            st.success(f"조도 데이터(TDMS) 로드 및 {found_channels}개 채널 표준화 성공")

        return df_illuminance
        
    except Exception as e:
        st.error(f"TDMS 파일 처리 오류: {e}")
        return None


def synchronize_data(df_vehicle, df_illuminance, offset):
    """차량과 조도 데이터를 시간 동기화합니다."""
    if df_vehicle is None or df_illuminance is None:
        return None

    df_v = df_vehicle.copy()
    df_i = df_illuminance.copy()

    df_i["Timestamp"] = df_i["Timestamp"] + pd.to_timedelta(offset, unit='s')
    st.info(f"{offset}초 오프셋 적용 완료.")

    df_v = df_v.sort_values("Timestamp").reset_index(drop=True)
    df_i = df_i.sort_values("Timestamp").reset_index(drop=True)

    st.info("가장 가까운 시간의 조도 데이터를 매칭합니다.")
    df_synced = pd.merge_asof(
        left=df_v,
        right=df_i,
        on="Timestamp",
        direction="nearest"
    )

    original_samples = len(df_v)
    st.info("병합 후 데이터 유효성 검사: 필수 좌표(PosLocalX/Y) 또는 모든 조도 채널이 없는 행을 제거합니다.")
    df_synced.dropna(subset=['PosLocalX', 'PosLocalY'], inplace=True)
    mid_gain_cols = [f'Middle Gain {i}' for i in range(1, 11)]
    df_synced.dropna(subset=mid_gain_cols, how='all', inplace=True)
    synced_samples = len(df_synced)
    
    st.success(f"데이터 동기화 완료. 유효 샘플: {synced_samples} / {original_samples}")
    
    if synced_samples < 2:
        st.error("동기화 후 데이터 샘플이 2개 미만입니다. 시간 오프셋 또는 입력 파일을 확인하세요.")
        return None

    return df_synced

def calculate_distance(df, zero_point):
    """조도계 원점 대비 차량의 직선거리를 계산합니다."""
    if df is None: return None
    df_processed = df.copy()
    if zero_point:
        x0, y0 = zero_point
        raw_distance = np.sqrt((df_processed['PosLocalX'] - x0)**2 + (df_processed['PosLocalY'] - y0)**2)
    else:
        # For straight sections (e.g., ST, STP) where zero_point is None,
        # calculate distance relative to PosLocalX = 359.
        raw_distance = (df_processed['PosLocalX'] - 359).abs()
    df_processed['Distance'] = raw_distance
    st.success("거리가 성공적으로 계산되었습니다.")
    return df_processed
def get_evaluation_ranges(course, scenario):
    """프롬프트에 명시된 코스별 평가 거리 범위를 반환합니다."""
    if scenario == 'Oncoming':
        if course == 'ST': return [(15, 220)]
        if course == '100L': return [(15, 59.9)]
        if course == '230L': return [(15, 150)]
        if course == '370L': return [(15, 220)]
        if course == '230R': return [(15, 59.9)]
        if course == '370R': return [(15, 70)]
    elif scenario == 'Preceding':
        if course in ['ST', 'STP']: return [(15, 100)]
        if course in ['230L', '230LP']: return [(15, 100)]
    return []

def filter_by_evaluation_range(df, course, scenario):
    """데이터를 프롬프트에 명시된 평가 구간에 따라 필터링합니다."""
    if df is None or df.empty:
        return None
    
    eval_ranges = get_evaluation_ranges(course, scenario)
    if not eval_ranges:
        st.warning(f"{course} - {scenario}에 대한 평가 구간이 정의되지 않았습니다. 전체 데이터를 사용합니다.")
        return df

    combined_mask = pd.Series(False, index=df.index)
    for start, end in eval_ranges:
        mask = df['Distance'].between(start, end, inclusive='both')
        combined_mask |= mask
    
    df_filtered = df[combined_mask].copy() 
    
    st.info(f"평가 구간 {eval_ranges} 내의 데이터만 사용합니다. (총 {len(df_filtered)}개 샘플)")
    
    if df_filtered.empty:
        st.error("지정된 평가 구간 내에 데이터가 없습니다. 주행 경로 또는 코스 설정을 확인하세요.")
        return None
        
    return df_filtered

def get_baseline(course, scenario, distance_axis):
    """'ADB 결과 분석 프롬프트.md' 기반으로 코스별 기준선(Upper Limit) 데이터를 생성합니다."""
    limit = pd.Series(np.nan, index=distance_axis.index)
    
    course_eval_ranges = get_evaluation_ranges(course, scenario)
    if not course_eval_ranges:
        return limit

    if scenario == 'Oncoming':
        criteria = {
            (15, 29.9): 3.1,
            (30, 59.9): 1.8,
            (60, 119.9): 0.6,
            (120, 220): 0.3
        }
    elif scenario == 'Preceding':
        criteria = {
            (30, 59.9): 18.9,
            (60, 119.9): 4.0
        }
    else:
        criteria = {}

    for course_start, course_end in course_eval_ranges:
        for (crit_start, crit_end), lx_val in criteria.items():
            apply_start = max(course_start, crit_start)
            apply_end = min(course_end, crit_end)
            
            if apply_start <= apply_end:
                limit[distance_axis.between(apply_start, apply_end, inclusive='both')] = lx_val
                
    return limit

def generate_report(df, course, scenario, fig_path=None, fig_curve=None, fig_pitch=None, csv_filename=None, tdms_filename=None, df_pitch_exceeded=None, pitch_stats=None):
    """분석 결과를 'report test.xlsx' 양식에 맞게 Excel 보고서로 생성하고, NG 행을 강조 표시합니다."""
    if df is None or df.empty:
        st.warning("보고서를 생성할 데이터가 없습니다.")
        return None, None

    # --- 1. Data Processing for Report Table ---
    df_sorted = df.sort_values('Distance').reset_index(drop=True)
    if df_sorted['Distance'].duplicated().any():
        df_sorted = df_sorted.groupby('Distance').mean(numeric_only=True).reset_index()

    min_dist, max_dist = df_sorted['Distance'].min(), df_sorted['Distance'].max()
    if pd.isna(min_dist) or pd.isna(max_dist):
        return None, None

    distance_grid = np.arange(np.floor(min_dist), np.ceil(max_dist), DISTANCE_GRID_INTERVAL)
    df_indexed = df_sorted.set_index('Distance')
    new_index = pd.Index(distance_grid, name='Distance')
    
    df_numeric_indexed = df_indexed.select_dtypes(include=np.number)
    df_gridded = df_numeric_indexed.reindex(df_numeric_indexed.index.union(new_index)) \
                                   .interpolate(method='index') \
                                   .reindex(new_index)
    df_gridded.reset_index(inplace=True)

    if 'UpperLimit' not in df_gridded.columns:
        df_gridded['UpperLimit'] = get_baseline(course, scenario, df_gridded['Distance'])

    report_data = []
    if scenario == 'Oncoming':
        mid_gain_cols = [f'Middle Gain {i}' for i in [4, 5]]
    elif scenario == 'Preceding':
        mid_gain_cols = [f'Middle Gain {i}' for i in range(6, 11)]
    else:
        mid_gain_cols = []

    # --- NG Row Identification for LAW DATA highlighting ---
    ng_row_mask = pd.Series(False, index=df_gridded.index)
    for col in mid_gain_cols:
        if col in df_gridded.columns:
            exceeds = (df_gridded[col] > df_gridded['UpperLimit']) & df_gridded['UpperLimit'].notna()
            ng_row_mask |= exceeds

    for col in mid_gain_cols:
        if col in df_gridded.columns and not df_gridded[col].isnull().all():
            measurement = df_gridded[col]
            limit = df_gridded['UpperLimit']
            
            eval_mask = limit.notna()
            eval_ranges = get_evaluation_ranges(course, scenario)
            course_total_eval_dist = sum(end - start for start, end in eval_ranges)

            exceeds = (measurement > limit) & eval_mask
            ng_points = exceeds.sum()
            ng_dist = ng_points * DISTANCE_GRID_INTERVAL

            # Calculate Exceed Rate
            exceed_rate = (ng_dist / course_total_eval_dist) * 100 if course_total_eval_dist > 0 else 0
            
            overall = "OK" if ng_points == 0 else "NG"
            
            worst_exceed, worst_dist, rate_at_worst_point = 0, np.nan, 0
            if overall == "NG":
                diff = measurement - limit
                worst_exceed_series = diff[exceeds]
                if not worst_exceed_series.empty:
                    worst_idx = worst_exceed_series.idxmax()
                    worst_exceed = worst_exceed_series.max()
                    worst_dist = df_gridded.loc[worst_idx, 'Distance']
                    
                    limit_at_worst_point = limit.loc[worst_idx]
                    if limit_at_worst_point > 0:
                        rate_at_worst_point = (worst_exceed / limit_at_worst_point) * 100
                    else:
                        rate_at_worst_point = np.inf # Handle case where limit is 0 to avoid division error

            report_data.append({
                "Channel": col, 
                "Overall (OK/NG)": overall, 
                "초과율 (%)": f"{exceed_rate:.2f}",
                "누적 NG 거리 (m)": f"{ng_dist:.2f}",
                "WorstExceed (lx)": f"{worst_exceed:.2f}", 
                "WorstDist (m)": f"{worst_dist:.2f}",
                "최대 초과지점 초과율 (%)": f"{rate_at_worst_point:.2f}"
            })
    df_report = pd.DataFrame(report_data)

    # --- 2. Create Excel Report using xlsxwriter ---
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        report_sheet = workbook.add_worksheet('Report')

        # --- Define Formats ---
        title_format = workbook.add_format({'bold': True, 'font_size': 20, 'align': 'center'})
        subtitle_format = workbook.add_format({'bold': True, 'font_size': 12})
        header_format = workbook.add_format({'bold': True, 'font_size': 11, 'align': 'center', 'valign': 'vcenter', 'bg_color': '#DDEBF7', 'border': 1})
        ok_format = workbook.add_format({'bold': True, 'font_size': 11, 'align': 'center', 'bg_color': '#C6EFCE', 'font_color': '#006100'})
        ng_format = workbook.add_format({'bold': True, 'font_size': 11, 'align': 'center', 'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        cell_format = workbook.add_format({'align': 'center'})
        info_header_format = workbook.add_format({'bold': True, 'align': 'right'})
        yellow_format = workbook.add_format({'bg_color': '#FFFF00'}) # Yellow for NG rows

        # --- Sheet 1: Formatted Report ---
        report_sheet.set_column('A:A', 25) # Adjusted for new explanation header
        report_sheet.set_column('B:G', 20)
        report_sheet.merge_range('A1:G1', 'ADB Test Report', title_format)

        # --- Test Information ---
        report_sheet.write('A3', 'Test Information', subtitle_format)
        report_sheet.write('A4', 'CSV File:', info_header_format)
        report_sheet.write('B4', csv_filename)
        report_sheet.write('A5', 'TDMS File:', info_header_format)
        report_sheet.write('B5', tdms_filename)
        report_sheet.write('A6', 'Course:', info_header_format)
        report_sheet.write('B6', course)
        report_sheet.write('A7', 'Scenario:', info_header_format)
        report_sheet.write('B7', scenario)

        # --- Add user-editable fields ---
        report_sheet.write('D4', '시험 일자:', info_header_format)
        report_sheet.write('D5', '평가자:', info_header_format)
        report_sheet.write('D6', '차종명:', info_header_format)
        editable_format = workbook.add_format({'bg_color': '#F2F2F2', 'border': 1})
        report_sheet.write('E4', '', editable_format)
        report_sheet.write('E5', '', editable_format)
        report_sheet.write('E6', '', editable_format)

        # --- Pitch Statistics ---
        if pitch_stats:
            pitch_info_start_row = report_sheet.dim_row + 2
            report_sheet.write(f'A{pitch_info_start_row}', 'Pitch Statistics', subtitle_format)
            report_sheet.write(f'A{pitch_info_start_row + 1}', 'Avg Pitch (deg):', info_header_format)
            report_sheet.write(f'B{pitch_info_start_row + 1}', f"{pitch_stats['avg']:.3f}")
            report_sheet.write(f'A{pitch_info_start_row + 2}', 'Max Pitch (deg):', info_header_format)
            report_sheet.write(f'B{pitch_info_start_row + 2}', f"{pitch_stats['max']:.3f}")
            report_sheet.write(f'A{pitch_info_start_row + 3}', 'Min Pitch (deg):', info_header_format)
            report_sheet.write(f'B{pitch_info_start_row + 3}', f"{pitch_stats['min']:.3f}")

        # --- Overall Result Summary ---
        summary_start_row = 9
        final_judgment = "OK" if "NG" not in df_report['Overall (OK/NG)'].unique() else "NG"
        report_sheet.write(f'A{summary_start_row}', 'Final Result:', subtitle_format)
        report_sheet.merge_range(f'B{summary_start_row}:C{summary_start_row}', final_judgment, ok_format if final_judgment == "OK" else ng_format)
        
        # --- Detailed Result Table ---
        table_start_row = summary_start_row + 2
        headers = ["Channel", "Overall (OK/NG)", "초과율 (%)", "누적 NG 거리 (m)", "WorstExceed (lx)", "WorstDist (m)", "최대 초과지점 초과율 (%)"]
        for col_num, header in enumerate(headers):
            report_sheet.write(table_start_row, col_num, header, header_format)
        
        if not df_report.empty:
            for row_num, row_data in df_report.iterrows():
                current_row = table_start_row + row_num + 1
                report_sheet.write(current_row, 0, row_data['Channel'], cell_format)
                report_sheet.write(current_row, 1, row_data['Overall (OK/NG)'], ok_format if row_data['Overall (OK/NG)'] == "OK" else ng_format)
                
                exceed_rate = float(row_data['초과율 (%)'])
                ng_dist = float(row_data['누적 NG 거리 (m)'])
                worst_exceed = float(row_data['WorstExceed (lx)'])
                worst_dist = float(row_data['WorstDist (m)'])
                rate_at_worst = float(row_data['최대 초과지점 초과율 (%)'])

                report_sheet.write_number(current_row, 2, exceed_rate, cell_format)
                report_sheet.write_number(current_row, 3, ng_dist, cell_format)

                if np.isnan(worst_exceed) or worst_exceed == 0:
                    report_sheet.write(current_row, 4, "-", cell_format)
                else:
                    report_sheet.write_number(current_row, 4, worst_exceed, cell_format)

                if np.isnan(worst_dist):
                    report_sheet.write(current_row, 5, "-", cell_format)
                else:
                    report_sheet.write_number(current_row, 5, worst_dist, cell_format)

                if np.isnan(rate_at_worst) or rate_at_worst == 0:
                    report_sheet.write(current_row, 6, "-", cell_format)
                else:
                    report_sheet.write_number(current_row, 6, rate_at_worst, cell_format)

        # --- Add Explanations ---
        explanation_start_row = table_start_row + len(df_report) + 2
        report_sheet.merge_range(f'A{explanation_start_row}:G{explanation_start_row}', '용어 설명', subtitle_format)
        
        explanation_format = workbook.add_format({'font_size': 10, 'valign': 'top', 'text_wrap': True, 'border': 1})
        explanation_header_format = workbook.add_format({'bold': True, 'font_size': 10, 'align': 'center', 'valign': 'vcenter', 'bg_color': '#F2F2F2', 'border': 1})

        report_sheet.write(f'A{explanation_start_row + 1}', '초과율 (%)', explanation_header_format)
        report_sheet.merge_range(f'B{explanation_start_row + 1}:G{explanation_start_row + 1}', '전체 평가 구간 거리 중, 법규 조도 기준을 초과한 누적 거리의 비율입니다.', explanation_format)
        
        report_sheet.write(f'A{explanation_start_row + 2}', '최대 초과지점 초과율 (%)', explanation_header_format)
        report_sheet.merge_range(f'B{explanation_start_row + 2}:G{explanation_start_row + 2}', '법규 기준을 가장 크게 위반한 지점에서, 기준값 대비 초과된 조도의 비율입니다. ((측정값 - 기준값) / 기준값 * 100)', explanation_format)

        # --- Place Graphs ---
        graph_start_row = explanation_start_row + 4 # Adjust graph start row
        if fig_path:
            report_sheet.write(f'A{graph_start_row}', 'Graph 1: Driving Path', subtitle_format)
            img_path_data = io.BytesIO()
            fig_path.write_image(img_path_data, format='png', width=720, height=405)
            report_sheet.insert_image(f'B{graph_start_row + 1}', 'path_graph.png', {'image_data': img_path_data})

        if fig_curve:
            curve_graph_start_row = graph_start_row + 23
            report_sheet.write(f'A{curve_graph_start_row}', 'Graph 2: Illuminance vs. Distance', subtitle_format)
            img_curve_data = io.BytesIO()
            fig_curve.write_image(img_curve_data, format='png', width=720, height=405)
            report_sheet.insert_image(f'B{curve_graph_start_row + 1}', 'curve_graph.png', {'image_data': img_curve_data})
        
        if fig_pitch:
            pitch_graph_start_row = graph_start_row + 46
            report_sheet.write(f'A{pitch_graph_start_row}', 'Graph 3: Pitch vs. Distance', subtitle_format)
            img_pitch_data = io.BytesIO()
            fig_pitch.write_image(img_pitch_data, format='png', width=720, height=405)
            report_sheet.insert_image(f'B{pitch_graph_start_row + 1}', 'pitch_graph.png', {'image_data': img_pitch_data})

        # --- Sheet 2: Raw Data with Highlighting ---
        raw_data_sheet = workbook.add_worksheet('LAW DATA')
        raw_data_sheet.set_column('A:Z', 15)

        # Write header
        for col_num, value in enumerate(df_gridded.columns.values):
            raw_data_sheet.write(0, col_num, value, header_format)

        # Write data with conditional formatting
        for row_num, row_data in df_gridded.iterrows():
            row_format = yellow_format if ng_row_mask[row_num] else None
            for col_num, cell_value in enumerate(row_data):
                if pd.isna(cell_value):
                    raw_data_sheet.write_blank(row_num + 1, col_num, None, row_format)
                elif isinstance(cell_value, (int, float)):
                    raw_data_sheet.write_number(row_num + 1, col_num, cell_value, row_format)
                else:
                    raw_data_sheet.write(row_num + 1, col_num, cell_value, row_format)

        # --- Sheet 3: Pitch Exceedance Data ---
        if df_pitch_exceeded is not None and not df_pitch_exceeded.empty:
            pitch_sheet = workbook.add_worksheet('Pitch 초과 지점')
            pitch_sheet.set_column('A:Z', 18)

            pitch_sheet.merge_range('A1:D1', 'Pitch 기준선 (Avg ± 0.3 deg) 초과 지점', title_format)

            if scenario == 'Oncoming':
                relevant_channels = [f'Middle Gain {i}' for i in [4, 5] if f'Middle Gain {i}' in df_pitch_exceeded.columns]
            else: # Preceding
                relevant_channels = [f'Middle Gain {i}' for i in range(6, 11) if f'Middle Gain {i}' in df_pitch_exceeded.columns]
            
            display_cols = ['Distance', 'Pitch'] + relevant_channels
            
            for col_num, value in enumerate(display_cols):
                pitch_sheet.write(2, col_num, value, header_format)

            for row_num, row_data in df_pitch_exceeded[display_cols].reset_index(drop=True).iterrows():
                for col_num, cell_value in enumerate(row_data):
                    if pd.isna(cell_value):
                        pitch_sheet.write_blank(row_num + 3, col_num, None, cell_format)
                    elif isinstance(cell_value, (int, float)):
                        pitch_sheet.write_number(row_num + 3, col_num, cell_value, cell_format)
                    else:
                        pitch_sheet.write(row_num + 3, col_num, cell_value, cell_format)

    return output.getvalue(), df_report

def plot_distance_illuminance_curve(df, course, scenario, file_name, fig=None, xaxis_max=None, color=None):
    """거리-조도 곡선 그래프를 생성하거나 기존 그래프에 데이터를 추가합니다."""
    if fig is None:
        fig = go.Figure()
        # Add layout elements only for the first plot
        fig.update_layout(title=f"거리-조도 곡선 ({scenario} - {course})", xaxis_title="거리 (m)", yaxis_title="조도 (lx)", legend_title="파일 - 채널", font_family="Noto Sans CJK KR")
        fig.update_xaxes(range=[0, xaxis_max])
        fig.update_yaxes(type="linear", autorange=True)

        # Add vrects and Upper Limit only once
        eval_ranges = get_evaluation_ranges(course, scenario)
        vrect_colors = px.colors.qualitative.Pastel
        for i, (start, end) in enumerate(eval_ranges):
            fig.add_vrect(x0=start, x1=end, fillcolor=vrect_colors[i % len(vrect_colors)], opacity=0.15, layer="below", line_width=0, annotation_text=f"평가구간 {i+1}", annotation_position="top left")
        
        temp_df_for_limit = df.copy()
        if 'UpperLimit' not in temp_df_for_limit.columns:
             temp_df_for_limit['UpperLimit'] = get_baseline(course, scenario, temp_df_for_limit['Distance'])
        df_limit = temp_df_for_limit[['Distance', 'UpperLimit']].dropna().sort_values('Distance')
        fig.add_trace(go.Scatter(x=df_limit['Distance'], y=df_limit['UpperLimit'], mode='lines', name='Upper Limit', line=dict(color="black", dash="dot", width=2.5)))

    if 'UpperLimit' not in df.columns:
        df['UpperLimit'] = get_baseline(course, scenario, df['Distance'])

    if scenario == 'Oncoming': channels_to_plot = [4, 5]
    elif scenario == 'Preceding': channels_to_plot = [6, 7, 8, 9, 10]
    else: channels_to_plot = range(1, 11)

    for i in channels_to_plot:
        channel_name = f"Middle Gain {i}"
        if channel_name in df.columns and not df[channel_name].isnull().all():
            trace_name = f"{file_name} - {channel_name}"
            legend_group = f"{file_name}-{i}"
            fig.add_trace(go.Scatter(x=df['Distance'], y=df[channel_name], mode='lines', name=trace_name, legendgroup=legend_group, line=dict(color=color)))
            
            exceeded_line = df[channel_name].copy()
            exceeded_line[(df[channel_name] <= df['UpperLimit']) | (df['UpperLimit'].isnull())] = np.nan
            fig.add_trace(go.Scatter(x=df['Distance'], y=exceeded_line, mode='lines', name=f'{trace_name} 초과', legendgroup=legend_group, showlegend=False, line=dict(color='red', width=2.5)))

    return fig

def plot_pitch_curve(df, file_name, course, scenario, fig=None, xaxis_max=None, color=None, pitch_stats=None):
    """거리-Pitch 곡선 그래프를 생성하거나 기존 그래프에 데이터를 추가하고, 통계 참조선을 그립니다."""
    if 'Pitch' not in df.columns or df['Pitch'].isnull().all():
        st.warning(f"[{file_name}] Pitch 데이터가 없어 그래프를 그릴 수 없습니다.")
        return fig # Return original figure if no data

    if fig is None:
        fig = go.Figure()
        fig.update_layout(
            font_family="Noto Sans CJK KR",
            title="거리-Pitch 변화 (유효구간 통계 기반 참조선)",
            xaxis_title="거리 (m)",
            yaxis_title="Pitch (deg)",
            showlegend=True,
            legend_title="File"
        )
        fig.update_xaxes(range=[0, xaxis_max])

        # Add vrects for evaluation ranges
        eval_ranges = get_evaluation_ranges(course, scenario)
        vrect_colors = px.colors.qualitative.Pastel
        for i, (start, end) in enumerate(eval_ranges):
            fig.add_vrect(x0=start, x1=end, fillcolor=vrect_colors[i % len(vrect_colors)], opacity=0.15, layer="below", line_width=0, annotation_text=f"평가구간 {i+1}", annotation_position="top left")

    # Add main pitch trace
    fig.add_trace(go.Scatter(
        x=df['Distance'],
        y=df['Pitch'],
        mode='lines',
        name=f'{file_name}',
        line=dict(color=color),
        legendgroup=file_name
    ))

    # Add statistics lines if available
    if pitch_stats:
        avg_pitch = pitch_stats['avg']
        
        # Average line
        fig.add_trace(go.Scatter(
            x=df['Distance'], y=[avg_pitch] * len(df),
            mode='lines',
            name=f'Avg Pitch ({avg_pitch:.3f})',
            line=dict(color=color, dash='dot'),
            legendgroup=file_name,
            showlegend=True
        ))
        # +/- 0.3 lines
        fig.add_trace(go.Scatter(
            x=df['Distance'], y=[avg_pitch + 0.3] * len(df),
            mode='lines',
            name='+0.3',
            line=dict(color=color, dash='dash', width=1),
            legendgroup=file_name,
            showlegend=False # Hide from legend to reduce clutter
        ))
        fig.add_trace(go.Scatter(
            x=df['Distance'], y=[avg_pitch - 0.3] * len(df),
            mode='lines',
            name='-0.3',
            line=dict(color=color, dash='dash', width=1),
            legendgroup=file_name,
            showlegend=False # Hide from legend to reduce clutter
        ))

    return fig

# --- 3. Streamlit UI ---

def main():
    st.set_page_config(layout="wide")
    st.title("북미 ADB 실차평가 데이터 분석 도구 (v9 - 자동 시간 동기화)")

    # --- Session State Initialization ---
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.fig_path = None
        st.session_state.fig_curve = None
        st.session_state.fig_pitch = None
        st.session_state.all_summaries = []
        st.session_state.report_downloads = []

    # --- Sidebar Setup ---
    st.sidebar.header("입력 설정")
    vehicle_csv_files = st.sidebar.file_uploader("1. 주행 데이터 업로드 (CSV)", type="csv", accept_multiple_files=True)
    illuminance_tdms_files = st.sidebar.file_uploader("2. 조도 데이터 업로드 (TDMS)", type="tdms", accept_multiple_files=True)

    st.sidebar.header("분석 설정")
    
    scenario = st.sidebar.selectbox(
        "시나리오 선택",
        options=["Oncoming", "Preceding"],
        index=0,
        key='scenario'
    )

    oncoming_courses = ['ST', '100L', '230L', '230R', '370L', '370R']
    preceding_display_map = { 'ST': 'STP', '230L': '230LP' }

    if scenario == 'Oncoming':
        course_options = [c for c in oncoming_courses if c in ZERO_MAP]
        course = st.sidebar.selectbox("코스 선택", options=course_options, index=0, key='course_oncoming')
    else: # Preceding
        course_display_options = list(preceding_display_map.keys())
        course_selected_display = st.sidebar.selectbox("코스 선택", options=course_display_options, index=0, key='course_preceding')
        course = preceding_display_map[course_selected_display]



    manual_offset = st.sidebar.number_input("수동 조도 오프셋 추가 (lx)", value=0.0, step=0.001, format="%.3f", key='manual_illuminance_offset')
    
    st.sidebar.header("출력 설정")
    save_path_input = st.sidebar.text_input("결과 저장 폴더 (선택 사항)", placeholder="예: C:\\Users\\Desktop")

    if st.sidebar.button("분석 시작", type="primary"):
        st.session_state.analysis_done = False # Reset on new analysis
        if not vehicle_csv_files or not illuminance_tdms_files:
            st.sidebar.warning("주행 및 조도 파일을 모두 업로드하세요.")
        elif len(vehicle_csv_files) != len(illuminance_tdms_files):
            st.sidebar.error("주행(CSV)과 조도(TDMS) 파일의 개수가 일치해야 합니다.")
        else:
            with st.spinner("전체 분석 진행 중..."):
                # --- Setup for multiple file processing ---
                fig_path = go.Figure()
                fig_curve = None
                fig_pitch = None
                all_summaries = []
                report_downloads = []
                plot_colors = px.colors.qualitative.Plotly

                # --- Main processing loop ---
                for i, (vehicle_csv_file, illuminance_tdms_file) in enumerate(zip(vehicle_csv_files, illuminance_tdms_files)):
                    csv_filename = vehicle_csv_file.name
                    tdms_filename = illuminance_tdms_file.name
                    file_identifier = csv_filename.replace('.csv', '')
                    color = plot_colors[i % len(plot_colors)]

                    st.markdown(f"--- \n ## 분석 ({i+1}/{len(vehicle_csv_files)}): {file_identifier}")
                    
                    df_vehicle = load_vehicle_csv(vehicle_csv_file)
                    df_illuminance = load_illuminance_tdms(illuminance_tdms_file)

                    if df_vehicle is None or df_illuminance is None:
                        st.error(f"'{file_identifier}' 파일 로드에 실패했습니다. 다음 파일로 넘어갑니다.")
                        continue

                    time_offset = TIME_OFFSET_SECONDS
                    df_synced = synchronize_data(df_vehicle, df_illuminance, time_offset)

                    if df_synced is None:
                        st.error(f"'{file_identifier}' 데이터 동기화에 실패했습니다. 다음 파일로 넘어갑니다.")
                        continue
                    
                    # --- 자동 및 수동 조도 오프셋 계산 및 적용 ---
                    zero_point = ZERO_MAP.get(course)
                    # 1. 오프셋 계산을 위해 임시로 거리 계산
                    temp_df_with_dist = calculate_distance(df_synced.copy(), zero_point)

                    auto_offset = 0.0
                    if temp_df_with_dist is not None:
                        min_dist_idx = temp_df_with_dist['Distance'].idxmin()
                        zero_point_data = temp_df_with_dist.loc[min_dist_idx]
                        
                        if scenario == 'Oncoming':
                            channels_for_ambient = [f'Middle Gain {i}' for i in [4, 5]]
                        else: # Preceding
                            channels_for_ambient = [f'Middle Gain {i}' for i in range(6, 11)]
                        
                        ambient_values = [zero_point_data.get(ch) for ch in channels_for_ambient if pd.notna(zero_point_data.get(ch))]

                        if ambient_values:
                            ambient_baseline = min(ambient_values)
                            auto_offset = -ambient_baseline
                        else:
                            st.warning("0m 지점 조도값을 찾을 수 없어 자동 오프셋을 적용할 수 없습니다.")
                    else:
                        st.error(f"'{file_identifier}' 거리 계산 실패로 오프셋을 적용할 수 없습니다.")

                    # 2. 최종 오프셋 계산 및 정보 표시
                    total_offset = auto_offset + manual_offset
                    
                    st.write("##### 조도 오프셋 정보")
                    c1, c2, c3 = st.columns(3)
                    c1.metric(label="계산된 자동 오프셋", value=f"{auto_offset:.3f} lx")
                    c2.metric(label="수동 추가 오프셋", value=f"{manual_offset:.3f} lx")
                    c3.metric(label="최종 적용 오프셋", value=f"{total_offset:.3f} lx", delta=f"{manual_offset:.3f} lx")


                    # 3. 원본 데이터에 최종 오프셋 적용
                    if total_offset != 0.0:
                        mid_gain_cols = [f'Middle Gain {i}' for i in range(1, 11)]
                        for col in mid_gain_cols:
                            if col in df_synced.columns:
                                df_synced[col] = pd.to_numeric(df_synced[col], errors='coerce') + total_offset
                        st.info(f"조도 값에 최종 오프셋 {total_offset:.3f} lx를 적용했습니다.")
                    
                    # 4. 오프셋이 적용된 데이터로 최종 거리 계산
                    df_with_dist = calculate_distance(df_synced, zero_point)
                    
                    if df_with_dist is None or df_with_dist.empty:
                        st.warning(f"'{file_identifier}' 거리 계산에 실패했습니다.")
                        continue

                    min_dist_idx = df_with_dist['Distance'].idxmin()
                    if isinstance(min_dist_idx, pd.Series): min_dist_idx = min_dist_idx.iloc[0]
                    df_approaching = df_with_dist.loc[:min_dist_idx].copy()
                    st.info(f"차량 접근 구간 데이터 필터링 완료. (총 {len(df_approaching)}개 샘플)")

                    df_final_for_report = filter_by_evaluation_range(df_approaching, course, scenario)

                    # --- Pitch 통계 계산 (유효 구간 기준) ---
                    pitch_stats = None
                    if df_final_for_report is not None and not df_final_for_report.empty and 'Pitch' in df_final_for_report.columns and not df_final_for_report['Pitch'].isnull().all():
                        pitch_series = df_final_for_report['Pitch']
                        pitch_stats = {
                            'avg': pitch_series.mean(),
                            'max': pitch_series.max(),
                            'min': pitch_series.min()
                        }
                        st.markdown(f"##### [{file_identifier}] 유효구간 Pitch 통계")
                        col1, col2, col3 = st.columns(3)
                        col1.metric(label="평균 Pitch (deg)", value=f"{pitch_stats['avg']:.3f}")
                        col2.metric(label="최대 Pitch (deg)", value=f"{pitch_stats['max']:.3f}")
                        col3.metric(label="최소 Pitch (deg)", value=f"{pitch_stats['min']:.3f}")
                    else:
                        st.info(f"[{file_identifier}] 유효구간에 Pitch 데이터가 없어 통계를 계산할 수 없습니다.")

                    # --- Pitch 기준선 초과 지점 분석 ---
                    df_pitch_exceeded = None # Initialize
                    if pitch_stats and df_final_for_report is not None:
                        avg_pitch = pitch_stats['avg']
                        upper_bound = avg_pitch + 0.3
                        lower_bound = avg_pitch - 0.3

                        exceed_mask = (df_final_for_report['Pitch'] > upper_bound) | (df_final_for_report['Pitch'] < lower_bound)
                        df_pitch_exceeded = df_final_for_report[exceed_mask]

                        st.markdown(f"##### [{file_identifier}] Pitch 기준선 (Avg ± 0.3 deg) 초과 지점")
                        if not df_pitch_exceeded.empty:
                            if scenario == 'Oncoming':
                                relevant_channels = [f'Middle Gain {i}' for i in [4, 5] if f'Middle Gain {i}' in df_pitch_exceeded.columns]
                            else: # Preceding
                                relevant_channels = [f'Middle Gain {i}' for i in range(6, 11) if f'Middle Gain {i}' in df_pitch_exceeded.columns]
                            
                            display_cols = ['Distance', 'Pitch'] + relevant_channels
                            
                            st.warning(f"{len(df_pitch_exceeded)}개의 지점에서 Pitch 기준선을 초과했습니다.")
                            st.dataframe(df_pitch_exceeded[display_cols].style.format({
                                'Distance': '{:.2f}m',
                                'Pitch': '{:.3f}°',
                                **{ch: '{:.3f} lx' for ch in relevant_channels}
                            }))
                        else:
                            st.success("모든 유효구간에서 Pitch 기준선을 만족합니다.")


                    fig_path_single = px.line(df_vehicle, x='PosLocalX', y='PosLocalY', title=f'주행 경로 ({file_identifier})')
                    fig_path_single.update_traces(line=dict(color='lightgrey'), name='전체 경로', showlegend=True)
                    if df_final_for_report is not None and not df_final_for_report.empty:
                        fig_path_single.add_trace(go.Scatter(x=df_final_for_report['PosLocalX'], y=df_final_for_report['PosLocalY'], mode='markers', name='분석 구간', marker=dict(color='red', size=3)))
                    fig_path_single.update_layout(xaxis_title="PosLocalX (m)", yaxis_title="PosLocalY (m)", font_family="Noto Sans CJK KR")
                    fig_path_single.update_yaxes(scaleanchor="x", scaleratio=1)

                    eval_ranges = get_evaluation_ranges(course, scenario)
                    xaxis_max = max(end for start, end in eval_ranges) * 1.1 if eval_ranges else None
                    
                    fig_curve_single = None
                    fig_pitch_single = None
                    if not df_approaching.empty:
                        fig_curve_single = plot_distance_illuminance_curve(df_approaching, course, scenario, file_identifier, fig=None, xaxis_max=xaxis_max, color=color)
                        fig_pitch_single = plot_pitch_curve(df_approaching.copy(), file_identifier, course, scenario, fig=None, xaxis_max=xaxis_max, color=color, pitch_stats=pitch_stats)

                    fig_path.add_trace(go.Scatter(x=df_vehicle['PosLocalX'], y=df_vehicle['PosLocalY'], mode='lines', name=f'{file_identifier} (전체)', legendgroup=f'group{i}', line=dict(color=color, dash='dot')))
                    if df_final_for_report is not None and not df_final_for_report.empty:
                        fig_path.add_trace(go.Scatter(x=df_final_for_report['PosLocalX'], y=df_final_for_report['PosLocalY'], mode='lines', name=f'{file_identifier} (분석)', legendgroup=f'group{i}', line=dict(color=color, width=3)))

                    if not df_approaching.empty:
                        fig_curve = plot_distance_illuminance_curve(df_approaching, course, scenario, file_identifier, fig=fig_curve, xaxis_max=xaxis_max, color=color)
                        fig_pitch = plot_pitch_curve(df_approaching, file_identifier, course, scenario, fig=fig_pitch, xaxis_max=xaxis_max, color=color, pitch_stats=pitch_stats)

                    if df_final_for_report is not None and not df_final_for_report.empty:
                        report_bytes, df_summary = generate_report(df_final_for_report, course, scenario, fig_path=fig_path_single, fig_curve=fig_curve_single, fig_pitch=fig_pitch_single, csv_filename=csv_filename, tdms_filename=tdms_filename, df_pitch_exceeded=df_pitch_exceeded, pitch_stats=pitch_stats)
                        if df_summary is not None and not df_summary.empty: all_summaries.append((file_identifier, df_summary))
                        if report_bytes: report_downloads.append((file_identifier, course, scenario, report_bytes))
                    else:
                        st.warning(f"[{file_identifier}] 유효 평가 구간 내에 분석할 데이터가 없습니다.")

                # --- Store results in session state ---
                st.session_state.fig_path = fig_path
                st.session_state.fig_curve = fig_curve
                st.session_state.fig_pitch = fig_pitch
                st.session_state.all_summaries = all_summaries
                st.session_state.report_downloads = report_downloads
                st.session_state.analysis_done = True
                st.session_state.save_path = save_path_input # Store save path

    # --- Display results from session state ---
    if st.session_state.analysis_done:
        st.markdown("--- \n # 종합 결과")

        st.subheader("주행 경로")
        st.plotly_chart(st.session_state.fig_path, use_container_width=True)

        st.subheader("거리-조도 곡선")
        if st.session_state.fig_curve:
            st.plotly_chart(st.session_state.fig_curve, use_container_width=True)
        
        st.subheader("거리-Pitch 변화")
        if st.session_state.fig_pitch:
            st.plotly_chart(st.session_state.fig_pitch, use_container_width=True)

        st.subheader("최종 판정 요약")
        with st.expander("용어 설명 보기"):
            st.markdown("""
            - **초과율 (%)**: 전체 평가 구간 거리 중, 법규 조도 기준을 초과한 누적 거리의 비율입니다.
            - **최대지점 초과율 (%)**: 법규 기준을 가장 크게 위반한 지점에서, 기준값 대비 초과된 조도의 비율입니다. (`(측정값 - 기준값) / 기준값 * 100`)
            """
            )

        for file_identifier, df_summary in st.session_state.all_summaries:
            st.markdown(f"#### {file_identifier}")
            df_summary.rename(columns={'Overall (OK/NG)': '판정', 'WorstExceed (lx)': '최대 초과량 (lx)', 'WorstDist (m)': '최대 초과 지점 (m)', '최대 초과지점 초과율 (%)': '최대지점 초과율 (%)'}, inplace=True)
            if '초과율 (%)' not in df_summary.columns: df_summary['초과율 (%)'] = '0.0'
            if '누적 NG 거리 (m)' not in df_summary.columns: df_summary['누적 NG 거리 (m)'] = '0.0'
            for col in ['초과율 (%)', '누적 NG 거리 (m)', '최대 초과량 (lx)', '최대 초과 지점 (m)', '최대지점 초과율 (%)']:
                df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce')
            final_judgment = "OK" if "NG" not in df_summary['판정'].unique() else "NG"
            if final_judgment == "OK": st.success(f"**{file_identifier} 최종 판정: OK**")
            else: st.error(f"**{file_identifier} 최종 판정: NG**")
            display_cols = ['Channel', '판정', '초과율 (%)', '누적 NG 거리 (m)', '최대 초과량 (lx)', '최대 초과 지점 (m)', '최대지점 초과율 (%)']
            st.dataframe(df_summary[display_cols].style.format({'초과율 (%)': '{:.2f}%', '누적 NG 거리 (m)': '{:.2f}', '최대 초과량 (lx)': '{:.2f}', '최대 초과 지점 (m)': '{:.2f}', '최대지점 초과율 (%)': '{:.2f}%'}))

        if st.session_state.report_downloads:
            st.subheader("보고서 저장 및 다운로드")
            save_path = st.session_state.get('save_path', '')

            if st.button("선택한 폴더에 모든 리포트 저장"):
                if save_path and os.path.isdir(save_path):
                    num_saved = 0
                    for file_identifier, course, scenario, report_bytes in st.session_state.report_downloads:
                        file_name = f"ADB_Report_{file_identifier}_{course}_{scenario}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        try:
                            full_path = os.path.join(save_path, file_name)
                            with open(full_path, "wb") as f:
                                f.write(report_bytes)
                            st.success(f"보고서 저장 완료: `{full_path}`")
                            num_saved += 1
                        except Exception as e:
                            st.error(f"파일 저장 오류: {e}")
                    if num_saved > 0:
                        st.balloons()
                elif not save_path:
                    st.warning("저장할 폴더 경로를 먼저 입력해주세요.")
                else:
                    st.error(f"입력한 경로 '{save_path}'를 찾을 수 없거나 폴더가 아닙니다.")

            st.markdown("--- ")
            st.write("개별 파일 다운로드:")
            for file_identifier, course, scenario, report_bytes in st.session_state.report_downloads:
                file_name = f"ADB_Report_{file_identifier}_{course}_{scenario}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.download_button(
                    label=f"📥 {file_identifier} 보고서 다운로드",
                    data=report_bytes,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{file_identifier}"
                )
    else:
        st.info("파일을 업로드하고 분석 설정을 마친 후 '분석 시작'을 클릭하세요.")

if __name__ == "__main__":
    main()