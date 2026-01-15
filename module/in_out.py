import pandas as pd
from datetime import timedelta
from pathlib import Path
import os

class StockDataAnalyzer:
    def __init__(self, data_dir, output_dir):
        """初始化分析器"""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.consecutive_outside_count = 0
        self.consecutive_inside_count = 0
        self.cumulative_outside_volume = 0
        self.cumulative_inside_volume = 0
        self.cumulative_total_volume = 0  
        
    def read_stock_data(self, file_path, remove_first_last=False):
        """
        讀取股票資料檔案，可選擇是否移除第一筆和最後一筆資料
        """
        try:
            df = pd.read_csv(file_path)
            df['ts'] = pd.to_datetime(df['ts'])
            
            # 排序資料
            df = df.sort_values('ts')
            
            # 根據參數決定是否移除首尾資料
            if remove_first_last and len(df) > 2:
                df = df.iloc[1:-1]
                
            return df
        except Exception as e:
            print(f"讀取檔案 {file_path.name} 時發生錯誤: {str(e)}")
            return None
    
    def create_time_intervals(self, start_time, end_time, interval_seconds=5):
        """
        創建固定的時間區間
        """
        intervals = []
        current_time = start_time.floor('5s')
        end_time = end_time.ceil('5s')
        
        while current_time < end_time:
            interval_end = current_time + timedelta(seconds=interval_seconds)
            intervals.append((current_time, interval_end))
            current_time = interval_end
            
        return intervals
    
    def calculate_total_volume(self, df, start_time, end_time, is_first_interval):
        """
        計算總成交量（包含首尾資料）
        """
        if is_first_interval:
            mask = (df['ts'] >= start_time) & (df['ts'] <= end_time)
        else:
            mask = (df['ts'] > start_time) & (df['ts'] <= end_time)
        
        interval_data = df[mask]
        return float(interval_data['volume'].sum() if not interval_data.empty else 0)
            
    def group_by_time_interval(self, full_data, filtered_data, interval_seconds=5):
        """
        將資料按固定的時間間隔分組並計算統計資訊
        """
        records = []
        self.consecutive_outside_count = 0
        self.consecutive_inside_count = 0
        self.cumulative_outside_volume = 0
        self.cumulative_inside_volume = 0
        self.cumulative_total_volume = 0
        last_trade_time = None
        
        if filtered_data.empty:
            return records
            
        # 確保數據類型
        filtered_data['tick_type'] = pd.to_numeric(filtered_data['tick_type'])
        filtered_data['volume'] = pd.to_numeric(filtered_data['volume'])
        filtered_data['close'] = pd.to_numeric(filtered_data['close'])
        full_data['volume'] = pd.to_numeric(full_data['volume'])
        
        # 創建時間區間
        intervals = self.create_time_intervals(
            filtered_data['ts'].min(),
            filtered_data['ts'].max(),
            interval_seconds
        )
        
        # 對每個時間區間進行統計
        for i, (start_time, end_time) in enumerate(intervals):
            if i == 0:
                filtered_mask = (filtered_data['ts'] >= start_time) & (filtered_data['ts'] <= end_time)
            else:
                filtered_mask = (filtered_data['ts'] > start_time) & (filtered_data['ts'] <= end_time)
                
            interval_data = filtered_data[filtered_mask]
            
            if not interval_data.empty:
                interval_volume = interval_data['volume'].sum()
                outside_volume = interval_data[interval_data['tick_type'] == 1]['volume'].sum()
                inside_volume = interval_data[interval_data['tick_type'] == 2]['volume'].sum()
                last_price = interval_data.iloc[-1]['close']
                
                # 檢查是否有較長的無交易間隔
                current_time = start_time
                if last_trade_time is not None and interval_volume > 0:
                    time_gap = (current_time - last_trade_time).total_seconds()
                    if time_gap > interval_seconds:
                        self.consecutive_outside_count = 0
                        self.consecutive_inside_count = 0
                        self.cumulative_outside_volume = 0
                        self.cumulative_inside_volume = 0
                
                if interval_volume > 0:
                    last_trade_time = current_time
                
                # 計算連外次和連內次
                if outside_volume == 0 and inside_volume == 0:
                    consecutive_outside_number = None
                    consecutive_outside_volume = 0
                    consecutive_inside_number = None
                    consecutive_inside_volume = 0
                else:
                    # 處理外盤優勢
                    if outside_volume > inside_volume:
                        self.consecutive_outside_count += 1
                        self.cumulative_outside_volume += outside_volume
                        self.consecutive_inside_count = 0
                        self.cumulative_inside_volume = 0
                    # 處理內盤優勢
                    elif inside_volume > outside_volume:
                        self.consecutive_inside_count += 1
                        self.cumulative_inside_volume += inside_volume
                        self.consecutive_outside_count = 0
                        self.cumulative_outside_volume = 0
                    # 內外盤相等
                    else:
                        if interval_volume > 0:
                            self.consecutive_outside_count += 1
                            self.consecutive_inside_count += 1
                            self.cumulative_outside_volume += outside_volume
                            self.cumulative_inside_volume += inside_volume
                    
                    consecutive_outside_number = self.consecutive_outside_count if self.consecutive_outside_count > 0 else None
                    consecutive_outside_volume = self.cumulative_outside_volume
                    consecutive_inside_number = self.consecutive_inside_count if self.consecutive_inside_count > 0 else None
                    consecutive_inside_volume = self.cumulative_inside_volume
            else:
                interval_volume = 0
                outside_volume = 0
                inside_volume = 0
                last_price = None  # 改為 None，表示此區間無交易價格
                consecutive_outside_number = None
                consecutive_outside_volume = 0
                consecutive_inside_number = None
                consecutive_inside_volume = 0
            
            # 計算總成交量
            total_volume = self.calculate_total_volume(full_data, start_time, end_time, i == 0)
            self.cumulative_total_volume += total_volume
            
            # 檢查是否需要標記大額交易（外盤）
            needs_mark_outside = False
            if consecutive_outside_volume > 0 and last_price is not None:
                transaction_value = consecutive_outside_volume * last_price * 1000
                needs_mark_outside = transaction_value > 10000000
                
            # 檢查是否需要標記大額交易（內盤）
            needs_mark_inside = False
            if consecutive_inside_volume > 0 and last_price is not None:
                transaction_value = consecutive_inside_volume * last_price * 1000
                needs_mark_inside = transaction_value > 10000000
            
            record = {
                '開始時間': start_time.strftime('%H:%M:%S'),
                '結束時間': end_time.strftime('%H:%M:%S'),
                '成交量': interval_volume,
                '外盤': outside_volume,
                '內盤': inside_volume,
                '連外次': consecutive_outside_number,
                '連外量': consecutive_outside_volume,
                '成交價': last_price,
                '連內次': consecutive_inside_number,
                '連內量': consecutive_inside_volume,
                '總成交量': self.cumulative_total_volume,
                '需要標記外盤': needs_mark_outside,
                '需要標記內盤': needs_mark_inside
            }
            records.append(record)
                
        return records
        
    def analyze_and_export(self, file_name, date, interval_seconds=5):
        """分析資料並輸出到格式化的CSV檔案"""
        print(f"讀取檔案: {file_name}, {date}")
        
        full_df = self.read_stock_data(file_name, remove_first_last=False)
        filtered_df = self.read_stock_data(file_name, remove_first_last=True)
        
        if full_df is not None and not full_df.empty and filtered_df is not None and not filtered_df.empty:
            full_df = full_df.sort_values('ts')
            filtered_df = filtered_df.sort_values('ts')
            
            all_records = self.group_by_time_interval(full_df, filtered_df, interval_seconds)
            result_df = pd.DataFrame(all_records)
            
            def format_volume(value):
                """格式化一般數量"""
                return f"{float(value):>8.1f}"
            
            def format_price(value):
                """格式化價格"""
                if pd.isna(value):
                    return "        "  # 8個空格
                return f"{float(value):>8.2f}"
            
            def format_consecutive_count(value):
                """格式化連續次數，超過10加上星號"""
                if pd.isna(value):
                    return "    "
                if float(value) >= 10:
                    return f"*{value:>3.1f}"
                return f"{value:>4.1f}"
            
            def format_volume_with_mark(row, volume_col, mark_col):
                """格式化成交量，根據條件添加星號"""
                volume = row[volume_col]
                if pd.isna(volume) or volume == 0:
                    return f"{0:>8.1f}"
                
                if row[mark_col]:
                    return f"*{float(volume):>7.1f}"
                return f"{float(volume):>8.1f}"
            
            # 應用格式化
            result_df['成交量'] = result_df['成交量'].apply(format_volume)
            result_df['外盤'] = result_df['外盤'].apply(format_volume)
            result_df['內盤'] = result_df['內盤'].apply(format_volume)
            result_df['連外次'] = result_df['連外次'].apply(format_consecutive_count)
            result_df['連內次'] = result_df['連內次'].apply(format_consecutive_count)
            result_df['連外量'] = result_df.apply(lambda row: format_volume_with_mark(row, '連外量', '需要標記外盤'), axis=1)
            result_df['連內量'] = result_df.apply(lambda row: format_volume_with_mark(row, '連內量', '需要標記內盤'), axis=1)
            result_df['總成交量'] = result_df['總成交量'].apply(format_volume)
            result_df['成交價'] = result_df['成交價'].apply(format_price)
            
            # 從原始檔名取得股票代碼和日期
            stock_code = os.path.basename(file_name).split('_')[0]

            # 構建輸出檔案名稱
            output_file = self.output_dir / date / f'{stock_code}_{date}_5sec.csv'
            
            # 確保完整的目錄路徑存在
            os.makedirs(output_file.parent, exist_ok=True)  # 確保 `self.output_dir / date` 目錄存在

            # 寫入CSV
            with open(output_file, 'w', encoding='utf-8') as f:
                headers = ['開始時間', '結束時間', '成交量', '外盤', '內盤', '連外次', '連外量', 
                          '成交價', '連內次', '連內量', '總成交量']
                f.write(','.join(headers) + '\n')
                
                for _, row in result_df.iterrows():
                    formatted_row = [
                        row['開始時間'],
                        row['結束時間'],
                        row['成交量'],
                        row['外盤'],
                        row['內盤'],
                        row['連外次'],
                        row['連外量'],
                        row['成交價'],
                        row['連內次'],
                        row['連內量'],
                        row['總成交量']
                    ]
                    f.write(','.join(formatted_row) + '\n')
            
            print(f"分析結果已保存到: {output_file}")
            return result_df
        else:
            print(f"檔案 {file_name} 無法處理或沒有資料")
            return None
        