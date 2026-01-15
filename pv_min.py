import os
import pandas as pd
from datetime import datetime, time
import re

def sanitize_filename(filename):
    """
    Remove or replace invalid characters in filename
    """
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    return filename

def get_stock_name(stock_code, date, pv_list_path):
    """
    從對應日期的 CSV 檔案中獲取股票名稱
    """
    file_name = f"stocklist_{date}_3days.csv"
    file_path = os.path.join(pv_list_path, file_name)
    
    try:
        df = pd.read_csv(file_path)
        stock_info = df[df['Stock Code'].astype(str) == str(stock_code)]
        if not stock_info.empty:
            return stock_info.iloc[0]['Stock Name']
    except Exception as e:
        print(f"Warning: Could not read stock name from {file_path}: {e}")
    
    return None

def get_date_stock_pairs(tick_path):
    """
    直接從tick資料夾獲取所有日期和股票代碼
    返回格式: [(date, stock_code), ...]
    """
    date_stock_pairs = []
    
    for date_folder in os.listdir(tick_path):
        date_folder_path = os.path.join(tick_path, date_folder)
        if os.path.isdir(date_folder_path):
            for file in os.listdir(date_folder_path):
                if file.endswith('_ticks.csv'):
                    stock_code = file.split('_')[0]
                    formatted_date = date_folder.replace('-', '_')
                    date_stock_pairs.append((date_folder, formatted_date, stock_code))
    
    return date_stock_pairs

def process_tick_data(tick_path, date, stock_code):
    file_name = f"{stock_code}_{date}_ticks.csv"
    file_path = os.path.join(tick_path, date, file_name)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    print(f"Processing {file_name}")
    
    if 'ts' not in df.columns:
        print(f"Warning: 'ts' column not found in {file_name}")
        return None
    
    df['datetime'] = pd.to_datetime(df['ts'])
    df = df[(df['datetime'].dt.time >= time(9, 0)) & (df['datetime'].dt.time <= time(13, 30))]
    return df

def generate_price_volume_details(df):
    if df is None or df.empty:
        return None
    price_volume = df.groupby('close')['volume'].sum().reset_index()
    price_volume = price_volume.sort_values('close', ascending=False)
    return price_volume

def find_extreme_price_times(df, volume_threshold, max_volume):
    if df is None or df.empty:
        return pd.DataFrame({'price': ['N/A'], 'start_time': ['N/A'], 'end_time': ['N/A'], 'duration': ['N/A']}), pd.DataFrame({'price': ['N/A'], 'start_time': ['N/A'], 'end_time': ['N/A'], 'duration': ['N/A']})
    
    def track_price(price_func, comp_func):
        results = []
        current_volumes = {}
        current_tracking_price = None
        tracking_start_time = None
        
        for _, row in df.iterrows():
            price = row['close']
            volume = row['volume']
            current_time = row['datetime']
            
            if price not in current_volumes:
                current_volumes[price] = 0
            current_volumes[price] += volume
            
            min_price = min(current_volumes.keys())  # 改為找最低價
            
            if current_tracking_price is None:
                if volume_threshold <= current_volumes[min_price] <= max_volume:
                    current_tracking_price = min_price
                    tracking_start_time = current_time
            else:
                if min_price < current_tracking_price:  # 改為比較最低價
                    if tracking_start_time is not None:
                        results.append({
                            'price': current_tracking_price,
                            'start_time': tracking_start_time,
                            'end_time': current_time,
                            'duration': (current_time - tracking_start_time).total_seconds()
                        })
                    current_tracking_price = None
                    tracking_start_time = None
                    if volume_threshold <= current_volumes[min_price] <= max_volume:
                        current_tracking_price = min_price
                        tracking_start_time = current_time
                elif current_volumes[current_tracking_price] > max_volume:
                    results.append({
                        'price': current_tracking_price,
                        'start_time': tracking_start_time,
                        'end_time': current_time,
                        'duration': (current_time - tracking_start_time).total_seconds()
                    })
                    current_tracking_price = None
                    tracking_start_time = None
        
        if current_tracking_price is not None and volume_threshold <= current_volumes[current_tracking_price] <= max_volume:
            results.append({
                'price': current_tracking_price,
                'start_time': tracking_start_time,
                'end_time': df['datetime'].iloc[-1],
                'duration': (df['datetime'].iloc[-1] - tracking_start_time).total_seconds()
            })
        
        results.sort(key=lambda x: x['start_time'])
        
        return pd.DataFrame(results) if results else pd.DataFrame({'price': ['N/A'], 'start_time': ['N/A'], 'end_time': ['N/A'], 'duration': ['N/A']})
    
    min_summary = track_price(min, lambda x, y: x < y)  # 改為先返回最低價結果
    max_summary = track_price(max, lambda x, y: x > y)
    
    return min_summary, max_summary  # 交換返回順序

def format_duration(seconds):
    """
    將秒數轉換為 HH:MM:SS 格式
    """
    try:
        hours, remainder = divmod(int(float(seconds)), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return 'N/A'

def write_to_csv(date, stock_code, stock_name, long_duration_records, output_csv_path):
    """
    將長時間持續的記錄寫入CSV檔案
    """
    import csv
    import os
    from datetime import datetime, time
    
    if not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)
        
    csv_file = os.path.join(output_csv_path, 'long_duration_signals.csv')
    file_exists = os.path.exists(csv_file)
    
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    formatted_date = f"{date_obj.year - 1911}/{date_obj.month:02d}/{date_obj.day:02d}"
    
    one_minute_time = time(9, 1, 0)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            headers = [
                '日期', 'Stock Code', 'Stock Name', '盤中訊號出現時間', '訊號持續時間',
                '最高價格', '連外次', '連外量', '連外次結束時間', '連內次', '連內量',
                '連內次結束時間', '成交價', '下單後可獲利', '1分內', '連內成交額',
                '連外成交額', '備註'
            ]
            writer.writerow(headers)
        
        for price, start_time, end_time, duration, _ in long_duration_records:
            if isinstance(start_time, pd.Timestamp):
                signal_time = start_time.strftime('%H:%M:%S')
                is_within_one_minute = start_time.time() <= one_minute_time
            else:
                signal_time = start_time.split(' ')[1]
                time_obj = datetime.strptime(signal_time, '%H:%M:%S').time()
                is_within_one_minute = time_obj <= one_minute_time
            
            row = [
                formatted_date,
                stock_code,
                stock_name,
                signal_time,
                duration,
                f"{float(price):.2f}",
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                'Y' if is_within_one_minute else '',
                '',
                '',
                ''
            ]
            writer.writerow(row)

def write_to_file(file_path, stock_code, stock_name, date, price_volume_details, max_times, min_times, duration_threshold):
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    safe_filename = sanitize_filename(filename)
    safe_file_path = os.path.join(directory, safe_filename)
    
    long_duration_records = []
    
    with open(safe_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Stock: {stock_code} {stock_name if stock_name else ''}\n")
        f.write(f"Date: {date}\n\n")
        
        f.write("Price-Volume Details (Sorted by price, high to low):\n")
        if not price_volume_details.empty:
            f.write(" close  volume\n")
            # Convert 'close' column to float and handle any conversion errors
            try:
                price_volume_details['close'] = pd.to_numeric(price_volume_details['close'], errors='coerce')
                price_volume_details['close'] = price_volume_details['close'].apply(lambda x: f" {x:>6.2f}" if pd.notnull(x) else "   N/A")
                price_volume_details['volume'] = price_volume_details['volume'].apply(lambda x: f"{x:>8}")
                f.write(price_volume_details.to_string(index=False))
            except Exception as e:
                print(f"Warning: Error formatting price-volume details for {stock_code}: {e}")
                f.write("Error formatting price-volume details\n")
        f.write("\n\n")
        
        normal_duration_records = []
        
        for _, row in max_times.iterrows():
            if row['price'] == 'N/A':
                continue
            
            try:
                # Convert price to float if it's a string
                price = float(row['price']) if isinstance(row['price'], str) else row['price']
                duration_seconds = float(row['duration'])
                if duration_seconds == 0:
                    continue
                
                formatted_duration = format_duration(duration_seconds)
            except (ValueError, TypeError) as e:
                print(f"Warning: Error processing record for {stock_code}: {e}")
                continue
            
            record = (price, row['start_time'], row['end_time'], formatted_duration, duration_seconds)
            
            if duration_seconds > duration_threshold:
                long_duration_records.append(record)
            else:
                normal_duration_records.append(record)
        
        f.write("Times of min price with volume 1-10:\n")
        for price, start_time, end_time, duration, _ in normal_duration_records:
            try:
                price_str = f"{float(price):>6.2f}"
            except (ValueError, TypeError):
                price_str = "   N/A"
            start_time_str = start_time if pd.notnull(start_time) else 'N/A'
            end_time_str = end_time if pd.notnull(end_time) else 'N/A'
            f.write(f"{price_str} {start_time_str} {end_time_str} {duration}\n")
        
        if long_duration_records:
            f.write(f"\nTimes of max price with duration longer than {duration_threshold} seconds:\n")
            for price, start_time, end_time, duration, _ in long_duration_records:
                try:
                    price_str = f"{float(price):>6.2f}"
                except (ValueError, TypeError):
                    price_str = "   N/A"
                start_time_str = start_time if pd.notnull(start_time) else 'N/A'
                end_time_str = end_time if pd.notnull(end_time) else 'N/A'
                f.write(f"{price_str} {start_time_str} {end_time_str} {duration}\n")
    
    return long_duration_records

def main():
    # 路徑設定
    tick_data_path = r"D:\FTP_Jim\代碼備份\backtest\test\pv_tick"
    output_path = r"D:\FTP_Jim\代碼備份\backtest\test\output"
    pv_list_path = r"D:\FTP_Jim\代碼備份\backtest\test\pv_list"
    output_csv_path = r"D:\FTP_Jim\代碼備份\backtest\test\output\CSV"
    
    # 參數設定
    volume_threshold = 1    # 成交量下限
    max_volume = 10   # 成交量上限 (原本是10，現在改為5)
    duration_threshold = 30 # 持續時間閾值（秒）

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    date_stock_pairs = get_date_stock_pairs(tick_data_path)
    
    for date_folder, formatted_date, stock_code in date_stock_pairs:
        print(f"Processing date: {date_folder}, stock: {stock_code}")
        
        stock_name = get_stock_name(stock_code, formatted_date, pv_list_path)
        
        tick_data = process_tick_data(tick_data_path, date_folder, stock_code)
        if tick_data is None:
            print(f"No tick data found for {stock_code} on {date_folder}")
            continue

        price_volume_details = generate_price_volume_details(tick_data)
        max_volume_times, min_volume_times = find_extreme_price_times(tick_data, volume_threshold, max_volume)

        if price_volume_details is not None:
            file_name = f"{date_folder}_{stock_code}.txt"
            file_path = os.path.join(output_path, file_name)
            
            long_duration_records = write_to_file(file_path, stock_code, stock_name, date_folder, 
                                                price_volume_details, max_volume_times, min_volume_times, 
                                                duration_threshold)
            
            if long_duration_records:
                write_to_csv(date_folder, stock_code, stock_name, long_duration_records, output_csv_path)
            
            print(f"Data written to {file_path}")
        else:
            print(f"No valid data to write for {stock_code} on {date_folder}")

        print("\n")

if __name__ == "__main__":
    main()