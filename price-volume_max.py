import os
import pandas as pd
from datetime import datetime, time
import re
import multiprocessing as mp
from functools import partial
import psutil
import logging
import gc
import time as time_module
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_processing.log"),
        logging.StreamHandler()
    ]
)

def get_memory_usage():
    """返回目前記憶體使用量（MB）"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024

def sanitize_filename(filename):
    """
    Remove or replace invalid characters in filename
    """
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    return filename

def get_stock_name(stock_code, date, pv_list_path, stock_names_cache={}):
    """
    從對應日期的 CSV 檔案中獲取股票名稱，使用快取加速
    """
    cache_key = f"{stock_code}_{date}"
    if cache_key in stock_names_cache:
        return stock_names_cache[cache_key]
    
    # 嘗試多種可能的檔案名格式
    possible_file_names = [
        f"stocklist_{date}_3days.csv",  # 原始格式
        f"stocklist_{date}.csv",        # 簡化格式
        f"stocklist_{date.replace('_', '-')}_3days.csv",  # 使用連字符
        f"stocklist_{date.replace('_', '-')}.csv"         # 簡化格式使用連字符
    ]
    
    for file_name in possible_file_names:
        file_path = os.path.join(pv_list_path, file_name)
        if os.path.exists(file_path):
            try:
                # 嘗試不同的列名
                try:
                    df = pd.read_csv(file_path)
                    
                    # 判斷檔案中實際存在的欄位名稱
                    col_names = df.columns.tolist()
                    stock_code_col = None
                    stock_name_col = None
                    
                    # 尋找可能的股票代碼欄位名稱
                    for col in ['Stock Code', 'StockCode', 'Code', '股票代碼', 'code']:
                        if col in col_names:
                            stock_code_col = col
                            break
                    
                    # 尋找可能的股票名稱欄位名稱
                    for col in ['Stock Name', 'StockName', 'Name', '股票名稱', 'name']:
                        if col in col_names:
                            stock_name_col = col
                            break
                    
                    if stock_code_col and stock_name_col:
                        # 找到適合的欄位名稱，進行過濾
                        stock_info = df[df[stock_code_col].astype(str) == str(stock_code)]
                        if not stock_info.empty:
                            name = stock_info.iloc[0][stock_name_col]
                            stock_names_cache[cache_key] = name
                            logging.info(f"Found stock name '{name}' for code {stock_code} in {file_name}")
                            return name
                
                except Exception as e:
                    logging.warning(f"Error reading stock name from {file_path} with auto column detection: {e}")
            
            except Exception as e:
                logging.warning(f"Could not read stock name from {file_path}: {e}")
    
    # 如果所有嘗試都失敗，使用硬編碼的映射
    # 這裡可以添加常見的股票代碼和名稱對應
    stock_code_map = {
        "1503": "士電",
        "6163": "華電網",
        # 在這裡添加更多你知道的股票代碼和名稱對應
    }
    
    if stock_code in stock_code_map:
        name = stock_code_map[stock_code]
        stock_names_cache[cache_key] = name
        logging.info(f"Using hardcoded stock name '{name}' for code {stock_code}")
        return name
    
    logging.warning(f"Could not find stock name for code {stock_code}, tried all possible files")
    stock_names_cache[cache_key] = None
    return None

def get_date_stock_pairs(tick_path, limit=None):
    """
    直接從tick資料夾獲取所有日期和股票代碼
    返回格式: [(date, stock_code), ...]
    limit: 可選，限制處理的項目數
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
                    if limit and len(date_stock_pairs) >= limit:
                        return date_stock_pairs
    
    return date_stock_pairs

def process_tick_data(tick_path, date, stock_code):
    file_name = f"{stock_code}_{date}_ticks.csv"
    file_path = os.path.join(tick_path, date, file_name)
    
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return None
    
    try:
        # 只讀取需要的列以節省記憶體
        df = pd.read_csv(file_path, usecols=['ts', 'close', 'volume'])
        logging.info(f"Processing {file_name}, size: {len(df)}")
        
        if 'ts' not in df.columns:
            logging.warning(f"'ts' column not found in {file_name}")
            return None
        
        # 轉換時間並過濾交易時間
        df['datetime'] = pd.to_datetime(df['ts'])
        df = df[(df['datetime'].dt.time >= time(9, 0)) & (df['datetime'].dt.time <= time(13, 30))]
        
        # 確保數值型別正確
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # 移除無用的列以節省記憶體
        df.drop('ts', axis=1, inplace=True)
        
        # 排除無效資料
        df = df.dropna(subset=['close', 'volume'])
        
        return df
    except Exception as e:
        logging.error(f"Error processing {file_name}: {e}")
        return None

def generate_price_volume_details(df):
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        # 使用 numpy 來加速分組操作
        df_values = df[['close', 'volume']].values
        unique_prices = np.unique(df_values[:, 0])
        
        result = []
        for price in unique_prices:
            mask = df_values[:, 0] == price
            volume_sum = np.sum(df_values[mask, 1])
            result.append([price, volume_sum])
        
        price_volume = pd.DataFrame(result, columns=['close', 'volume'])
        price_volume = price_volume.sort_values('close', ascending=False)
        return price_volume
    except Exception as e:
        logging.error(f"Error generating price volume details: {e}")
        return pd.DataFrame()

def find_extreme_price_times(df, volume_threshold, max_volume):
    if df is None or df.empty:
        empty_df = pd.DataFrame({'price': ['N/A'], 'start_time': ['N/A'], 'end_time': ['N/A'], 'duration': ['N/A']})
        return empty_df, empty_df
    
    def track_price(price_func, comp_func):
        results = []
        current_volumes = {}
        current_tracking_price = None
        tracking_start_time = None
        
        # 將 DataFrame 轉換為 numpy 數組以加速處理
        sorted_df = df.sort_values('datetime')
        close_values = sorted_df['close'].values
        volume_values = sorted_df['volume'].values
        datetime_values = sorted_df['datetime'].values
        
        for i in range(len(sorted_df)):
            price = close_values[i]
            volume = volume_values[i]
            current_time = datetime_values[i]
            
            if price not in current_volumes:
                current_volumes[price] = 0
            current_volumes[price] += volume
            
            # 找出目前的最高價格
            max_price = max(current_volumes.keys())
            
            if current_tracking_price is None:
                if volume_threshold <= current_volumes[max_price] <= max_volume:
                    current_tracking_price = max_price
                    tracking_start_time = current_time
            else:
                if max_price > current_tracking_price:
                    if tracking_start_time is not None:
                        # 處理 numpy.timedelta64 對象
                        time_diff = current_time - tracking_start_time
                        if hasattr(time_diff, 'total_seconds'):
                            duration = time_diff.total_seconds()
                        else:
                            # numpy.timedelta64 轉換為秒
                            duration = time_diff / np.timedelta64(1, 's')
                        
                        results.append({
                            'price': current_tracking_price,
                            'start_time': tracking_start_time,
                            'end_time': current_time,
                            'duration': duration
                        })
                    current_tracking_price = None
                    tracking_start_time = None
                    if volume_threshold <= current_volumes[max_price] <= max_volume:
                        current_tracking_price = max_price
                        tracking_start_time = current_time
                elif current_volumes[current_tracking_price] > max_volume:
                    # 處理 numpy.timedelta64 對象
                    time_diff = current_time - tracking_start_time
                    if hasattr(time_diff, 'total_seconds'):
                        duration = time_diff.total_seconds()
                    else:
                        # numpy.timedelta64 轉換為秒
                        duration = time_diff / np.timedelta64(1, 's')
                    
                    results.append({
                        'price': current_tracking_price,
                        'start_time': tracking_start_time,
                        'end_time': current_time,
                        'duration': duration
                    })
                    current_tracking_price = None
                    tracking_start_time = None
        
        if current_tracking_price is not None and volume_threshold <= current_volumes[current_tracking_price] <= max_volume:
            last_time = datetime_values[-1]
            
            # 處理 numpy.timedelta64 對象
            time_diff = last_time - tracking_start_time
            if hasattr(time_diff, 'total_seconds'):
                duration = time_diff.total_seconds()
            else:
                # numpy.timedelta64 轉換為秒
                duration = time_diff / np.timedelta64(1, 's')
            
            results.append({
                'price': current_tracking_price,
                'start_time': tracking_start_time,
                'end_time': last_time,
                'duration': duration
            })
        
        # 根據開始時間排序結果
        results.sort(key=lambda x: x['start_time'])
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame({'price': ['N/A'], 'start_time': ['N/A'], 
                               'end_time': ['N/A'], 'duration': ['N/A']})
    
    try:
        max_summary = track_price(max, lambda x, y: x > y)
        min_summary = track_price(min, lambda x, y: x < y)
        
        return max_summary, min_summary
    except Exception as e:
        logging.error(f"Error finding extreme price times: {e}")
        empty_df = pd.DataFrame({'price': ['N/A'], 'start_time': ['N/A'], 
                               'end_time': ['N/A'], 'duration': ['N/A']})
        return empty_df, empty_df

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
    將長時間持續的記錄寫入CSV檔案，只包含指定欄位
    """
    import csv
    from datetime import datetime, time
    
    if not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)
        
    csv_file = os.path.join(output_csv_path, 'long_duration_signals.csv')
    
    try:
        # 使用鎖來確保多進程寫入安全
        lock = mp.Lock()
        with lock:
            file_exists = os.path.exists(csv_file)
            
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                formatted_date = f"{date_obj.year - 1911}/{date_obj.month:02d}/{date_obj.day:02d}"
            except ValueError:
                # 如果日期格式不是 '%Y-%m-%d'，嘗試其他格式
                try:
                    date_obj = datetime.strptime(date, '%Y_%m_%d')
                    formatted_date = f"{date_obj.year - 1911}/{date_obj.month:02d}/{date_obj.day:02d}"
                except ValueError:
                    logging.warning(f"無法解析日期格式: {date}，使用原始日期")
                    formatted_date = date
            
            # 確保股票名稱有值
            if stock_name is None or stock_name == '':
                # 從硬編碼字典中再次嘗試獲取股票名稱
                stock_code_map = {
                    "1503": "士電",
                    "6163": "華電網",
                    # 添加更多已知的股票代碼和名稱
                }
                stock_name = stock_code_map.get(stock_code, f"未知-{stock_code}")
                logging.info(f"CSV 輸出時使用備用股票名稱: {stock_name} 對於代碼 {stock_code}")
            
            with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    headers = ['日期', 'Stock Code', 'Stock Name', '盤中訊號出現時間', '訊號持續時間', '最高價格']
                    writer.writerow(headers)
                
                for price, start_time, end_time, duration, _ in long_duration_records:
                    # 處理不同格式的時間戳
                    if isinstance(start_time, pd.Timestamp):
                        signal_time = start_time.strftime('%H:%M:%S')
                    elif isinstance(start_time, np.datetime64):
                        signal_time = pd.Timestamp(start_time).strftime('%H:%M:%S')
                    elif isinstance(start_time, str) and ' ' in start_time:
                        try:
                            signal_time = start_time.split(' ')[1]
                        except IndexError:
                            signal_time = start_time
                    else:
                        signal_time = str(start_time)
                    
                    # 確保價格是有效的數值
                    try:
                        price_str = f"{float(price):.2f}"
                    except (ValueError, TypeError):
                        price_str = str(price)
                    
                    row = [
                        formatted_date,
                        stock_code,
                        stock_name,
                        signal_time,
                        duration,
                        price_str
                    ]
                    writer.writerow(row)
                    
            logging.info(f"成功將 {len(long_duration_records)} 筆記錄寫入 CSV，股票代碼: {stock_code}，名稱: {stock_name}")
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")
        import traceback
        logging.error(traceback.format_exc())

def write_to_file(file_path, stock_code, stock_name, date, price_volume_details, max_times, min_times, duration_threshold):
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    safe_filename = sanitize_filename(filename)
    safe_file_path = os.path.join(directory, safe_filename)
    
    long_duration_records = []
    
    try:
        with open(safe_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Stock: {stock_code} {stock_name if stock_name else ''}\n")
            f.write(f"Date: {date}\n\n")
            
            f.write("Price-Volume Details (Sorted by price, high to low):\n")
            if not price_volume_details.empty:
                f.write(" close  volume\n")
                try:
                    # 使用 map 函數代替 apply 以提高效率
                    price_volume_details['close'] = pd.to_numeric(price_volume_details['close'], errors='coerce')
                    price_volume_details['close'] = price_volume_details['close'].map(
                        lambda x: f" {x:>6.2f}" if pd.notnull(x) else "   N/A")
                    price_volume_details['volume'] = price_volume_details['volume'].map(lambda x: f"{x:>8}")
                    f.write(price_volume_details.to_string(index=False))
                except Exception as e:
                    logging.warning(f"Error formatting price-volume details for {stock_code}: {e}")
                    f.write("Error formatting price-volume details\n")
            f.write("\n\n")
            
            normal_duration_records = []
            
            for _, row in max_times.iterrows():
                if row['price'] == 'N/A':
                    continue
                
                try:
                    price = float(row['price']) if isinstance(row['price'], str) else row['price']
                    duration_seconds = float(row['duration'])
                    if duration_seconds == 0:
                        continue
                    
                    formatted_duration = format_duration(duration_seconds)
                except (ValueError, TypeError) as e:
                    logging.warning(f"Error processing record for {stock_code}: {e}")
                    continue
                
                record = (price, row['start_time'], row['end_time'], formatted_duration, duration_seconds)
                
                if duration_seconds > duration_threshold:
                    long_duration_records.append(record)
                else:
                    normal_duration_records.append(record)
            
            f.write("Times of max price with volume 1-10:\n")
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
    except Exception as e:
        logging.error(f"Error writing to file {safe_file_path}: {e}")
        return []
    
    return long_duration_records

def process_single_stock(args):
    """
    處理單一股票的函數，用於多進程處理
    """
    date_folder, formatted_date, stock_code, paths, params, stock_names_cache = args
    tick_data_path, output_path, pv_list_path, output_csv_path = paths
    volume_threshold, max_volume, duration_threshold = params
    
    try:
        logging.info(f"Processing date: {date_folder}, stock: {stock_code}")
        start_time = time_module.time()
        
        # 使用共享快取獲取股票名稱
        stock_name = get_stock_name(stock_code, formatted_date, pv_list_path, stock_names_cache)
        
        # 處理 tick 數據
        mem_before = get_memory_usage()
        tick_data = process_tick_data(tick_data_path, date_folder, stock_code)
        if tick_data is None:
            logging.info(f"No tick data found for {stock_code} on {date_folder}")
            return
        
        # 生成價格與成交量詳情
        price_volume_details = generate_price_volume_details(tick_data)
        max_volume_times, min_volume_times = find_extreme_price_times(tick_data, volume_threshold, max_volume)
        
        # 清除 tick_data 節省記憶體
        del tick_data
        gc.collect()
        
        mem_after = get_memory_usage()
        logging.debug(f"Memory usage: {mem_before:.2f}MB -> {mem_after:.2f}MB (diff: {mem_after-mem_before:.2f}MB)")
        
        if price_volume_details is not None and not price_volume_details.empty:
            file_name = f"{date_folder}_{stock_code}.txt"
            file_path = os.path.join(output_path, file_name)
            
            long_duration_records = write_to_file(file_path, stock_code, stock_name, date_folder,
                                                price_volume_details, max_volume_times, min_volume_times,
                                                duration_threshold)
            
            if long_duration_records:
                write_to_csv(date_folder, stock_code, stock_name, long_duration_records, output_csv_path)
            
            logging.info(f"Data written to {file_path}")
        else:
            logging.info(f"No valid data to write for {stock_code} on {date_folder}")
        
        # 記錄處理時間
        elapsed_time = time_module.time() - start_time
        logging.info(f"Completed {stock_code} on {date_folder} in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error processing {stock_code} on {date_folder}: {e}")
    
    # 強制清理記憶體
    gc.collect()
    return f"{date_folder}_{stock_code}"

def process_in_batches(process_args, batch_size, num_cores):
    """批次處理以避免記憶體溢出"""
    total_items = len(process_args)
    batches = [process_args[i:i + batch_size] for i in range(0, total_items, batch_size)]
    
    logging.info(f"Processing {total_items} items in {len(batches)} batches of {batch_size}")
    
    processed_count = 0
    for batch_idx, batch in enumerate(batches):
        logging.info(f"Starting batch {batch_idx+1}/{len(batches)}")
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {executor.submit(process_single_stock, args): args for args in batch}
            
            for future in tqdm(as_completed(futures), total=len(batch), desc=f"Batch {batch_idx+1}"):
                try:
                    stock_key = future.result()
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logging.info(f"Progress: {processed_count}/{total_items} ({(processed_count/total_items*100):.2f}%)")
                        logging.debug(f"Memory usage: {get_memory_usage():.2f}MB")
                except Exception as e:
                    args = futures[future]
                    date_folder, _, stock_code = args[0], args[1], args[2]
                    logging.error(f"Batch processing error for {date_folder}_{stock_code}: {e}")
        
        # 批次處理完成後強制回收記憶體
        gc.collect()
        logging.info(f"Completed batch {batch_idx+1}, memory usage: {get_memory_usage():.2f}MB")
    
    return processed_count

def main():
    start_time = time_module.time()
    
    # 路徑設定
    tick_data_path = r"D:\FTP_Jim\代碼備份\backtest\test\pv_tick"
    output_path = r"D:\FTP_Jim\代碼備份\backtest\test\output"
    pv_list_path = r"D:\FTP_Jim\代碼備份\backtest\test\pv_list"
    output_csv_path = r"D:\FTP_Jim\代碼備份\backtest\test\output\CSV"
    
    # 參數設定
    volume_threshold = 1
    max_volume = 10
    duration_threshold = 30
    
    # 新增的配置參數
    batch_size = 50  # 每批處理的股票數量
    
    # 手動設定 CPU 核心數量
    available_cores = mp.cpu_count()
    num_cores = min(12, available_cores - 1)  # 留一個核心給系統使用
    
    # 顯示系統總核心數和設定使用的核心數
    logging.info(f"System total CPU cores: {available_cores}")
    logging.info(f"Using {num_cores} CPU cores")
    logging.info(f"Initial memory usage: {get_memory_usage():.2f}MB")

    # 確保輸出目錄存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)

    # 獲取所有需要處理的日期和股票對
    date_stock_pairs = get_date_stock_pairs(tick_data_path)
    logging.info(f"Total pairs to process: {len(date_stock_pairs)}")
    
    # 準備多進程處理的參數
    paths = (tick_data_path, output_path, pv_list_path, output_csv_path)
    params = (volume_threshold, max_volume, duration_threshold)
    
    # 共享的股票名稱快取
    stock_names_cache = {}
    
    # 為每個股票準備參數
    process_args = [(date_folder, formatted_date, stock_code, paths, params, stock_names_cache) 
                   for date_folder, formatted_date, stock_code in date_stock_pairs]
                   
    # 預載股票名稱快取以提高效率
    logging.info("預先載入股票名稱快取...")
    for date_folder, formatted_date, stock_code in date_stock_pairs[:100]:  # 只預載前100個作為樣本
        get_stock_name(stock_code, formatted_date, pv_list_path, stock_names_cache)
    logging.info(f"預載完成，快取中有 {len(stock_names_cache)} 個股票名稱")
    
    try:
        # 使用批次處理策略
        processed_count = process_in_batches(process_args, batch_size, num_cores)
            
    except Exception as e:
        logging.error(f"Error in processing: {e}")
    
    total_time = time_module.time() - start_time
    logging.info(f"Processing completed in {total_time:.2f} seconds!")
    logging.info(f"Processed {processed_count} items out of {len(date_stock_pairs)}")
    logging.info(f"Final memory usage: {get_memory_usage():.2f}MB")

if __name__ == "__main__":
    # 在 Windows 上需要這個判斷來避免遞迴調用
    mp.freeze_support()
    main()