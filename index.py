import pandas as pd
from itertools import islice
from datetime import datetime, time, timedelta
import traceback
import os, math, logging, shutil, gc, psutil, signal, sys
from multiprocessing import Pool, Manager
from dotenv import load_dotenv
load_dotenv()
from module.in_out import StockDataAnalyzer

# 創建logger的方法
def setup_logger(stock_code, log_file):
    logger = logging.getLogger(str(stock_code))
    logger.setLevel(logging.INFO)

    # 創建handler，將log寫入到以股票代碼命名的檔案中，並設置編碼為'utf-8'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    
    # 設定log格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 將handler加到logger
    logger.addHandler(file_handler)
    
    return logger

# 判斷是否是有效日期
def get_valid_trading_dates(start_date, end_date, tick_dir):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    valid_dates = []
    current = start
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        date_dir = os.path.join(tick_dir, date_str)
        if os.path.exists(date_dir):
            valid_dates.append(date_str)
        else:
            print(f"Skipping date {date_str}: No data directory found in {date_dir}")
        current += timedelta(days=1)
    return valid_dates

#計算全局結果
def analyze_global_results(result_list, **params):
    try:
        # 按日期和策略类型分组结果
        result_groups = {}
        for result in result_list:
            date = result.get('date')
            strategy_type = 'Long' if params.get('long_short', True) else 'Short'
            dynamic_type = 'dynamic' if params.get('dynamic', False) else 'static'
            key = (date, strategy_type, dynamic_type)
            if key not in result_groups:
                result_groups[key] = []
            result_groups[key].append(result)
    
        # 对每组结果进行分析
        for (date, strategy_type, dynamic_type), results in result_groups.items():
            dir_path = os.path.join('result', date)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            total_stocks = len(results)
            if total_stocks == 0:
                print(f'No trading results for {date}, skipping analysis')
                continue
            
            win_count = sum(1 for result in results if 'pnl' in result and result['pnl'] > 0)
            total_pnl = sum(result['pnl'] for result in results if 'pnl' in result)

            win_rate = win_count / total_stocks * 100 if total_stocks > 0 else 0
            avg_pnl = total_pnl / total_stocks if total_stocks > 0 else 0

            # 全局持仓时间和损益统计
            total_avg_duration = sum(result.get('avg_duration', 0) for result in results) / total_stocks
            total_max_duration = max(result.get('max_duration', 0) for result in results)
            total_min_duration = min(result.get('min_duration', float('inf')) for result in results)
            total_max_profit = max(result.get('max_profit', 0) for result in results)
            total_max_loss = max(result.get('max_loss', 0) for result in results)

            # 写入到文件
            filename = f"{strategy_type}_{dynamic_type}_{date}.txt"
            with open(os.path.join(dir_path, filename), 'w', encoding='utf-8') as f:
                f.write('------------------- 股票池结果 -------------------\n')

                total_pnl = 0
                for result in results:
                    if 'stock_code' in result and 'pnl' in result:
                        pnl = round(result['pnl'])
                        f.write(f"股票代號:{result['stock_code']}, 交易後餘額:{result['remain']}, 盈虧:{pnl}\n")
                        total_pnl += pnl

                f.write(f"\n今天的總盈虧為: {round(total_pnl)}\n")

                f.write('------------------- 全局统计 -------------------\n')  
                f.write(f"总回测股票数量: {total_stocks}\n")
                f.write(f"赚钱股票数量: {win_count}\n")
                f.write(f"胜率: {win_rate:.2f}%\n")
                f.write(f"平均盈亏: {avg_pnl:.2f}\n")
                f.write(f"总平均持仓时间: {round(total_avg_duration, 2)}\n")  
                f.write(f"总最大持仓时间: {round(total_max_duration, 2)}\n")  
                f.write(f"总最小持仓时间: {round(total_min_duration, 2)}\n")  
                f.write(f"总最大盈利: {round(total_max_profit, 2)}\n")  
                f.write(f"总最大亏损: {round(total_max_loss,2)}\n")
                f.write(f"策略类型: {'做多' if params.get('long_short', True) else '做空'}\n")
                f.write(f"止盈止损类型: {'动态' if params.get('dynamic', False) else '静态'}\n")
                f.write(f"止盈: {(params.get('profit_ratio', 0)) * 100}%, 止损: {(params.get('loss_ratio', 0)) * 100}%\n")
                f.write(f"进场Ticks价差数: {params.get('ticks', 0)}\n")
                f.write(f"成交量门槛: {params.get('volume_threshold', 0)}\n")
                f.write(f"成交量观察秒数: {params.get('period', 0)}\n")
                f.write(f"平仓时间: {params.get('last_trade_hour', 0)}:{params.get('last_trade_minute', 0)} \n")

        print('本次交易參數:', params)
    except Exception as e:
        print(f'自訂分析結果Analyze_global_result出錯: {e}')

# 百分比進場
def pct_entry(price, pct, type='long'):
    if type == 'short':
        price = price * (1 - pct)
        
         # 根据当前价格判断 tick_size
        if price > 100:
            tick_size = 0.5
        elif 50 <= price <= 100:
            tick_size = 0.1
        else:
            tick_size = 0.05

        price = math.floor(price / tick_size) * tick_size
        
    elif type =='long':
        price = price * (1 + pct)
        
        # 根据当前价格判断 tick_size
        if price > 100:
            tick_size = 0.5
        elif 50 <= price <= 100:
            tick_size = 0.1
        else:
            tick_size = 0.05       
        
        price = math.ceil(price / tick_size) * tick_size
        
    return round(price, 2)

# tick進場(獲取價格相對應的tick)
def calculate_ticks(price, ticks, type='long'):
    result_price = price
    remaining_ticks = ticks
    
    while remaining_ticks > 0: # 获取当前价格对应的 tick_size
        if result_price > 100:
            tick_size = 0.5
        elif 50 <= result_price <= 100:
            tick_size = 0.1
        else:
            tick_size = 0.05

        # 计算下一个跳动的价格
        if type == 'short':
            result_price -= tick_size
        elif type =='long':
            result_price += tick_size
        remaining_ticks -= 1

    return round(result_price, 2)

# 获取動態止盈止損tick數值
def calculate_price_with_dynamic_ticks(target_price, direction='up'):
    """
    根据价格区间动态调整价格，考虑 tick_size 的变化。
    :param target_price: 目标价格
    :param direction: 调整方向，'up'為空 表示向上调整，'down'為多, 表示向下调整
    :return: 动态调整后的价格
    """
    current_price = target_price

    while True:
        # 根据当前价格判断 tick_size
        if current_price > 100:
            tick_size = 0.5
        elif 50 <= current_price <= 100:
            tick_size = 0.1
        else:
            tick_size = 0.05

        # 根据方向进行价格调整
        if direction == 'up':
            new_price = math.ceil(current_price / tick_size) * tick_size # 空
        else:
            new_price = math.floor(current_price / tick_size) * tick_size # 多

        # 如果价格不再变化，退出循环
        if new_price == current_price:
            break
            
        current_price = new_price

    return round(current_price, 2)

# 動態止盈止損計算
def dynamic_order_caculate(price1, price2, limit_up, limit_down, type='long', **params):
    """
    动态计算止损与止盈价格，考虑不同价格区间的 tick_size。
    :param price1: 止損金額
    :param price2: 止盈金額
    :param type: 'long' or 'short' 方向
    :return: 止损价格，止盈价格
    """
    if type == 'long':
        # 动态计算止损价格，向上调整
        stop_loss_price = calculate_price_with_dynamic_ticks(price1 * (1 - params.get('loss_ratio', 0.01)), 'up')
        # 动态计算止盈价格，向下调整
        profit_stop_price = calculate_price_with_dynamic_ticks(price2 * (1 + params.get('profit_ratio', 0.03)), 'down')
        return stop_loss_price, min(limit_up, profit_stop_price)
    else:
        # 动态计算止损价格，向下调整
        stop_loss_price = calculate_price_with_dynamic_ticks(price1 * (1 + params.get('loss_ratio', 0.01)), 'down')
        # 动态计算止盈价格，向上调整
        profit_stop_price = calculate_price_with_dynamic_ticks(price2 * (1 - params.get('profit_ratio', 0.03)), 'up')
        return stop_loss_price, max(limit_down, profit_stop_price)

# 部位計算
def sizer(init_cash, entry_price, unit_size=1000):
    """
    計算能買入的整數張數（1000股為一張）
    :param init_cash: 可用資金
    :param entry_price: 進場價格
    :param unit_size: 每張股票的單位股數 (默認為1000股)
    :return: (可買入整數張數的股數, 總成本)
    """
    # 計算能買入的整數張數
    return (init_cash // (entry_price * unit_size)) * unit_size

# 價格依照tick級距進行調整
def adjust_price(price):
    """ 根據給定的百分比調整價格。 """
    # 修正價格的邏輯
    if 10 <= price < 50:
        return round(math.floor(price / 0.05) * 0.05, 2)
    elif 50 <= price < 100:
        return round(math.floor(price / 0.1) * 0.1, 2)
    elif price >= 100:
        return round(math.floor(price / 0.5) * 0.5, 1)
    else:
        return round(price, 2)

# 動態止盈利止損
def dynamic(i, r, trades, logger, type='long', **params):
    # 獲取止盈和止損的比例
    profit_ratio = params.get('profit_ratio', 0.03)
    loss_ratio = params.get('loss_ratio', 0.01)
    
    # 獲取手續費、稅率和初始資金
    init_cash = params.get('initial_cash', 10000)  # 初始資金
    commision = params.get('commision', 0.001425)  # 手續費率
    tax = params.get('tax', 0.003)  # 稅率
    
    # 獲取進場時間, 股數, 進場價格, 動態止盈價(price), 動態止損價(price)
    entry_time, shares, entry_price, profit_price, loss_price = r['position']
    
    # 獲取最高和最低價格限制
    limit_up = r['limit_up']
    limit_down = r['limit_down']
    
    if type == 'short':
        if r['price_volume_pairs'] == 0: return 0, 0 # 因為是填充的沒有數值故跳過
        hp, _ = r['price_volume_pairs'][0]

        if profit_price == 0 and loss_price == 0: # 判斷是否有止盈止損價, 若沒有則使用entry_price
            stop_loss_price, stop_profit_price = dynamic_order_caculate(entry_price, entry_price, limit_up, limit_down, type, **params) 
        else: # 判斷是否有止盈止損價, 若已經計算過, 則使用原本的價錢
            stop_loss_price, stop_profit_price = loss_price, profit_price

        # 當指損價>最高價, 使用最高價當止損價
        if stop_loss_price > hp: stop_loss_price = hp
        
        logger.info(f'時間:{i}, 做空(動態), 止盈:{stop_profit_price}, 止損:{stop_loss_price}, 止盈%數:{profit_ratio}, 止損%數:{loss_ratio}')
        
        # 獲取當前價格ask_price價格
        current_ask_price = r['ask_price'][0]

        # 檢查是否觸發止盈或止損
        if current_ask_price <= stop_profit_price and stop_profit_price != limit_down:
             # 觸發止盈, 更新止盈止損
            new_stop_loss_price, new_stop_profit_price = dynamic_order_caculate(stop_profit_price, stop_profit_price, limit_up, limit_down, type, **params)
            logger.info(f'時間:{i}, 觸發止盈價, 做空動態更新, 當前金額:{current_ask_price}, 更新止盈:{new_stop_profit_price}, 更新止損:{new_stop_loss_price}')
            return 1, (i, shares, entry_price, new_stop_profit_price, new_stop_loss_price)
        elif current_ask_price == 0 and stop_profit_price <= limit_down: # 跌停
            exit_price = r['close']
            logger.info(f'時間:{i}, 跌停, 出場金額:{exit_price}, 止損:{stop_loss_price}')
        elif current_ask_price >= stop_loss_price:
            # 觸發止損
            exit_price = current_ask_price
            logger.info(f'時間:{i}, 觸發止損價, 當前金額:{current_ask_price}, 止損:{stop_loss_price}')
        else: # 未觸發任何訊號
            logger.info(f'時間:{i}, 做空動態當前價格{current_ask_price}都未觸發止盈或止損\n')
            return 2, 0
        
        # 計算進場和出場的手續費與稅費（進場只計手續費與稅，出場計手續費）
        entry_trade_value = entry_price * shares
        exit_trade_value = exit_price * shares

        # 進場時手續費（做空）
        entry_fee = entry_trade_value * commision
        entry_tax = entry_trade_value * tax

        # 出場時的手續費與稅金
        exit_fee = exit_trade_value * commision
        
        # 計算總手續費與稅金
        total_fees = entry_fee + exit_fee + entry_tax

    # 計算持倉時間
    holding_time = i - entry_time

    # 計算盈虧：每股的盈虧 * 股數，再扣除手續費和稅金
    pnl = round((entry_price - exit_price) * shares - total_fees, 2)
    
    exit_type = 'profit' if pnl > 0 else 'loss'

    # 更新可用資金（做空時）
    init_cash += pnl  # 更新總資金，考慮盈利/虧損

    # 記錄交易信息到 trades 列表
    trades.append({
        'entry_time': entry_time,
        'exit_time': i,
        'shares': shares,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'holding_time': holding_time,
        'pnl': pnl,
        'total_fees': total_fees,
        'final_cash': init_cash,  # 記錄交易後的總資金
        'exit_type': exit_type
    })
    
    logger.info(f"做空動態平倉: 時間={i}, 狀態:{exit_type}, 出場價格:{exit_price}, 盈虧:{pnl}, 手續費:{total_fees}, 剩餘資金:{init_cash}\n")

    return 3, 0

# 靜態止盈利止損
def static(i, r, trades, logger, type='long', **params):
    """
    靜態止盈止損檢查，考慮手續費、稅費，並更新總資產
    :param i: 當前時間
    :param r: 當前行的資料 (包括價格、倉位等)
    :param trades: 交易紀錄的列表
    :param params: 參數列表，包含止盈止損比例等
    """
    # 獲取止盈和止損的比例
    profit_ratio = params.get('profit_ratio', 0.03)
    loss_ratio = params.get('loss_ratio', 0.01)
    
    # 獲取手續費、稅率和初始資金
    init_cash = params.get('initial_cash', 10000)  # 初始資金
    commision = params.get('commision', 0.001425)  # 手續費率
    tax = params.get('tax', 0.003)  # 稅率
    
    # 獲取進場的價格和股數
    entry_time, shares, entry_price, _, _ = r['position']  # position 包含進場時間, 股數, 進場價格

    if type == 'short':
        if r['price_volume_pairs'] == 0: return 0 # 因為是填充的沒有數值故跳過
        
        # 計算止盈和止損的價格範圍 
        stop_profit_price = max(adjust_price(entry_price * (1 - profit_ratio)), r['limit_down']) # 止盈價    
        stop_loss_price = min(adjust_price(entry_price * (1 + loss_ratio)), r['limit_up']) # 止損價
        logger.info(f'時間:{i}, 做空(靜態), 止盈:{stop_profit_price}, 止損:{stop_loss_price}, 止盈%數:{profit_ratio}, 止損%數:{loss_ratio}')
        
        # 獲取當前價格ask_price價格
        current_ask_price = r['ask_price'][0]

        # 檢查是否觸發止盈或止損
        if current_ask_price <= stop_profit_price:
            # 觸發止盈
            exit_price = current_ask_price
            exit_type = 'profit'  # 記錄是止盈
            logger.info(f'時間:{i}, 當前金額:{current_ask_price}, 做空靜態止盈:{stop_profit_price}')
        
        elif current_ask_price >= stop_loss_price:
            # 觸發止損
            exit_price = current_ask_price
            exit_type = 'loss'    # 記錄是止損
            logger.info(f'時間:{i}, 當前金額:{current_ask_price}, 做空靜態止損:{stop_loss_price}')
        
        else: # 未觸發止盈止損，返回不操作
            logger.info(f'時間:{i}, 做空靜態當前價格{current_ask_price}都未觸發止盈或止損\n')
            return 0

        # 計算進場和出場的手續費與稅費（進場只計手續費，出場計手續費與稅）
        entry_trade_value = entry_price * shares
        exit_trade_value = exit_price * shares

        # 進場時手續費（做空）
        entry_fee = entry_trade_value * commision
        entry_tax = entry_trade_value * tax

        # 出場時的手續費與稅金
        exit_fee = exit_trade_value * commision
        
        # 計算總手續費與稅金
        total_fees = entry_fee + exit_fee + entry_tax

    # 計算持倉時間
    holding_time = i - entry_time

    # 計算盈虧：每股的盈虧 * 股數，再扣除手續費和稅金
    pnl = round((entry_price - exit_price) * shares - total_fees, 2)

    # 更新可用資金（做空時）
    init_cash += pnl  # 更新總資金，考慮盈利/虧損

    # 記錄交易信息到 trades 列表
    trades.append({
        'entry_time': entry_time,
        'exit_time': i,
        'shares': shares,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'holding_time': holding_time,
        'pnl': pnl,
        'total_fees': total_fees,
        'final_cash': init_cash,  # 記錄交易後的總資金
        'exit_type': exit_type
    })
    
    logger.info(f"做空靜態平倉: 時間={i}, 狀態:{exit_type}, 出場價格:{exit_price}, 盈虧:{pnl}, 手續費:{total_fees}, 剩餘資金:{init_cash}\n")
    
    return

# 平倉
def close(i, r, trades, logger, type='long', **params):
    init_cash = params.get('initial_cash', 10000)
    commision = params.get('commision', 0.001425)  # 手續費率
    tax = params.get('tax', 0.003)  # 稅率（賣出時適用）
    
    if type == 'short':
        if r['price_volume_pairs'] == 0: return 0  # 因為是填充的沒有數值故跳過
        exit_price = r['ask_price'][0]  # 以當前的ask_price平倉
        entry_time, shares, entry_price, _, _ = r['position']  # 解構倉位
                
        # 計算平倉總價值
        trade_value = exit_price * shares  
                
        # 計算平倉手續費
        exit_fee = trade_value * commision
                
        # 計算進場手續費+稅
        entry_fee = entry_price * shares * (commision + tax)
                
        # 計算盈虧
        pnl = round((entry_price - exit_price) * shares - entry_fee - exit_fee, 2)  # 扣除費用的盈虧
        
        # 總手續費：進場手續費+稅 + 出場手續費
        total_fees = entry_fee + exit_fee  # 總手續費（含進場稅）
    
    # 計算持倉時間
    holding_time = i - entry_time

    # 計算最終資金：加入盈利或虧損，並減去總手續費
    init_cash += pnl  # 最終資金只需加上 PNL，因為已經在計算中考慮了費用
                
    # 設定 exit_type 為 "forced_close"（平倉）
    exit_type = 'forced_close'

    # 記錄交易信息到 trades 列表
    trades.append({
        'entry_time': entry_time,
        'exit_time': i,
        'shares': shares,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'holding_time': holding_time,
        'pnl': pnl,
        'total_fees': total_fees,
        'final_cash': init_cash,  # 記錄交易後的總資金
        'exit_type': exit_type
    })

    logger.info(f"超時平倉: 時間={i}, 出場價格:{exit_price}, 盈虧:{pnl}, 手續費:{total_fees}, 剩餘資金:{init_cash}")
    
    return 1

# 做多回測
def long(df, stock_logger, **params):
    pass

# 做空回測
def short(df, stock_logger, **params):
    try:
        init_cash = params.get('initial_cash', 10000)

        # 交易使用參數
        last_trade_time = time(params.get('last_trade_hour', 13), params.get('last_trade_minute', 0)) # 最後交易時間
        close_position_time = time(params.get('close_position_hour', 13), params.get('close_position_miinute', 0)) # 平倉時間

        # 初始化交易記錄列表，儲存每次交易的資訊
        trades = []
        entered_trade = False  # 用來記錄是否已經進場
        df['position'] = df['position'].astype(object)
        position = df['position'].copy()
        signal_start_time = None # 紀錄訊號出現後過了多少秒, 超過30訊號就作廢
        current_highest_price = None  # 訊號出現時的價錢, 超過30秒則判斷最高價是否刷新

        for i, r in df.iterrows():
            if i.time() > close_position_time:
                if position.loc[i] != 0:  # 檢查是否有持倉
                    df.at[i, 'position'] = position.at[i]
                    status = close(i, df.loc[i], trades, stock_logger, type='short', **params)

                    if status == 1: # 清除當前持倉
                        position.loc[i:] = 0
                        break

            # 有倉位
            if position.loc[i] != 0:
                if params.get('dynamic', False) is False:
                    df.at[i, 'position'] = position.at[i]
                    status = static(i, df.loc[i], trades, stock_logger, type='short', **params)

                    if status != 0: # 已因為觸發止盈止損出場
                        position.loc[i:] = 0
                else:
                    df.at[i, 'position'] = position.at[i]
                    status, info = dynamic(i, df.loc[i], trades, stock_logger, type='short', **params)

                    if status == 0 or status == 2: continue

                    if status == 1: # 更新當前倉位(調整後的止盈止損)
                        position.loc[i:] = [info] * len(position.loc[i:])

                    if status == 3: # 已因為觸發止盈止損出場
                        position.loc[i:] = 0

            # 無倉位判斷是否進場
            if not entered_trade and r['short'] == -1 and position.loc[i] == 0 and i.time() < last_trade_time:
                entry_price = r['price_volume_pairs'][0][0] # 取得當前時刻的價量(最高價, 量)
                bid_price = r['bid_price'][0] # 取得最高的bid_price
                
                if signal_start_time is None:  # 開始記錄訊號出現時間
                    signal_start_time = i
                    initial_highest_price = entry_price
                    
                # 若訊號出現已超過30秒
                if (i - signal_start_time).seconds > 30:
                    if entry_price > initial_highest_price:  # 情況1：價格有刷新
                        stock_logger.info(f"時間:{i}, 最高價已刷新從{initial_highest_price}到{entry_price}，重置計時器")
                        signal_start_time = i  # 重置計時器
                        initial_highest_price = entry_price  # 更新最高價
                    else:  # 情況2：價格無刷新
                        stock_logger.info(f"時間:{i}, 訊號已超過30秒且最高價{initial_highest_price}無刷新，訊號作廢\n")
                        continue
                
                stock_logger.info(f"時間:{i}, 訊號起始:{i - pd.Timedelta(seconds=params.get('period', 30))}, 當前最高價{entry_price}與量{r['price_volume_pairs'][0][1]}, 斷是否進場")

                if params.get('is_pct', False) is True:
                    entry_price = pct_entry(entry_price, pct=params.get('pct', 0.01), type='short')
                else:
                    entry_price = calculate_ticks(entry_price, ticks=params.get('ticks', 5), type='short')

                if entry_price <= bid_price: # 最高價小於等於bid_price
                    # 使用 sizer 函數來計算可買入的股數
                    shares = sizer(init_cash, r['price_volume_pairs'][0][0])
                    # 確認實際買入的價格與股數
                    trade_value = bid_price * shares  # 進場價格 = bid_price * 股數

                    # 從初始資金中扣除
                    init_cash -= trade_value

                    # 記錄資金變化到 stock_logger
                    stock_logger.info(f"進場: 時間={i}, 股數={shares}, 進場價格={entry_price}, 當前委買={bid_price} 進場總價值={trade_value}, 剩餘資金={init_cash}\n")

                    # 記錄進場的時間(entry_time)、股數(shares)、進場價格(price)、動態止盈價(price)、動態止損價(price)
                    position.loc[i:] = [(i, shares, bid_price, 0, 0)] * len(position.loc[i:])

                    # 標記已進場過
                    entered_trade = True
                    
                else: # 未進場原因顯示
                    stock_logger.info(f"時間:{i}, 當前使用{'百分比' if params.get('is_pct', False) else 'tick'}進場, 金額{entry_price}未小於委買{bid_price}, 故不進場\n")

        # 迴圈結束, 將最後的數值回寫
        df['position'] = position

        return trades
    except Exception as e:
        print(f'Something went wrong on short(): {e}')

# 分析交易結果
def analyzer(result, stock_code, stock_logger, **params):
    try:
        if not result:
            stock_logger.info(f"統計交易結果, 没有交易记录, 交易0次")
            return

        data2append = {'stock_code': stock_code}
        
        # 提取交易數據
        profits = [trade['pnl'] for trade in result]
        durations = [trade['holding_time'].total_seconds() for trade in result]  # 持仓时间单位为秒

        # 计算损益比
        total_profit = sum(p for p in profits if p > 0)
        total_loss = -sum(p for p in profits if p < 0)

        profit_loss_ratio = 0 if len(profits) == 1 else (total_profit / total_loss if total_loss != 0 else float('inf'))

        # 计算单笔最大盈利和最大亏损
        max_profit = max(profits)
        max_loss = min(profits)

        # 计算平均、最长、最短持仓时间
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations)
        min_duration = min(durations)

        # 获取当前日期
        date = result[-1]['exit_time'].date()  # 假设最后一笔交易时间为交易日期

        # 输出统计结果
        stock_logger.info(f"日期: {date}, 代號: {stock_code}")
        stock_logger.info(f"损益比: {profit_loss_ratio:.2f}")
        stock_logger.info(f"单笔最大盈利: {max_profit:.2f}")
        stock_logger.info(f"单笔最大亏损: {max_loss:.2f}")
        stock_logger.info(f"平均持仓时间: {avg_duration:.2f} 秒")
        stock_logger.info(f"最长持仓时间: {max_duration:.2f} 秒")
        stock_logger.info(f"最短持仓时间: {min_duration:.2f} 秒\n")

        data2append['date'] = date
        data2append['pnl'] = sum(trade['pnl'] for trade in result)
        data2append['remain'] = round(result[-1]['final_cash'])
        data2append['profit_loss_ratio'] = profit_loss_ratio
        data2append['max_profit'] = max_profit
        data2append['max_loss'] = max_loss
        data2append['avg_duration'] = avg_duration
        data2append['max_duration'] = max_duration
        data2append['min_duration'] = min_duration

        return data2append
    except Exception as e:
        print(f'While doing the analyzer is failed: {e}')

#生成訊號
def generate_singal(stock_code, combined_df, date_to_test, **params):
    try:
        # 生成完整的時間範圍，這裡不包含日期
        start_time = pd.to_datetime(date_to_test + ' 09:00:00')
        end_time = pd.to_datetime(date_to_test + ' 13:30:00')
        # 创建时间索引
        time_index = pd.date_range(start=start_time, end=end_time, freq='s')

        # 創建 DataFrame，使用時間作為索引
        df2 = pd.DataFrame(index=time_index)

        # 通过索引时间进行分组
        grouped = combined_df.groupby(combined_df.index)

        # ------------- 漲跌停計算 -----------------
        # 取得第一筆的 close 值
        initial_price = combined_df.iloc[0]['close']
        
        # 使用初始價格計算漲跌停限制
        price_limits = {
            'limit_up': round((math.floor(initial_price * (1 + 0.10) / 0.05) * 0.05), 2) if 10 <= initial_price < 50 else
                        round((math.floor(initial_price * (1 + 0.10) / 0.1) * 0.1), 2) if 50 <= initial_price < 100 else
                        round((math.floor(initial_price * (1 + 0.10) / 0.5) * 0.5), 1) if initial_price >= 100 else
                        round(initial_price * (1 + 0.10), 2),

            'limit_down': round((math.ceil(initial_price * (1 - 0.10) / 0.05) * 0.05), 2) if 10 <= initial_price < 50 else
                          round((math.ceil(initial_price * (1 - 0.10) / 0.1) * 0.1), 2) if 50 <= initial_price < 100 else
                          round((math.ceil(initial_price * (1 - 0.10) / 0.5) * 0.5), 1) if initial_price >= 100 else
                          round(initial_price * (1 - 0.10), 2),

            'limit_long_danger': math.floor(initial_price * (1 + 0.07)),
            'limit_short_danger': math.ceil(initial_price * (1 - 0.07))
        }

        # 將 price_limits 的值擴展到與 grouped 的大小一致，並創建一個新的 DataFrame
        price_limits_df = pd.DataFrame([price_limits] * len(grouped), index=grouped.groups.keys())
        
        # ------------- 開始建立價量表 -----------------

        # 從第二個分組開始
        slice_group = islice(grouped, 1, None)

        # 建立一個新列表，用來存放結果
        results = []

        # 初始化一個空字典用來保存前一秒的結果
        previous_prices = {}

        # 對每個分組進行價量計算
        for name, group in slice_group:
            price_volume_dict = {}

            # 遍歷該秒內的所有交易
            for _, row in group.iterrows():
                price = row['close']
                volume = row['volume']

                # 如果該價格已經存在於字典中，則累加成交量
                if price in price_volume_dict:
                    price_volume_dict[price] += volume
                else:
                    price_volume_dict[price] = volume

            # 從第二個分組開始進行累加
            if previous_prices:  # 如果不是第一個分組
                for price, vol in price_volume_dict.items():
                    if price in previous_prices:
                        previous_prices[price] += vol  # 累加成交量
                    else:
                        previous_prices[price] = vol  # 新增價格和成交量
            else:
                # 如果是第一個分組，直接將其加入到 previous_prices
                previous_prices = price_volume_dict.copy()

            # 對累加結果進行排序
            sorted_prices = sorted(previous_prices.items(), key=lambda x: x[0], reverse=True)

            # 合并 bid_price 和 ask_price，去重，并从大到小排序
            bid_prices = sorted(group['bid_price'].unique(), reverse=True)  # 获取唯一的 bid_price 并从大到小排序
            ask_prices = sorted(group['ask_price'].unique(), reverse=True)  # 获取唯一的 ask_price 并从大到小排序

            # 將結果以 tuple 的形式存放，並添加到結果列表中
            last_row = group.iloc[-1]
            results.append((name, sorted_prices, last_row['close'], last_row['volume'], bid_prices, last_row['bid_volume'], ask_prices, last_row['ask_volume'], last_row['open']))

            # 更新前一秒的價格記錄
            previous_prices = dict(sorted_prices)  # 將排序後的結果轉換為字典以供下次使用

        # 將結果轉換為 DataFrame，並創建對應的列
        price_vol_df = pd.DataFrame(results, columns=['datetime', 'price_volume_pairs', 'close', 'volume', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'open'])
        price_vol_df['datetime'] = pd.to_datetime(price_vol_df['datetime'])
        price_vol_df.set_index('datetime', inplace=True) # 設置時間為索引

        # ------------- 开始生成信号 -----------------

        # 使用 merge 合併 df2 和 price_vol_df，然后将 price_limits_df 合并到 merged_df
        merged_df = pd.merge(df2, price_vol_df, left_index=True, right_index=True, how='left')
        merged_df = pd.merge(merged_df, price_limits_df, left_index=True, right_index=True, how='left')

        merged_df.fillna(0, inplace=True)

        # 將信號寫回 DataFrame
        merged_df['long'], merged_df['short'] = 0, 0  # 初始化多頭, 空頭信號列
        long_signals, short_signals = [], [] # 空頭多頭信號

        # 設置參數，例如 period 為 30，成交量閾值為 10
        period = params.get('period', 30)
        vol_threshold = params.get('volume_threshold', 10)

        # ----------------- 做空訊號 -----------------
        # 初始化最高價和其位置
        max_price = -float('inf')
        max_price_index = 0

        for i, row in merged_df.iterrows():
            p_v = row['price_volume_pairs']

            # 如果 price_volume_pairs 是空的，則跳過
            if p_v == 0 or len(p_v) == 0:
                short_signals.append((i, 0))  # 無效信號設為0
                max_price_index += 1 # 累加 max_price_index
                continue
            
            if p_v != 0 and p_v[-1][0] < row['limit_short_danger']:
                short_signals.append((i, 0))  # 無效信號設為0
                max_price_index += 1 # 累加 max_price_index
                continue

            cp, cv = p_v[0] # 當前時刻的價格與成交量

            # 判別最高價是否刷新
            if cp > max_price or cv > vol_threshold:
                max_price = cp  # 更新最高價
                max_price_index = 0  # 規0重計算
                continue

            # 如果最高價未刷新，則累加 max_price_index
            max_price_index += 1

            # 當 max_price_index 達到 period 時，檢查成交量
            if max_price_index >= period:
                short_signals.append((i, -1))  # 做空信號和 tick_type 總和
            else:
                short_signals.append((i, 0))  # 如果未達到 period，則先標記無效信號
                
        # ----------------- 做多訊號 -----------------


        # ----------------- 多空統計 -----------------

        # 將空頭信號寫回 DataFrame
        for s_signal in short_signals:
            timestamp = s_signal[0]
            merged_df.loc[timestamp, 'short'] = s_signal[1]  # 寫入空頭信號

        # 將多頭信號寫回 DataFrame
        for l_signal in long_signals:
            timestamp = l_signal[0]
            merged_df.loc[timestamp, 'long'] = l_signal[1]  # 寫入多頭信號
    
        merged_df.fillna(0, inplace=True)
        merged_df['position'] = 0
        
        # 創建singal文件夾
        output_dir = f'./signal/{date_to_test}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 儲存 signal CSV 檔案
        merged_df.to_csv(f'{output_dir}/{stock_code}.csv')

        return merged_df
    except Exception as e:
        print(f'Loading and Generate {date_to_test} backtest singal {combined_df} is failed with :{e}')

# 回測主函數
def start_backtest(df, stock_code, stock_logger, **params):
    stock_logger.info(f'The current stock code is: {stock_code}')
    
    if df.empty:
        stock_logger.info(f'Empty dataframe for stock {stock_code}. Skipping.')
        return {}
        
    if params.get('long_short')  == False:
        result = short(df, stock_logger, **params)
    else:
        result = long(df, stock_logger, **params)
    
    if not result:
        stock_logger.info(f"没有交易记录, 交易0次\n")
        return {}
    
    # 调用 analyzer 函数统计结果
    return analyzer(result, stock_code, stock_logger, **params)

# 加載股票
def process_stock(stock_code, date_to_test, shared_result, tick_dir, lock, params):
    tick_file = os.path.join(tick_dir, date_to_test, f"{stock_code}_{date_to_test}_ticks.csv").replace('\\', '/')
    openprice_file = os.path.join(tick_dir, date_to_test, f"{stock_code}_{date_to_test}_openprice.csv").replace('\\', '/')
    
    log_file = f'{stock_code}_trading_log.log'
    stock_logger = setup_logger(stock_code, log_file)

    proc = psutil.Process()
    stock_logger.info(f"Process {proc.pid} (Stock: {stock_code}) starting. Process CPU usage: {proc.cpu_percent()}%, Total system CPU usage: {psutil.cpu_percent()}%, Memory usage: {proc.memory_info().rss / (1024 * 1024):.2f} MB, System RAM usage: {psutil.virtual_memory().percent}%\n")
    
    if os.path.exists(tick_file) and os.path.exists(openprice_file):
        try:
            # 讀取tick和openprice資料
            tick_df = pd.read_csv(tick_file)
            openprice_df = pd.read_csv(openprice_file)

            # 使用pandas.concat將openprice的資料加在tick資料的最前面
            if len(openprice_df) == 1:
                combined_df = pd.concat([openprice_df, tick_df], ignore_index=True)
                combined_df['ts'] = pd.to_datetime(combined_df['ts'])
                combined_df.rename(columns={'ts': 'datetime'}, inplace=True)
                combined_df['open'] = combined_df['close'].shift(1)
                combined_df['open'].fillna(combined_df['close'].iloc[0])
                combined_df.set_index('datetime', inplace=True)

            singal_df = generate_singal(stock_code=stock_code, combined_df=combined_df, date_to_test=date_to_test, **params)

            # ------------- 進行回測 -------------------
            bt_result = start_backtest(df=singal_df, stock_code=stock_code, stock_logger=stock_logger, **params)
            stock_logger.info(f"Process {proc.pid} (Stock: {stock_code}) finished. Process CPU usage: {proc.cpu_percent()}%, Total system CPU usage: {psutil.cpu_percent()}%, Memory usage: {proc.memory_info().rss / (1024 * 1024):.2f} MB, System RAM usage: { psutil.virtual_memory().percent}%")
            
            with lock:
                for handler in stock_logger.handlers:
                    handler.close()
                    stock_logger.removeHandler(handler)

                result_dir = os.path.join('result', f"{date_to_test}")
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                code_dir = os.path.join(result_dir, f"{date_to_test}_{stock_code}_{'Long' if params['long_short'] else 'Short'}_{'dynamic' if params['dynamic'] else 'static'}")
                os.makedirs(code_dir, exist_ok=True)

                try:
                    if os.path.exists(log_file):
                        shutil.move(log_file, os.path.join(code_dir, log_file))
                    else:
                        print(f"{log_file} 不存在")
                except IOError as e:
                    print(f"File move error for {stock_code}: {e}")

                bt_result['result_dir'] = result_dir
                bt_result['date'] = date_to_test

                shared_result.append(bt_result)
                
                # 删除变量并强制垃圾回收
                del tick_df, openprice_df, singal_df, bt_result
                gc.collect()  # 强制进行垃圾回收

        except pd.errors.EmptyDataError:
            print(f"Error processing stock {stock_code} for date {date_to_test}: Empty data file")
        except Exception as e:
            print(f"Unexpected error processing stock {stock_code} for date {date_to_test}: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
    else:
        print(f"Data files not found for stock {stock_code} on date {date_to_test}")

if __name__ == "__main__":
    # 回測資料夾與日期
    period = "5days"
    year = "2024"
    start_date = "2024-10-01"
    end_date = "2024-10-18"

    params = {
        'long_short': False, # True: 多, False: 空
        'dynamic': True, # True: 動態, False: 靜態
        'is_pct': True, # True: 使用百分比進場, False: 使用tick進場
        'pct': 0.01, # 百分比進場%數
        'ticks': 4, # tick進場數
        'profit_ratio': 0.02, # 止盈%數
        'loss_ratio': 0.015, # 止損%數
        'volume_threshold': 10, # 成交量門檻
        'period': 30, # 監測時間
        'last_trade_hour': 12, # 最後可交易時間(小時)
        'last_trade_minute': 0, # 最後可交易時間(分鐘)
        'close_position_hour': 13, # 平倉時間(小時)
        'close_position_miinute': 0, # 平倉時間(分鐘)
        'initial_cash': 200000, # 初始金額
        'commision': 0.001425, # 手續費
        'tax': 0.003 # 稅金
    }

    # 修正目录路径
    list_dir = os.path.join(os.getcwd(), "list", f"股票清單_{period}", year)
    tick_dir = os.path.join(os.getcwd(), "list", f"Ticks_{period}", year)

    print(f"List directory: {os.path.abspath(list_dir)}")
    print(f"Tick directory: {os.path.abspath(tick_dir)}")

    valid_dates = get_valid_trading_dates(start_date, end_date, tick_dir)

    if not valid_dates:
        print(f"No valid trading dates found between {start_date} and {end_date}")
    else:
        print(f"Valid trading dates: {valid_dates}")

    # 加載外內盤計算csv, 建立 StockDataAnalyzer 實例
    output_dir = os.path.join(os.getcwd(), 'in_out_output', f'in_out_{period}', year)
    os.makedirs(output_dir, exist_ok=True)
        
    analyzer = StockDataAnalyzer(tick_dir, output_dir)

    # 只獲取指定日期範圍內的 csv 文件
    csv_files = []
    for date in valid_dates:
        date_dir = os.path.join(tick_dir, date)
        date_output_dir = os.path.join(output_dir, date)  # 每個日期的輸出目錄
        
        print(date_output_dir)
        
        # 如果當前日期的輸出資料夾不存在，才執行轉換流程
        if not os.path.exists(date_output_dir):
            if os.path.exists(date_dir):
                for file in os.listdir(date_dir):
                    if file.endswith('_ticks.csv'):
                        csv_files.append(os.path.join(date_dir, file))

            # 處理找到的文件
            for file_name in csv_files:
                print(f"\n處理檔案: {file_name}")
                result_df = analyzer.analyze_and_export(file_name, date)

                if result_df is not None:
                    print(f"檔案 {file_name} 處理完成")
                    print(f"總記錄數: {len(result_df)}")
        else:
            print(f"{date_output_dir} 已存在，跳過日期 {date} 的處理")

    all_results = []
    try:
        for date_to_test in valid_dates:
            print(f"Processing date: {date_to_test}")
            
            # 修正文件名格式
            list_file = os.path.join(list_dir, f"stocklist_{date_to_test.replace('-', '_')}_{period}.csv")
            print(f"Looking for stock list file: {os.path.abspath(list_file)}")

            if not os.path.exists(list_file):
                print(f"Stock list file not found for date {date_to_test}, skipping.")
                continue

            try:
                stock_list = pd.read_csv(list_file)
                stock_codes = stock_list['Stock Code'].tolist()
                print(f"Found {len(stock_codes)} stocks for date {date_to_test}")

                with Manager() as manager:
                    shared_result = manager.list()
                    lock = manager.Lock()

                    with Pool(processes=int(os.cpu_count() / 4), maxtasksperchild=1) as pool:
                        for code in stock_codes:
                            pool.apply_async(process_stock, (code, date_to_test, shared_result, tick_dir, lock, params))
    
                        pool.close()  # 禁止再添加新的任務
                        pool.join()   # 等待所有進程完成
                            
                    all_results.extend(list(shared_result))
                    print("All processes have completed successfully.")
            
            except Exception as e:
                print(f"Error processing date {date_to_test}: {str(e)}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        if all_results:
            analyze_global_results(all_results, **params)
        else:
            print("No results to analyze. Check if any valid trading dates were processed.")

    print("Backtest completed for all dates.")