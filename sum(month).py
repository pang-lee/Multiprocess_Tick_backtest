import os
from datetime import datetime
from collections import defaultdict

def parse_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    total_profit = None
    trades = []
    
    for line in content.split('\n'):
        if line.startswith('今天的總盈虧為:'):
            total_profit = float(line.split(':')[1].strip())
        elif line.startswith('股票代號:'):
            parts = line.split(',')
            stock_code = parts[0].split(':')[1].strip()
            profit = float(parts[1].split(':')[1].strip())
            trades.append((stock_code, profit))
    
    return total_profit, trades

def analyze_monthly_data(root_dir):
    monthly_data = defaultdict(lambda: {'total_profit': 0, 'profitable_trades': 0, 'total_trades': 0, 'profits': 0, 'losses': 0})
    
    for date_folder in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, date_folder)):
            continue
        
        date = datetime.strptime(date_folder, "%Y-%m-%d")
        month_key = f"{date.year}-{date.month:02d}"
        
        for file in os.listdir(os.path.join(root_dir, date_folder)):
            if file.startswith("Short_static_") and file.endswith(".txt"):
                file_path = os.path.join(root_dir, date_folder, file)
                total_profit, trades = parse_file(file_path)
                
                if total_profit is not None:
                    monthly_data[month_key]['total_profit'] += total_profit
                    monthly_data[month_key]['total_trades'] += len(trades)
                    
                    for _, profit in trades:
                        if profit > 0:
                            monthly_data[month_key]['profitable_trades'] += 1
                            monthly_data[month_key]['profits'] += profit
                        else:
                            monthly_data[month_key]['losses'] -= profit  # losses are negative, so we subtract
    
    return monthly_data

def print_monthly_summary(monthly_data):
    for month, data in sorted(monthly_data.items()):
        print(f"月份: {month}")
        print(f"1. 月總盈虧: {data['total_profit']:.2f}")
        
        if data['losses'] != 0:
            profit_loss_ratio = data['profits'] / data['losses']
            print(f"2. 月損益比: {profit_loss_ratio:.2f}")
        else:
            print("2. 月損益比: 无法计算（没有亏损）")
        
        if data['total_trades'] > 0:
            win_rate = (data['profitable_trades'] / data['total_trades']) * 100
            print(f"3. 勝率: {win_rate:.2f}% ({data['profitable_trades']}/{data['total_trades']})")
        else:
            print("3. 勝率: 无法计算（没有交易）")
        
        print()

def main():
    root_dir = r"D:\backtest\result"
    monthly_data = analyze_monthly_data(root_dir)
    print_monthly_summary(monthly_data)

if __name__ == "__main__":
    main()