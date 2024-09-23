from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2


def calculate_average_precision(y_true, y_pred):
    total_precision = 0
    count = 0
    for real, pred in zip(y_true, y_pred):
        if real != 0:  # 避免除以零
            precision = abs(pred - real) / real
            total_precision += precision
            count += 1
    # 计算平均精度
    average_precision = total_precision / count if count else 0
    return average_precision

def read_data_from_file(file_path):
    y_pred = []
    y_true = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                predicted = float(parts[1])
                actual = float(parts[2])
                y_pred.append(predicted)
                y_true.append(actual)

    return y_true, y_pred



if __name__ == "__main__":
    
    file_path = './pre_gd_cnt.txt'  # 替换为你的文件路径
    y_true, y_pred = read_data_from_file(file_path)
    mse, mae, r2 = calculate_metrics(y_true, y_pred)
    cprecision= calculate_average_precision(y_true, y_pred)
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'R^2: {r2}')
    print(f'cprecision: {cprecision}')