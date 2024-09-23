def calculate_mae(file_path):
    total_abs_error = 0
    count = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()  # 移除空白并分割每行
            print(parts)
            if len(parts) >= 3:
                predicted = int(parts[1])
                actual = int(parts[2])
                # 计算绝对误差并累加
                total_abs_error += abs(predicted - actual)
                count += 1

    if count == 0:
        raise ValueError("没有找到预测值和真实值。")

    # 计算MAE
    mae = total_abs_error / count
    return mae


if __name__ == "__main__":
    
    file_path = './pre_gd_cnt.txt'  # 替换为你的文件路径
    try:
        mae = calculate_mae(file_path)
        print(f"MAE: {mae}")
    except ValueError as e:
        print(e)