import pandas as pd
from openai import OpenAI
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--st', type=int, default=0)
parser.add_argument('--ed', type=int, default=10)
args = parser.parse_args()
# args = get_args()


# 读取 CSV 文件
file_path = "/home/cz/MolTC-main/data/ddi_data/drug2.csv"
df = pd.read_csv(file_path)

df = df.iloc[args.st:args.ed]

# df = df.head()  # 只读取前五行数据

# 初始化 OpenAI 客户端
client = OpenAI(api_key="sk-b706e587730549d6ade178aba4c278ac", base_url="https://api.deepseek.com")

# 创建一个列表存储 API 结果
results = []

# 遍历 CSV 文件中的数据
for index, row in df.iterrows():
    # 第一个 SMILES 及其 ID
    molecule_id_1 = row[0]    # 第一个分子的编号
    smiles_1 = row[2]         # 第一个 SMILES 字符串
    
    # 第二个 SMILES 及其 ID
    molecule_id_2 = row[1]    # 第二个分子的编号
    smiles_2 = row[3]         # 第二个 SMILES 字符串

    # 调用 API 处理第一个分子
    response_1 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a professional chemist. Extract information from a SMILES expression and respond in a single, concise sentence following the format: This molecule contains the substructure ..., so it has the property .... List up to three substructures and their corresponding properties."},
            {"role": "user", "content": smiles_1},
        ],
        stream=False
    )
    molecule_info_1 = response_1.choices[0].message.content  # 获取第一个分子的信息

    # 调用 API 处理第二个分子
    response_2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a professional chemist. Extract information from a SMILES expression and respond in a single, concise sentence following the format: This molecule contains the substructure ..., so it has the property .... List up to three substructures and their corresponding properties."},
            {"role": "user", "content": smiles_2},
        ],
        stream=False
    )
    molecule_info_2 = response_2.choices[0].message.content  # 获取第二个分子的信息

    # 将结果存储到列表中
    results.append([molecule_id_1, smiles_1, molecule_info_1])
    results.append([molecule_id_2, smiles_2, molecule_info_2])

    print(f"Processed {molecule_id_1} and {molecule_id_2}")  # 显示进度

# 将结果保存为 CSV 文件
output_file = f"/home/cz/MolTC-main/data/pretrain_data/drug2_{args.st}_{args.ed}.csv"
output_df = pd.DataFrame(results, columns=["Molecule_ID", "SMILES", "Properties"])
output_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Results saved to {output_file}")
