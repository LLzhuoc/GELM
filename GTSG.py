import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import pandas as pd
import os
import threading

def get_morgan_fingerprint(smiles, radius=2, n_bits=1024):
    """计算SMILES的Morgan指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return fp

def tanimoto_similarity(fp1, fp2):
    """计算Tanimoto相似度"""
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def build_molecule_graph(data, threshold=0.3):
    """从data构建分子图"""
    G = nx.Graph()
    smiles_list = []
    descriptions = []
    ids = []
    fingerprints = {}

    # 解析data，提取分子ID、SMILES和描述
    for row in data:
        mol_id = row[0]  # 分子ID
        smiles = row[1]  # SMILES字符串
        description = row[2]  # 分子描述
        fingerprint = get_morgan_fingerprint(smiles)

        if fingerprint is not None:  # 确保SMILES有效
            smiles_list.append(smiles)
            descriptions.append(description)
            ids.append(mol_id)
            fingerprints[smiles] = fingerprint
            G.add_node(smiles, description=description, mol_id=mol_id)

    # 建立边（基于Tanimoto相似度）
    for i, smi1 in enumerate(smiles_list):
        for j, smi2 in enumerate(smiles_list):
            if i < j:
                fp1, fp2 = fingerprints[smi1], fingerprints[smi2]
                similarity = tanimoto_similarity(fp1, fp2)
                if similarity >= threshold:
                    G.add_edge(smi1, smi2, weight=similarity)

    return G

def traverse_molecule_graph(G):
    """遍历分子图，返回SMILES路径、分子ID及对应描述"""
    all_paths = []
    all_description_paths = []
    all_id_paths = []
    visited = set()

    while len(visited) < len(G.nodes):
        path = []
        description_path = []
        id_path = []

        # 选择度最小的未访问节点作为起点
        unvisited_nodes = [n for n in G.nodes if n not in visited]
        if not unvisited_nodes:
            break
        current_node = min(unvisited_nodes, key=lambda n: G.degree[n])

        while current_node not in visited:
            path.append(current_node)
            description_path.append(G.nodes[current_node]['description'])
            id_path.append(G.nodes[current_node]['mol_id'])
            visited.add(current_node)

            # 选择未访问邻居中相似性最高的
            neighbors = [(nbr, G[current_node][nbr]['weight']) for nbr in G.neighbors(current_node) if nbr not in visited]

            if neighbors:
                current_node = max(neighbors, key=lambda x: x[1])[0]
            else:
                break

        all_paths.append(path)
        all_description_paths.append(description_path)
        all_id_paths.append(id_path)

    return all_paths, all_description_paths, all_id_paths

def process_file(file_name, gpt_dir, output_dir, threshold):
    file_path = os.path.join(gpt_dir, file_name)
    data = pd.read_csv(file_path).to_numpy()

    # 构建分子图
    G = build_molecule_graph(data, threshold)

    # 遍历分子图
    traversal_paths, description_paths, id_paths = traverse_molecule_graph(G)

    # 保存处理后的数据
    output_file_path = os.path.join(output_dir, file_name)
    processed_data = []
    for ids, smiles, descriptions in zip(id_paths, traversal_paths, description_paths):
        for mol_id, smile, description in zip(ids, smiles, descriptions):
            processed_data.append([mol_id, smile, description])

    processed_df = pd.DataFrame(processed_data, columns=["Molecule_ID", "SMILES", "Description"])
    processed_df.to_csv(output_file_path, index=False)

    print("Processed data saved to:", output_file_path)

# 设置路径
gpt_dir = "/home/cz/MolTC-main/data/pretrain_data/gpt"
output_dir = "/home/cz/MolTC-main/data/pretrain_data/gpt/after"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置Tanimoto相似度阈值
threshold = 0.3

# 获取所有CSV文件
csv_files = [f for f in os.listdir(gpt_dir) if f.endswith(".csv")]

# 多线程处理
threads = []
for file_name in csv_files:
    thread = threading.Thread(target=process_file, args=(file_name, gpt_dir, output_dir, threshold))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()

print("All files processed.")