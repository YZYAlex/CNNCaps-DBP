from Bio import SeqIO
import pandas as pd


# 定义一个函数来计算单个序列中氨基酸的频率
def calculate_sequence_amino_acid_frequency(sequence):
    amino_acid_freq = {}
    for amino_acid in sequence:
        if amino_acid in amino_acid_freq:
            amino_acid_freq[amino_acid] += 1
        else:
            amino_acid_freq[amino_acid] = 1
    total_amino_acids = len(sequence)
    for amino_acid in amino_acid_freq:
        amino_acid_freq[amino_acid] /= total_amino_acids
    return amino_acid_freq


# 定义一个函数来处理FASTA文件并计算所有序列的氨基酸频率
def process_fasta_file(fasta_file):
    all_sequences_freq = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_freq = calculate_sequence_amino_acid_frequency(record.seq)
        all_sequences_freq.append(seq_freq)
    return all_sequences_freq


# 指定FASTA文件路径
fasta_file = 'tst.fasta'

# 处理FASTA文件并计算所有序列的氨基酸频率
all_sequences_freq = process_fasta_file(fasta_file)

# 创建一个DataFrame，其中包含所有序列的氨基酸频率
# 首先，确定所有可能的氨基酸列
all_amino_acids = set()
for seq_freq in all_sequences_freq:
    all_amino_acids.update(seq_freq.keys())

# 确保列的顺序一致
all_amino_acids = sorted(all_amino_acids)

# 创建DataFrame
df = pd.DataFrame(columns=all_amino_acids)

# 填充DataFrame
for seq_freq in all_sequences_freq:
    # 将单个序列的频率转换为 DataFrame 的一行
    row = pd.DataFrame([seq_freq])
    # 确保列顺序一致
    row = row.reindex(columns=all_amino_acids, fill_value=0)
    # 添加到总 DataFrame 中
    df = pd.concat([df, row], ignore_index=True)


# 保存到CSV文件
csv_file = 'test_AAC.csv'  # 你可以自定义文件名
df.to_csv(csv_file, index=False)

print(f"Amino acid frequencies for each sequence have been saved to {csv_file}")
