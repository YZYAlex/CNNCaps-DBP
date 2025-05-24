import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput

def main(client: ESMCInferenceClient, seq):
    # ================================================================
    # Example usage: one single protein
    # ================================================================
    protein = ESMProtein(seq)  # 初始化ESMC蛋白序列对象

    # Use logits endpoint. Using bf16 for inference optimization
    protein_tensor = client.encode(protein)  # 将序列转化为索引
    output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    assert isinstance(
        output, LogitsOutput
    ), f"LogitsOutput was expected but got {output}"
    assert output.logits is not None and output.logits.sequence is not None
    assert output.embeddings is not None and output.embeddings is not None
    print(
        f"Client returned logits with shape: {output.logits.sequence.shape} and embeddings with shape: {output.embeddings.shape}"
    )
    return output.embeddings


if __name__ == "__main__":
    model = ESMC.from_pretrained("esmc_600m")
    from Bio import SeqIO

    input = "train_1075.fasta"
    # 转为字典（ID为键，序列为值）
    seq_dict = {record.id: str(record.seq) for record in SeqIO.parse(input, "fasta")}
    sequences = []
    for value in seq_dict.values():
        max_length = 500
        if len(value) < max_length:
            temp = value + (max_length - len(value)) * "<pad>"
            sequences.append(temp)
        else:
            temp = value[:max_length]
            sequences.append(temp)
    embedding = []

    count = 0

    for i in sequences:
        te = main(model, i).squeeze().cpu().detach().numpy()
        embedding.append(te)
        count += 1
        print(count)
    embedding = np.array(embedding)
    np.save("train_1075_ESMc", embedding)
