import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from Bio import SeqIO
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("esm2")
model = AutoModel.from_pretrained("esm2")
model.to(device)

records = list(SeqIO.parse("tr.fasta", "fasta"))
sequences = [str(record.seq) for record in records]
print(f"Loaded {len(sequences)} sequences")

max_length = 500
batch_size = 16
sequence_batches = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]

all_embeddings = []
with torch.no_grad():
    for batch in tqdm(sequence_batches, desc="Encoding sequences"):
        encoded_inputs = tokenizer(batch, padding="max_length", truncation=True,
                                   max_length=max_length, return_tensors="pt")
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        outputs = model(**encoded_inputs)
        all_embeddings.append(outputs.last_hidden_state.cpu())

# 合并并保存
all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
np.save("train_esm2-650m.npy", all_embeddings)
print(all_embeddings.shape)
print("All embeddings saved to esm_train.npy")
