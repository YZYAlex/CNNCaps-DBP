# CNNCaps-DBP
The data and a user-friendly standalone program of CNNCaps-DBP
# Pre-requisite:
- Python3.10.0, pytorch(2.2.0 or higher), numpy(1.24.3 or higher), pandas
- esmc-600m(https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12)
# Installation:
- 1.Download the source code in this repository.
- 2.The source code includes the model and the dataset.
- 3.Download the esmc-600m from huggingface.
# Running
- [code]: The path of folder that contains all CNNCaps-DBP codes.
- [fasta_path]: The path of query protein sequences in fasta format.
- [result_dir]: The path of folder of model parameters.
- [save_dir]: The path of folder of experimental results of attention mechanism visualization.
- [device]: cuda or cpu
# Functionality description
- The protein sequence in fasta format will be first fed to ESMC model to generate a embedding feature representation; then, the embedding features will be fed into the networks consisting of attention augmented convolution, capsule neural network and multilayer perceptron to obatain the predcition results.
# Note
- The feature representation of ESMC will be generated named dataset_name.npy.
- The prediction results will be displayed in the terminal after the training is finished.
- The parameters of the trained model will be stored in a folder named checkpoints.
- If you have any question, please email to 20233005320@hainanu.edu.cn freely.
- All the best to you!
# Domain-adaptive pretraining
- [fasta_path]: The path of the datasets.
- [device]: 'cuda:0' is recommended.
- [batch_size]: This should depend on the available memory. Our memory size is 40GB, and a batch size of 128 is appropriate.
- [epoch]: 40 is recommended.
