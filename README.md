# Communication-Efficient and Personalized Federated Foundation Model Fine-Tuning via Tri-Matrix Adaptation


## Abstract

In federated learning, fine-tuning pre-trained foundation models poses significant challenges, particularly regarding high communication cost and suboptimal model performance due to data heterogeneity between the clients. To address these issues, this paper introduces communication-efficient federated low-rank adaptation (CE-LoRA), a method that employs a tri-factorization low-rank adaptation approach with personalized model parameter aggregation. We first present a novel LoRA parameter factorization by introducing a small-size dense matrix, which can significantly reduce the communication cost and achieve comparable empirical performance than transferring the low-rank parameter matrix used by existing methods. Without violating data privacy, the server considers the client similarity in both training dataset and model parameter space, and learns personalized weights for model aggregation. Our experiments on various large language model (LLM) and vision-language model (VLM) fine-tuning tasks demonstrate that CE-LoRA not only significantly reduces communication overhead but also improves performance under not independently and identically distributed data conditions. In addition, CE-LoRA improves data privacy protection, effectively mitigating gradient-based data reconstruction attacks. 

## Folder Structure
```grapha  
CE-LoRA/
│── run/                # Training entry scripts (trainers, server, client)
│── trainers/           # Server and client handlers
│── utils/              # Utilities and registry
│── configs/            # Example config files
│── requirements.txt    # Python dependencies
```  

This repo should be cloned into FedETuning:

```bash  
mkdir workspace  
cd workspace  
mkdir data  
mkdir code  
mkdir pretrain  
```  

## Usage

### Setup
```bash
pip install -r requirements
```

### Run

1. **Prepare pre-trained models**  
   Place your pre-trained checkpoints into the `pretrain/` directory.   

2. **Prepare datasets**  
   - Divide the datasets using the provided `tools/`.  
   - Place the processed datasets into the `data/` directory.  


3. **Run CE-LoRA**  
   Example: fine-tune **SST-2** with 10 clients using LoRA (tri-matrix factorization), ports starting from `10001`, and GPUs `0,1,2`:  

   ```bash
   python fed_seed_run.py /data/liyongle/CE-LoRA fedavg sst-2 lora 10001 0,1,2




