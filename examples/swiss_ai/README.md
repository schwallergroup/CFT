# Swiss AI Example

## Setup

### 1. Download a MACE model

```bash
mkdir models
cd models
wget https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model
```

### 2. Download the dataset

```bash
cd examples/swiss_ai/data
wget https://muellergroup.jhu.edu/qcd/data_json.tar.gz
tar -xzvf data_json.tar.gz
rm data_json.tar.gz
```
