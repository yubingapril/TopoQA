# TopoQA
## Installation
1. Download this repository
```bash
git clone https://github.com/yubingapril/TopoQA.git
```
2. Set up conda environment locally
```bash
cd TopoQA
conda env create --name TopoQA -f environment.yml
```
3. Activate conda environment
```bash
conda activate TopoQA
```
## Usage
Here is the inference.py script parameters' introduction
```python
python inference_model.py
-c --complex_folder  Protein complex complex_folder
-w --work_dir  Working directory to save all intermedia files and folders, it will be created if it is not exit
-r --result_folder  Result folder to save two ranking results, it will be created if it is not exit
```
