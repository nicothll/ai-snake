# AI-Snake

## Teach AI to play Snake with Python

### Installation

 - Run this script on your terminal to create a new conda environment

```bash
 conda env create -f envs/conda.yaml
```
- Activate your environment with:

```bash
conda activate ai_snake
```

- Then install pytorch with the following command according to your OS:
    - Win
    ```bash
    conda install pytorch torchvision cpuonly -c pytorch
    ```

    - MacOS
    ```bash
    conda install pytorch torchvision -c pytorch
    ```

    - Linux
    ```bash
    conda install pytorch torchvision cpuonly -c pytorch
    ```

- And last step install pygame with pip:

```bash
pip install pygame
```
### Run script

To run the AI, run the command:
```bash
python agent.py
```
