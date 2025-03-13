 

# SMAP: Self-enhanced Multi-Agent Code Generation for Competitive Problem Solving


## Quick Start

1. Create a new conda or python virtual environment and install neccessary libraries in the requirements.
```
pip install -r requirements.txt
```

2. Set up the .env file by seeing the example.

3. Run the following command to see the options of running this projects
```
python src/main.py --help
```

4. Finally run this project. An example is given below:
```
python src/main.py --model ChatGPT --dataset HumanEval --strategy Direct
```

5. To run this projects with competitive datasets you need to setup the [ExecEval](https://github.com/ntunlp/ExecEval) for docker execution. Please visit this [link](https://github.com/ntunlp/ExecEval) to setup a docker container and run it using 5000 port. Change the line 50 of the file `src\evaluations\api_comm.py` for different setup. 

## Open Source Models
To run on open source models, we recommend to set up following the [VLLM framework](https://github.com/vllm-project/vllm) which can speed the generation significantly.

