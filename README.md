 

# SMAP: Self-enhanced Multi-Agent Code Generation for Competitive Problem Solving
      



## Overview
![SMAP Overview](./images/overview.png)
We propose a self-enhanced multi-agent promoting (SMAP) approach, which aims to make LLM agents think like senior programmers for competitive problem-solving. Our framework enhances the unsophisticated programming cycle with four LLM agents, and it includes rule-based retriever, consensus planning, structured programming and plan-derived debugging. SMAP first provides insights into problem description through rule-based retriever. Then it refines the plans derived from retrieved examples by reflecting on differences in perspectives. Sequentially, it enhances the relationship among modules with hierarchically modular coding. Finally, plan-derived debugging refers to the sample I/O from the problem description for fixing bugs in generated code, which requires no extra test case generation. In this section, we first introduce the retriever agent, and then illustrate the planning and the programming in detail. 

## Quick Start
1. Clone our project
```
git clone https://github.com/Aisp1/SMAP && cd SMAP
```

2. Create a new conda or python virtual environment and run the following command
```
pip install -r requirements.txt
```

3. Set up the .env file by seeing the example.

4. Run the following command to see the options of running this projects
```
python src/main.py --help
```

5. Finally run this project. An example is given below:
```
python src/main.py --model ChatGPT --dataset HumanEval --strategy Direct
```

6. To run this projects with competitive datasets you need to setup the [ExecEval](https://github.com/ntunlp/ExecEval) for docker execution. Please visit this [link](https://github.com/ntunlp/ExecEval) to setup a docker container and run it using 5000 port. Change the line 50 of the file `src\evaluations\api_comm.py` for different setup. 

## Open Source Models
To run on open source models, we recommend to set up following the [VLLM framework](https://github.com/vllm-project/vllm) which can speed the generation significantly.

