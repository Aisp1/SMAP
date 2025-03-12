from typing import List
import tiktoken
import numpy as np
import os
import json
import re
import sys
import time
import torch
from copy import deepcopy
import xml.etree.ElementTree as ET
import openai
from openai import OpenAI
from openai._exceptions import OpenAIError
from .Base import BaseStrategy
from models.Base import BaseModel

from datasets.Dataset import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.MBPPDataset import MBPPDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset

from results.Results import Results
from evaluations.func_evaluate import evaluate_io

mapping = {
    1: "one (01)",
    2: "two (02)",
    3: "three (03)",
    4: "four (04)",
    5: "five (05)",
    6: "six (06)",
    7: "seven (07)",
    8: "eight (08)",
    9: "nine (09)",
}


# KB + Exemplars + Example Planning + Problem Planning + Code Generation + Sample IO testing + Code Improvement
def extract_plan(text):
    pattern = r"(?:##\s*)?\s*(Tutorial|Rationale):\s*((?:.*?\n?)*?)(?=(?:##|[A-Z][a-z]+:|$))"
    matches = re.findall(pattern, text, re.S)
    return {title: content.strip() for title, content in matches}
def remove_think(text):
    return re.split(r'</think>', text, flags=re.DOTALL)[-1].strip()

def validate_xml_format(output):
    pattern = r'^<root>\s*<structures>[\s\S]*?</structures>\s*<code>[\s\S]*?</code>\s*</root>$'
    return bool(re.match(pattern, output.strip(), re.DOTALL))


def getObjectXml(output):
    pattern = r"<root>([\s\S]*?)</root>"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(0)
    return None


def remove_module_tags(input_text):
    modules_section = re.search(r"(?<=# MODULES:\n)(.*?)(?=\n#|\Z)", input_text, re.DOTALL)
    if modules_section:
        return modules_section.group(0)
    lines = input_text.splitlines()
    result = [line for line in lines if line.strip() not in ["```module", "```"]]
    # result = [line for line in lines if line.strip() not in ["```python", "```"]]
    return "\n".join(result)


def remove_algorithm_tags(input_text):
    modules_section = re.search(r"(?<=# Algorithm:\n)(.*?)(?=\n#|\Z)", input_text, re.DOTALL)
    if modules_section:
        return modules_section.group(0)


def get_module(text):
    step2_pattern = r'STEP 2: GENERATE PYTHON CODE\n```python\n(.*?)```'
    match = re.search(step2_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def fix_xml_retrieve(xml_text):
    def replace_tag(match):
        tag = match.group(1)
        if f"<{tag}>" in xml_text:
            return match.group(0)
        return f"<{tag}>"

    planning_indices = [
        i for i in range(len(xml_text))
        if xml_text.startswith("planning>", i) and (i == 0 or xml_text[i - 1] != "/")
    ]

    # 从后往前处理，避免插入新内容影响后续索引
    for planning_index in reversed(planning_indices):
        if planning_index == 0 or xml_text[planning_index - 1] != "<":
            xml_text = xml_text[:planning_index] + "<" + xml_text[planning_index:]
    planning_indices = [i for i in range(len(xml_text)) if xml_text.startswith("<planning>", i)]

    for planning_index in reversed(planning_indices):
        if not xml_text[:planning_index].strip().endswith("</code>"):
            xml_text = xml_text[:planning_index] + "</code>\n" + xml_text[planning_index:]

    if not xml_text.endswith("</root>"):
        xml_text += "</root>"

    xml_text = re.sub(r'(?<!</)(\w+)>', replace_tag, xml_text)

    xml_text = re.sub(r'(<code>.*?)(?=</code>)', r'\1</code>', xml_text, flags=re.DOTALL)
    xml_text = xml_text.replace("</a<lgorithm>", "</algorithm>")
    xml_text = xml_text.replace("</d<escription>", "</description>")
    xml_text = xml_text.replace("</c<ode>", "</code>")
    xml_text = xml_text.replace("</p<lanning>", "</planning>")
    xml_text = xml_text.replace("</p<roblem>", "</problem>")
    xml_text = xml_text.replace("</s<imilarity>", "</similarity>")
    xml_text = xml_text.replace("</r<oot>", "</root>")

    return xml_text


def fix_xml_ver(xml_text: str) -> str:
    confidence_indices = [i for i in range(len(xml_text)) if xml_text.startswith("<confidence>", i)]

    for confidence_index in reversed(confidence_indices):
        if not xml_text[:confidence_index].strip().endswith("</explanation>"):
            xml_text = xml_text[:confidence_index] + "</explanation>\n" + xml_text[confidence_index:]

    root_index = xml_text.find("<root>")
    if root_index != -1:
        root_end_index = root_index + len("<root>")

        if not xml_text[root_end_index:].strip().startswith("<explanation>"):
            xml_text = (
                    xml_text[:root_end_index] + "\n<explanation>" + xml_text[root_end_index:]
            )

    if not xml_text.endswith("</root>"):
        xml_text += "</root>"

    return xml_text
def fix_ver(xml_data):
    root = ET.fromstring(xml_data)

    explanation = root.find('explanation').text.strip()
    confidence = root.find('confidence').text.strip()

    new_root = ET.Element('root')
    new_explanation = ET.SubElement(new_root, 'explanation')
    new_explanation.text = explanation
    new_confidence = ET.SubElement(new_root, 'confidence')
    new_confidence.text = confidence

    return ET.tostring(new_root, encoding='unicode')
class SMAP(BaseStrategy):
    def __init__(
            self,
            k: int = 3,
            t: int = 5,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.t = t

    def xml_to_dict(self, element):
        result = {}
        for child in element:
            if child:
                child_data = self.xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            else:
                result[child.tag] = child.text
        return result

    def parse_xml(self, response: str) -> dict:
        if '```xml' in response:
            response = response.replace('```xml', '')
        if '```python' in response:
            response = response.replace('```python', '')
        if '```' in response:
            response = response.replace('```', '')

        try:
            root = ET.fromstring(response)
        except ET.ParseError as e:
            print(f"XML Parsing Error: {e}. Trying to fix the format.", flush=True)
            try:
                root = ET.fromstring('<root>\n' + response + '\n</root>')
            except:
                root = ET.fromstring('<root>\n' + response)
        return self.xml_to_dict(root)

    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response

        code_pattern = r'```((.|\n)*?)```'
        if "```Python" in response:
            code_pattern = r'```Python((.|\n)*?)```'
        if "```Python3" in response:
            code_pattern = r'```Python3((.|\n)*?)```'
        if "```python" in response:
            code_pattern = r'```python((.|\n)*?)```'
        if "```python3" in response:
            code_pattern = r'```python3((.|\n)*?)```'
        if "```C" in response:
            code_pattern = r'```C((.|\n)*?)```'
        if "```c" in response:
            code_pattern = r'```c((.|\n)*?)```'
        if "```C++" in response:
            code_pattern = r'```C\+\+((.|\n)*?)```'
        if "```c++" in response:
            code_pattern = r'```c\+\+((.|\n)*?)```'
        if "```Java" in response:
            code_pattern = r'```Java((.|\n)*?)```'
        if "```java" in response:
            code_pattern = r'```java((.|\n)*?)```'
        if "```Node" in response:
            code_pattern = r'```Node((.|\n)*?)```'
        if "```node" in response:
            code_pattern = r'```node((.|\n)*?)```'
        if "```Rust" in response:
            code_pattern = r'```Rust((.|\n)*?)```'
        if "```rust" in response:
            code_pattern = r'```rust((.|\n)*?)```'
        if "```PHP" in response:
            code_pattern = r'```PHP((.|\n)*?)```'
        if "```php" in response:
            code_pattern = r'```php((.|\n)*?)```'
        if "```Go" in response:
            code_pattern = r'```Go((.|\n)*?)```'
        if "```go" in response:
            code_pattern = r'```go((.|\n)*?)```'
        if "```Ruby" in response:
            code_pattern = r'```Ruby((.|\n)*?)```'
        if "```ruby" in response:
            code_pattern = r'```ruby((.|\n)*?)```'
        if "```C#" in response:
            code_pattern = r'```C#((.|\n)*?)```'
        if "```c#" in response:
            code_pattern = r'```c#((.|\n)*?)```'
        if "```csharp" in response:
            code_pattern = r'```csharp((.|\n)*?)```'

        code_blocks = re.findall(code_pattern, response, re.DOTALL)

        if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
            code_str = "\n".join(code_blocks[-1])
        elif type(code_blocks[-1]) == str:
            code_str = code_blocks[-1]
        else:
            code_str = response

        return code_str

    @staticmethod
    def trim_text(text: str, trimmed_text: str):
        return text.replace(trimmed_text, '').strip()

    @staticmethod
    def replace_tag(text: str, tag: str):
        if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
            return text
        else:
            return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()

    @staticmethod
    def get_sample_io_str(sample_io: any) -> str:
        if len(sample_io) > 0:
            if type(sample_io[0]) == str:
                return "\n".join(sample_io)
            if type(sample_io[0]) == dict:
                return "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io])
        return sample_io

    def run_single_pass(self, item: dict):
        print("", flush=True)
        distiller_prompt = """
You are an expert in extracting and structuring essential information from complex problem descriptions. Your task is to distill key information into a structured format that comprehensively covers all elements required to solve the problem. Ensure clarity and thoroughness in your extraction process.
Please distill the information following the format below and cease response after the output of the distilled information.



Meta distiller Respond:


Problem Information:

1. Rules:
    # Include all rules, relationships between variables, and constraints described explicitly or implicitly in the problem.

2. Distilled task:
    # Clearly state the main task or goal that needs to be accomplished to solve the problem.
    # Provide a concise explanation of the input, expected output, and any key requirements for implementation.

**Note: The generation ends here. Do not show this message in your answer !**


"""
        input_question_dsitller = [
            {
                "role": "system",
                "content": distiller_prompt
            },
            {
                "role": "user",
                "content": f"{self.data.get_prompt(item)}"
            }
        ]
        problem_info, pr_tok, com_tok = self.gpt_chat(
            input_question_dsitller
        )
        problem_info = self.trim_text(
            problem_info, "**Note: The generation ends here. Do not show this message in your answer !**")

        input_kb_exemplars = [
            {
                "role": "user",
                "content": f"""Your goal is to retrieve tasks that are most similar to the given problem in both rules and task design. Given a problem, explain the core concepts in it and provide other relevant problems.
# Problem:
{problem_info}
# Instruction:
## Algorithms:
Identify the core concepts or algorithms used to solve the problem.

## Example Problems:
Provide {mapping[self.k]} examples of relevant and distinct competitive programming problems that involve these algorithms. For each problem, 
1. describe it
2. generate python code step by step to solve that problem
3. finally generate a planning to solve that problem
4. include similarities in rules and task goals

----------------
Important:
Your response must follow the following xml format-

<root>
<algorithm>
# Identify the core concepts or algorithms used to solve the original problem.
</algorithm>
<problem>
# Provide {mapping[self.k]} examples of relevant and distinct competitive programming problems. Write each problem in the following format.
<description>
# Give a description of the problem.
</description>
<code>
# Let's think step by step to solve this problem in {self.language} programming language.
</code>
<planning>
# Planning to solve this problem.
</planning>
<similarity>
# Explain how this problem is similar to the given problem, focusing on rules and task.
</similarity>
</problem>

# similarly add more problems here...

</root>
""",
            },
        ]

        print("\n\n________________________")
        print("Input for knowledge base and exemplars: ")
        print(input_kb_exemplars[0]['content'], flush=True)

        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            processed_input=input_kb_exemplars
        )
        if self.model.model_params['model'] == "Llama3.1-Instruct-8B":
            response = fix_xml_retrieve(response)
        item['api_calls'] = item.get('api_calls', 0) + 1
        item['api_calls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        # time.sleep(1)
        # Post processing
        response = self.trim_text(
            response, "# Planning to solve this problem.")
        response = self.trim_text(
            response, f"# Let's think step by step to solve this problem in {self.language} programming language.")
        response = self.trim_text(
            response, f"# Identify the core concepts or algorithms used to solve the original problem.")
        response = self.replace_tag(response, 'description')
        response = self.replace_tag(response, 'code')
        response = self.replace_tag(response, 'planning')
        response = self.replace_tag(response, 'similarity')
        response = self.replace_tag(response, 'algorithm')
        print("\n\n________________________")
        print("Response from knowledge base and exemplars: ")
        print(response, flush=True)

        response1 = self.parse_xml(response)
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{response1['algorithm']}"
        if type(self.data) != XCodeDataset:
            sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"
        else:
            sample_io = []
            for input, output in zip(item["sample_inputs"], item["sample_outputs"]):
                sample_io.append({
                    "input": input,
                    "output": [output]
                })
            sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(sample_io)}\n"
        # if type(self.data) != MBPPDataset and type(self.data) != XCodeDataset else ""
        plannings = []
        for example_no, example in enumerate(response1["problem"], start=1):
            example_problem = example["description"]
            example_plan = example["planning"]
            input_for_problem_planning = [

                {
                    "role": "user",
                    "content": f"""Given a competitive programming problem generate a concrete planning to solve the problem.\n# Problem:\n{example_problem}\n# Planning:\n{example_plan}\n{algorithm_prompt}\n## Problem to be solved:\n{problem_info}\n{sample_io_prompt}\n## Tutorial:\n----------------\nImportant: 1、Propose a clever and efficient high-level Tutorial about the above mentioned algorithms for original problem. Let’s think things through one step at a time to come up with a clever tutorial. Do not generate code.
                    """
                }
            ]
            print("\n\n________________________")
            print(input_for_problem_planning[0]['content'], flush=True)

            plan_res, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_problem_planning
            )
            planning = plan_res
            item['api_calls'] += 1
            # time.sleep(1)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n________________________")
            print("Response from our problem planning: ")
            print(planning, flush=True)

            plannings.append((
                planning,
                example_no
            ))

        contra_prompt = f"""
You are a programming specialist who specializes in analyzing plans. Given some candidate plans for a programming problem, you should carefully compare the difference for each two plans in their solving steps.

When you compare , you need to consider the following questions :
1: Do the two plans produce different outputs or handle edge cases differently?
2: Where are the differences in their implementation and logic?
3: Why are the two solutions different?

After contrasting , you should generate a checklist based on these differences between candidate plans. You should carefully consider each discrepancy and the reasons behind it , summarizing them into a few checking instructions in the checklist. This checklist can guide others to re - examine the input question and these candidate plans to eliminate these discrepancies.

Input Format :
The programming problem is
{problem_info}
Plan1 :
{plannings[0][0]}
Plan2 :
{plannings[1][0]}
Plan3 :
{plannings[2][0]}
Output Format :
For Plan1 and Plan2 : {{Difference1}}
For Plan1 and Plan3 : {{Difference2}}
For Plan2 and Plan3 : {{Difference3}}
Checklist : {{Directive1}}, {{Directive2}}, ...
                        """
        input_for_planning_contra = [
            {
                "role": "user",
                "content": contra_prompt
            }
        ]
        contar_res, pr_tok_1, com_tok_1 = self.gpt_chat(
            input_for_planning_contra
        )
        if self.k == 3:
            candidates = f"""
Plan1 :
{plannings[0][0]}
Plan2 :
{plannings[1][0]}
Plan3 :
{plannings[2][0]}
            """
        elif self.k == 5:
            candidates = f"""
Plan1 :
{plannings[0][0]}
Plan2 :
{plannings[1][0]}
Plan3 :
{plannings[2][0]}
Plan4 :
{plannings[3][0]}
Plan5 :
{plannings[4][0]}
"""
        else:
            candidates = f"""
Plan1 :
{plannings[0][0]}
Plan2 :
{plannings[1][0]}
Plan3 :
{plannings[2][0]}
Plan4 :
{plannings[3][0]}
Plan5 :
{plannings[4][0]}
Plan6 :
{plannings[5][0]}
Plan7 :
{plannings[6][0]}
"""
        fix_prompt = f"""
Given a programming problem, multiple inconsistent plans, their differences in solving processes, and a checklist, your task is to analyze the differences between the plans, resolve their discrepancies, and generate a new, unified plan. The new plan should integrate the strengths of the original plans while addressing all inconsistencies.
1.  Review the checklist carefully to resolve conflicts between different plans. The checklist highlights the differences and provides specific directives on how to correct them.
2.  Use the checklist directives to address conflicts and prioritize logic that aligns with the problem's requirements.
3.  Combine the complementary aspects of each plan to create a new, unified solution.

The programming problem is
{problem_info}
The candidate plans and their discrepancy are as follows :
{candidates}

Checklist:
{contar_res}

Analyze the differences between the candidate plans, resolve their discrepancies, and generate a single, unified plan that meets the requirements of the programming problem.
----------------
Important:
Your response must star with ###
                """
        input_for_planning_fix = [
            {
                "role": "user",
                "content": fix_prompt
            }
        ]
        fix_res, pr_tok_1, com_tok_1 = self.gpt_chat(
            input_for_planning_fix
        )
        plannings.append((
            fix_res,
            -1
        ))
        f_plans = []
        for plan_e in plannings:
            plan, no = plan_e
            input_for_planning_verification = [
                {
                    "role": "user",
                    "content": f"""
Evaluate the structural characteristics of the given Plan based on the following criteria:
1. Logical Steps:
- Verify whether the plan’s logic aligns with the problem statement.
- Are the steps logically connected, making it easy to follow the overall thought process?

2. Modularity:
- Is the Plan divided into distinct phases or modules (e.g., input handling, computation, and output generation)?

3. Performance Considerations:
- Does the Plan discuss or imply the algorithm's efficiency (e.g., time and space complexity)?
- Are the steps justified in terms of their computational feasibility?
# Problem:
{problem_info}
# Plan:
{plan}
Important: Your response must follow the following xml format-
<root>
<explanation> Provide a detailed explanation of whether the Plan is structured. Discuss strengths, weaknesses, and specific areas for improvement in its structure. </explanation>
<confidence> Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100. </confidence>
</root>
```
                        """
                }
            ]

            print("Input for planning verification: ")
            print(input_for_planning_verification[0]['content'], flush=True)

            verification_res, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_planning_verification
            )
            if self.model.model_params['model'] == "Llama3.1-Instruct-8B":
                verification_res = fix_xml_ver(verification_res)
                print(verification_res)
            item['api_calls'] += 1
            # time.sleep(1)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            verification_res = self.replace_tag(
                verification_res, 'explanation')
            verification_res = self.replace_tag(verification_res, 'confidence')
            try:
                verification_res = self.parse_xml(verification_res)
            except:
                verification_res = fix_ver(verification_res)
                verification_res = self.parse_xml(verification_res)
            verification_res['confidence'] = int(
                str(verification_res['confidence']).strip())

            print("Response from planning verification: ")
            print(verification_res, flush=True)
            f_plans.append((
                plan,
                verification_res['confidence']
            ))
        f_plans.sort(key=lambda x: x[1], reverse=True)
        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: \nStrictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
            input_require = "4. Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""
            input_require = ""

        for planning_with_ex in f_plans:  # 遍历根据置信度排序的plan。
            planning, confidence = planning_with_ex
            file_path = './SMAP/src/promptings/module_example'
            with open(file_path, 'r', encoding='utf-8') as infile:
                module_ex = infile.read()

            module_prompt = f"""
*Instruction*
Develop a well-structured {self.language} solution for the provided plan that obeys the constraints and passes the example test cases. Ensure modularity and considering potential edge cases and failures. Start by outlining the required code modules, including function headers and signatures.

In simpler terms, Enhance the provided plan to include all intermediate steps and modular testing. Break it down into smaller parts (modules) with clear function names and input/output specifications.

### Example 1

{module_ex}
-----------------
### Example 2

### PLAN:
{planning}
### RESPONSE:

"""
            input_for_module_code_generation = [
                {
                    "role": "user",
                    "content": f"{module_prompt}"
                }
            ]

            module_res, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_module_code_generation
            )
            item['api_calls'] += 1
            # time.sleep(1)
            module_code = self.parse_code(module_res)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            # 以下为code agent
            if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(
                    self.data) == XCodeDataset:
                file_path = './SMAP/src/promptings/final_code_example'
                with open(file_path, 'r', encoding='utf-8') as infile:
                    c_example = infile.read()
            else:
                file_path = './SMAP/src/promptings/final_code_example_n'
                with open(file_path, 'r', encoding='utf-8') as infile:
                    c_example = infile.read()

            final_code = f"""
Given a competitive programming problem generate {self.language} code to solve the problem.\n## Problem information:\n{self.data.get_prompt(item)}\n{algorithm_prompt}\n## Planning:\n{planning}\n##Relevant Functions:\n{module_code}\n{sample_io_prompt}\n## Let's think step by step.
## Requirements:
1. Your response must contain only the {self.language} code to solve this problem. 
2. Analyze the structural relationships between modules (functions/classes), including:
   - Their hierarchies (low-level utilities, mid-level logic, top-level integration).
   - Call relationships (who calls whom).
   - Division of responsibilities (what each module handles).
3. Reduce the time complexity of the code as much as possible, leveraging additional space if necessary to achieve faster execution.
{input_require}

Important:
Generate the output strictly in the XML format of the Example:
### Exemplar :
{c_example}

### Response format:
<root>
<structures>
[Analyze the structural relationships between modules.]
</structures>
<code>
[Insert the complete solution code here, with no extra symbols or formatting.]
</code>
</root>
"""
            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": final_code
                }
            ]

            print("\n\n________________________")
            print("Input for final code generation: ")
            print(input_for_final_code_generation[0]['content'], flush=True)
            item['api_calls'] += 1
            for attempt in range(0, 3):
                code, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_final_code_generation
                )
                item['api_calls'] += 1
                if validate_xml_format(code):
                    break
            # time.sleep(1)
            is_xml = validate_xml_format(code)
            if self.model.model_params['model'] == "DeepSeek-R1-Distill-Qwen-7B" and not is_xml:
                code = self.parse_code(code)
            else:
                if not is_xml:
                    code = getObjectXml(code)
                result = self.replace_tag(code, 'code')
                result = self.replace_tag(result, 'structures')
                result = self.parse_xml(result)
                code = self.parse_code(result["code"])
            pr_tok += pr_tok_1
            com_tok += com_tok_1
            print("\n\n________________________")
            print("Response from final code generation: ")
            print(code, flush=True)
            # code = self.parse_code(code_res)
            response = f"## Planning: {planning}\n## Code:\n```\n{code}\n```"
            passed = False

            for i in range(1, self.t + 1):
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    code,
                    self.language
                )

                if passed:
                    break
                # 以下为debugging agent
                print(f"Input for improving code generation: {i}")
                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{response}\n## Test Report:\n{test_log}\n## Modified Planning:\n## Let's think step by step to modify {self.language} Code for solving this problem.\n## Reducing code time complexity as much as possible, leveraging additional space if necessary to achieve faster execution. \n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem."
                    }
                ]

                print("\n\n________________________")
                print("Input for improving code generation: ")
                print(input_for_improving_code[0]['content'], flush=True)

                response, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_improving_code
                )
                if self.model.model_params['model'] == "DeepSeek-R1-Distill-Qwen-7B":
                    response = remove_think(response)
                item['api_calls'] += 1
                # time.sleep(1)

                code = self.parse_code(response)
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                print("\n\n________________________")
                print("Response from improving code generation: ")
                print(response, flush=True)

            # got a code that passed all sample test cases
            if passed:
                break
        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok