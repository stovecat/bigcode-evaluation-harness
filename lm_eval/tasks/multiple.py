"""MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation
https://arxiv.org/abs/2107.03374

MultiPL-E is a dataset for evaluating large language models for code generation that supports 18 programming languages.
It takes the OpenAI "HumanEval" and the MBPP Python benchmarks and uses little compilers to translate them to other languages.

Homepage: https://nuprl.github.io/MultiPL-E/
"""

import json
import os
import re
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from time import time

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from lm_eval.base import Task
from lm_eval.tasks.custom_metrics.multiple_metrics.evaluation import \
    evaluate_problem
from lm_eval.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import \
    for_file

from data_utils import *

_CITATION = """
@article{cassano2022scalable,
  title={A Scalable and Extensible Approach to Benchmarking NL2Code for 18 Programming Languages},
  author={Cassano, Federico and Gouwar, John and Nguyen, Daniel and Nguyen, Sydney and Phipps-Costin, Luna and Pinckney, Donald and Yee, Ming Ho and Zi, Yangtian and Anderson, Carolyn Jane and Feldman, Molly Q and others},
  journal={arXiv preprint arXiv:2208.08227},
  year={2022}
}
"""

LANGUAGES = [
    "py",
    "sh",
    "cpp",
    "cs",
    "d",
    "go",
    "java",
    "js",
    "jl",
    "lua",
    "pl",
    "php",
    "r",
    "rkt",
    "rb",
    "rs",
    "scala",
    "swift",
    "ts",
]
BENCHMARKS = ["humaneval", "mbpp"]


def create_all_tasks(custom_test=None):
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {f"multiple-{multiple_benchmark}-{language}": create_task(language, multiple_benchmark, custom_test) for multiple_benchmark in BENCHMARKS for language in LANGUAGES}


def create_task(language, multiple_benchmark, custom_test=None):
    class MultiPLE(GeneralMultiPLE):
        def __init__(self, multiple_benchmark, custom_test):
            super().__init__(language, multiple_benchmark, custom_test)

    return MultiPLE


def get_alternable_dataset(immutable_dataset):
    import copy
    alterable_dataset = {}
    for key in immutable_dataset.keys():
        alterable_dataset[key] = []
        for d in immutable_dataset[key]:
            tmp_dict = {}
            for k,v in d.items():
                tmp_dict[k] = v
            alterable_dataset[key].append(tmp_dict)
    return alterable_dataset


def python_name_to_java_name(dataset, custom_tcs):
    java_name = [dataset["test"][i]['tests'].split("    assert(")[1].split("(")[0] for i in range(len(dataset["test"]))]

    python_name = []
    for i in range(len(dataset["test"])):
        flag = True
        _chunk = "    assert("
        for suffix in ["Arrays.equals(", "!", ""]:
            chunk = _chunk + suffix
            splitted = custom_tcs[i].split(chunk)
            if len(splitted) > 1:
                python_name.append(splitted[1].split("(")[0])
                flag = False
                break
        if flag:
            python_name.append("THISISADUMMY")
    assert len(java_name) == len(python_name)
    return [(pn, jn) for pn, jn in zip(python_name, java_name)]


class GeneralMultiPLE(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "nuprl/MultiPL-E"
    DATASET_NAME = None
    DATASET_REVISION = "d23b094346c5dbda1080a74bb2a24c18adbf7409"

    def __init__(self, language, multiple_benchmark="humaneval", custom_test=None):
        assert multiple_benchmark in ["humaneval", "mbpp"]
        self.language = language
        self.DATASET_NAME = f"{multiple_benchmark}-{language}"
        # we need the dataset to get stop words for each language
        self.dataset = load_dataset(
            GeneralMultiPLE.DATASET_PATH,
            self.DATASET_NAME,
            revision=self.DATASET_REVISION)
        stop_words = self.dataset["test"][0]["stop_tokens"]
        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )
        if custom_test is not None:        
            self.dataset = get_alternable_dataset(self.dataset)
#             print(custom_test)
#             original_tcs = [v["tests"] for v in self.dataset["test"]]
#             dump_pkl(original_tcs, "/".join(custom_test.split("/")[:-1])+"/original_tcs.pkl")
#             print("/".join(custom_test.split("/")[:-1])+"/original_tcs.pkl")
#             assert 1 == 2
            custom_tcs = load_pkl(custom_test)
            p_to_j = python_name_to_java_name(self.dataset, custom_tcs)
            assert len(custom_tcs) == len(self.dataset["test"])
            for i in range(len(self.dataset["test"])):
                custom_tcs[i] = custom_tcs[i].replace(p_to_j[i][0],p_to_j[i][1])
                self.dataset["test"][i]["tests"] = custom_tcs[i]
                assert self.dataset["test"][i]["tests"] == custom_tcs[i]
#             print(f'{self.dataset["test"][10]["tests"]=}')
#             assert 1 == 2

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"].strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
#         print(doc["tests"])
#         assert 1 == 2
        return doc["tests"]

    @staticmethod
    def remove_last_block(string, stop_words):
        # Remove the last block of the code containing stop_words for HumanEval
        string_list = re.split("(%s)" % "|".join(stop_words), string)
        # last string should be ""
        return "".join(string_list[:-2])

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        completion = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(completion, self.stop_words)

    def process_results(self, generations, references, metric_output_path):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        # get prompts and problem names
        prompts_names = [
            {"prompt": doc["prompt"], "name": doc["name"]}
            for i, doc in enumerate(self.get_dataset())
            if i < len(generations)
        ]
        # a common temp dir for all the problems
        temp_dir = tempfile.gettempdir()
        list_files = []
        for (prompt_name, generation, reference) in zip(
            prompts_names, generations, references
        ):
            problem = {
                "name": prompt_name["name"],
                "language": self.language,
                "prompt": prompt_name["prompt"],
                "completions": generation,
                "tests": reference,
            }
            # each problem is save in a json file
            temp_file_name = os.path.join(temp_dir, f"{prompt_name['name']}.json")
            list_files.append(temp_file_name)
            with open(temp_file_name, "wt") as f:
                json.dump(problem, f)
        print(
            f"Saved {len(list_files)} problems in {temp_dir} for evaluation, each problem has {len(generations[0])} completions"
        )

        # execute the problems to evaluate them
        max_workers = cpu_count() - 1 if cpu_count() > 1 else 1
        for file in tqdm(list_files):
            evaluate_problem(temp_dir, file, max_workers)

        # compute pass@k scores
        result_array = np.array(
            [for_file(p) for p in Path(temp_dir).glob("*.results.json")]
        )
        result = result_array.mean(axis=0)
        name = (
            temp_dir.split("/")[-1]
            if temp_dir.split("/")[-1] != ""
            else temp_dir.split("/")[-2]
        )
        results = {
            f"pass@{k}": v
            for k, v in zip([1, 10, 100], result)
            if k <= len(generations[0])
        }
        
        raw_results = [raw_open_file(p) for p in Path(temp_dir).glob("*.results.json")]
        
        dump_pkl(raw_results, metric_output_path)
        
        return results
