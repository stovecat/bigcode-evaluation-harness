import inspect
from pprint import pprint

from . import (apps, codexglue_code_to_text, codexglue_text_to_text, conala,
               concode, ds1000, gsm, humaneval, humanevalpack, instruct_humaneval, mbpp, multiple, parity, python_bugs, quixbugs)

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    **multiple.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **ds1000.create_all_tasks(),
    "humaneval": humaneval.HumanEval,
    **humanevalpack.create_all_tasks(),
    "mbpp": mbpp.MBPP,
    "parity": parity.Parity,
    "python_bugs": python_bugs.PythonBugs,
    "quixbugs": quixbugs.QuixBugs,
    **gsm.create_all_tasks(),
    **instruct_humaneval.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["load_data_path"] = args.load_data_path
        if args.multiple_benchmark is not None:
            kwargs["multiple_benchmark"] = args.multiple_benchmark    
        kwargs["custom_test"] = args.custom_test    
        
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
