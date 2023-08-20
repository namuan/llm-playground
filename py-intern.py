#!/usr/bin/env python3
"""
A simple script

Usage:
./py-intern.py -h

./py-intern.py -q "I want to develop a quiz application in React"
"""
import logging
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from box import Box
from langchain.llms import Ollama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


def setup_logging(verbosity):
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
        ],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    logging.captureWarnings(capture=True)


def parse_args():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        required=True,
        help="Question to be processed",
    )
    return parser.parse_args()


class ModelWrapper:
    def __init__(self, model):
        self.llm = Ollama(model=model)

    def generate(self, question):
        return self.llm(question)


PROJECT_SUMMARY_PROMPT = """
[INST] You will receive a description of what the user is trying to build.
You will generate a single project name for the code project which should be in very short and limited to 3 words.
Formal instructions: {}
Project Description: {} [/INST]
"""


def project_name_generator(model, question):
    output_parser = StructuredOutputParser.from_response_schemas(
        [ResponseSchema(name="project_name", description="Project Name")]
    )
    format_instructions = output_parser.get_format_instructions()
    final_prompt = PROJECT_SUMMARY_PROMPT.format(format_instructions, question)
    logging.debug(f"Prompt for project name generator: {final_prompt}")
    generated_output = model.generate(final_prompt)
    return Box(output_parser.parse(generated_output)).project_name


def main(args):
    model = ModelWrapper(model="llama2:13b")
    project_name = project_name_generator(model, args.question)
    print(project_name)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
