import requests
import json
import datetime
import argparse
import logging


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run a ReAct agent")
    parser.add_argument(
        "--question", "-q", type=str, required=True, help="The question to answer"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def setup_logging(level: str):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level)


def extract_json(s):
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    counter = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            counter += 1
        elif s[i] == "}":
            counter -= 1
            if counter == 0:
                json_str = s[start : i + 1]
                return json.loads(json_str)
    raise ValueError("No complete JSON object found")


def create_chat_completions_body(
    messages, temperature=0.0, max_tokens=1024, stop_words=None
):
    body = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "stop": [] if stop_words is None else stop_words,
    }
    return body


def get_chat_completions(url, messages, stop_words=None):
    body = create_chat_completions_body(messages, stop_words=stop_words)
    response = requests.post(url, json=body)
    response.raise_for_status()
    response_body = response.json()
    return response_body


# tool functions
def get_current_time(seconds_offset=0):
    current_time = datetime.datetime.now()
    new_time = current_time + datetime.timedelta(seconds=seconds_offset)
    return new_time.strftime("%Y-%m-%d %H:%M:%S")


def get_weather(city, timestamp=None):
    """
    Get the weather of a city at a given timestamp
    """

    return json.dumps({"temperature": 25, "weather": "sunny", "city": city})


url = "http://localhost:8080/v1/chat/completions"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds_offset": {
                        "type": "number",
                        "description": "Offset in seconds",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather of a city at a given timestamp",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "Timestamp",
                    },
                },
            },
            "required": ["city", "timestamp"],
        },
    },
]
tool_names = [tool["function"]["name"] for tool in tools]
tool_maps = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
}

# I adopted the prompt template from the langsmith hub:
# https://smith.langchain.com/hub/hwchase17/react


prompt = f"""Answer the following questions as best as you can. You have access to the following tools:

{json.dumps(tools, indent=4)}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of the [{", ".join(tool_names)}]
Action Input: the input to the action, use a JSON line for the action input. DO NOT ADD anything after the end of the JSON line, do not add comments or an explanation after the JSON line
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


def main(args):
    setup_logging(args.log_level)
    question = args.question
    scratch_pad = prompt + f"\nQuestion: {question}"
    stop_words = ["Observation:"]
    final_answer_found = False
    current_iteration = 0

    logger = logging.getLogger("ReAct")

    while not final_answer_found:
        current_iteration += 1
        llm_response = get_chat_completions(
            url,
            [{"role": "user", "content": scratch_pad}],
            stop_words=stop_words,
        )

        if current_iteration > 10:
            logger.error("Max iterations reached. Exiting")
            print(scratch_pad)
            break

        response_message = llm_response["choices"][0]["message"]["content"]
        response_message = response_message.strip()

        logger.debug(response_message)

        scratch_pad = scratch_pad + f"\n{response_message}"

        if "Final Answer:" in response_message:
            final_answer_found = True
            final_answer = response_message.split("Final Answer:")[1].strip()
            logger.info(final_answer)
            logger.debug(scratch_pad)

        if "Action Input:" in response_message:
            action = response_message.split("Action:")[1].split("\n")[0].strip()
            action_input = (
                response_message.split("Action Input:")[1].split("\n")[0].strip()
            )

            tool = tool_maps.get(action)
            if tool is not None:
                tool_arguments = extract_json(action_input)
                tool_response = tool(**tool_arguments)
                scratch_pad = scratch_pad + f"\nObservation: {tool_response}"
                logger.debug(f"Observation: {tool_response}")
                continue
            else:
                logger.error("Tool not found. Exiting!")
                break


if __name__ == "__main__":
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()
    main(args)
