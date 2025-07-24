from openai import OpenAI
from requests.exceptions import Timeout
from llm_handler import run_template_llm
from time import sleep
import sys

from config_data import config_data

llm_key_file = config_data.get("llm_key_file")
llm_wait_time = config_data.get("llm_wait_time")
llm_max_retries = config_data.get("llm_max_retries")
endpoint = config_data.get("llm_endpoint")
llm_for_transformation = config_data.get("llm_for_transformation")

if llm_max_retries is None or llm_max_retries <= 0:
    llm_max_retries = 9999999

def read_sec_token():
    """
    get the security token stored in a non versioned file needed to access the endpoint
    @return:
    """
    print("Reading security token...")
    f = open(llm_key_file, "rt")
    mytoken = f.read()
    f.close()
    return (mytoken)

api_key = read_sec_token()

client = OpenAI(api_key = api_key, base_url = endpoint)

def get_llm_function(model):
    def run_llm(messages, max_retries=llm_max_retries):

        for attempt in range(max_retries):
            try:
                chat_completion = client.chat.completions.create(
                    model=model,
                	messages=messages,
                	stream = False,
                )

                generated_text = chat_completion.choices[0].message.content
                generated_text = str(generated_text)
                return generated_text
            
            except Timeout:
                print(f"Request timed out. Attempt {attempt + 1} of {max_retries}. Retrying...")
            except Exception as e:
                print(f"An error occurred: {e}")
                try:
                    # Ignore if the error is a content filtering error
                    filter_error_msgs = [
                        "Invalid response object from API: \'{\"detail\":\"Error code: 400 - {\\\'error\\\': {\\\'message\\\': \\\\\"The response was filtered due to the prompt triggering Azure OpenAI\\\'s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766\\\\\", \\\'type\\\': None, \\\'param\\\': \\\'prompt\\\', \\\'code\\\': \\\'content_filter\\\', \\\'status\\\': 400}}\"}\' (HTTP response code was 500)",
                        "Error code: 500 - {\'detail\': \'Error code: 400 - {\\\'error\\\': {\\\'message\\\': \"The response was filtered due to the prompt triggering Azure OpenAI\\\'s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766\", \\\'type\\\': None, \\\'param\\\': \\\'prompt\\\', \\\'code\\\': \\\'content_filter\\\', \\\'status\\\': 400}}\'}",
                    ]
                    if e.message in filter_error_msgs or e.user_message in filter_error_msgs or e.code == 'content_filter' or str(e) in filter_error_msgs or "The response was filtered due to the prompt triggering Azure OpenAI" in str(e):
                        print("Warning: Content filtering error")
                        return "The response was filtered due to the prompt triggering Azure OpenAI\'s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766"
                    rep_error_msgs = [
                        "An error occurred: Error code: 500 - {\'detail\': \'Error code: 400 - {\\\'error\\\': {\\\'message\\\': \"Sorry! We\\\'ve encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt.\", \\\'type\\\': \\\'invalid_request_error\\\', \\\'param\\\': \\\'prompt\\\', \\\'code\\\': \\\'invalid_prompt\\\'}}\'}"
                    ]
                    if e.message in rep_error_msgs or e.user_message in rep_error_msgs or str(e) in rep_error_msgs:
                        print("Warning: Repetitive patterns error")
                        return "Sorry! We\'ve encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt."
                except:
                    pass

                print(f"Attempt {attempt + 1} of {max_retries}. Retrying...")
                sleep(llm_wait_time)

        print(f"Failed to get response after {max_retries} attempts")
        sys.exit(1)
        # return None

    return run_llm


# this function is used as a tool; this llm is not being tested
def run_template_gpt(inputs : list, prompt_template : str, examples : list=[], placeholder_template="{INPUT_#}"):
    if not isinstance(inputs, list):
        inputs = [inputs]
    llm_func = get_llm_function(llm_for_transformation)
    return run_template_llm(llm_func, inputs, prompt_template, examples, placeholder_template)
