def use_template(input : list, prompt_template : str, placeholder_template="{INPUT_#}"):
    # placeholder text goes {INPUT_0}, {INPUT_1}, etc.
    prompt = prompt_template
    for i, text in enumerate(input):
        if text is None:
            text = ""
        placeholder_text = placeholder_template.replace("#", str(i))
        prompt = prompt.replace(placeholder_text, text)
    return prompt

# LLM function needed. Assumed to be formatted the same as OpenAI.
def run_template_llm(run_llm, inputs : list, prompt_template : str, examples : list=[], placeholder_template="{INPUT_#}") -> str | None:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for prompt, res in examples:
        messages.append({"role": "user", "content": use_template(prompt, prompt_template, placeholder_template)})
        messages.append({"role": "assistant", "content": res})
    messages.append({"role": "user", "content": use_template(inputs, prompt_template, placeholder_template)})
    response: str = run_llm(messages)
    return response
