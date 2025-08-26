# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

from vllm import LLM, SamplingParams

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
os.environ["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "1"

# Sample prompts.
prompts = [
    "Hello, my name is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(model="/data/models/Qwen3-30B-A3B", enforce_eager=True, tensor_parallel_size=4)

    llm.start_profile()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)
    # Prompt: 'Hello, my name is'
    # Generated text: ' Jamie and I'm a new user to the team, so I'll start with'
    # Generated Token Count: 16

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    main()
