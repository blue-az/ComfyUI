import json
import time
import urllib.parse
import urllib.request
import uuid
import subprocess
import os
import re

SERVER_ADDRESS = "127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())

PROMPT_TEMPLATE = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 7.0,
            "denoise": 1,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "seed": 872441993,
            "steps": 28,
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "sd_xl_base_1.0.safetensors",
        },
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024,
        },
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": "", # will be filled
        },
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": (
                "color, photorealistic, painterly, 3d render, blurry, low contrast, "
                "text, caption, speech bubble, watermark, logo, human, person, rider, businessman, "
                "separate bicycle, separate rider, multiple characters, extra limbs, extra wheels, "
                "deformed anatomy, cluttered background"
            ),
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2],
        },
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "", # will be filled
            "images": ["8", 0],
        },
    },
}

def query_ollama(model: str, system_prompt: str, user_prompt: str) -> str:
    url = "http://127.0.0.1:11434/api/generate"
    prompt_with_system = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
    payload = {
        "model": model,
        "prompt": prompt_with_system,
        "stream": False,
        "options": {
            "num_predict": 1024,
            "temperature": 0.3,
        }
    }
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            decoded = json.loads(body)
            response_text = decoded.get("response", "").strip()
            # Clean think blocks
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            # Clean markdown codeblocks
            response_text = re.sub(r"```[a-zA-Z]*\n?", "", response_text)
            response_text = response_text.replace("```", "").strip()
            # Clean double quotes around the whole string if present
            if response_text.startswith('"') and response_text.endswith('"'):
                response_text = response_text[1:-1].strip()
            return response_text
    except Exception as e:
        print(f"Error querying {model}: {e}")
        return ""

def check_comfy_running() -> bool:
    try:
        with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/history/test", timeout=2) as resp:
            return resp.getcode() == 200
    except Exception:
        return False

def queue_prompt(prompt_dict: dict, prompt_id: str) -> None:
    payload = {"prompt": prompt_dict, "client_id": CLIENT_ID, "prompt_id": prompt_id}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://{SERVER_ADDRESS}/prompt", data=data)
    urllib.request.urlopen(req).read()

def get_history(prompt_id: str) -> dict:
    with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/history/{prompt_id}") as response:
        return json.loads(response.read())

def render_image(prompt_text: str, filename_prefix: str) -> list[dict]:
    prompt_dict = json.loads(json.dumps(PROMPT_TEMPLATE))
    prompt_dict["6"]["inputs"]["text"] = prompt_text
    prompt_dict["9"]["inputs"]["filename_prefix"] = filename_prefix
    
    prompt_id = str(uuid.uuid4())
    queue_prompt(prompt_dict, prompt_id)
    history = None
    for _ in range(600):
        try:
            current = get_history(prompt_id)
            if prompt_id in current:
                history = current[prompt_id]
                break
        except Exception:
            pass
        time.sleep(1)

    if history is None:
        raise RuntimeError(f"Timed out waiting for prompt {prompt_id}")

    saved = []
    for node_output in history.get("outputs", {}).values():
        for image in node_output.get("images", []):
            saved.append(image)
    return saved

def main():
    system_instruction = (
        "You are a master Stable Diffusion prompt engineer. Write a single, highly detailed positive text prompt "
        "for Stable Diffusion XL (SDXL) representing a satirical newspaper political cartoon.\n"
        "The metaphor: a huge corporate elephant awkwardly riding a tiny child's tricycle (representing an applicant tracking system).\n"
        "Ensure the style is: black and white newspaper ink drawing, detailed pen and ink crosshatching, "
        "sharp satirical illustration, exaggerated proportions, classic newspaper cartoon, simple white background.\n"
        "Do NOT include any introduction, explanations, notes, or markdown backticks. Output ONLY the raw prompt string ready to be fed into Stable Diffusion."
    )
    user_instruction = "Generate the Stable Diffusion XL prompt for the political cartoon described."

    print("Querying gemma4:12b for image prompt...")
    prompt_12b = query_ollama("gemma4:12b", system_instruction, user_instruction)
    print(f"\n[gemma4:12b Prompt]:\n{prompt_12b}\n")

    print("Querying gemma4:26b for image prompt...")
    prompt_26b = query_ollama("gemma4:26b", system_instruction, user_instruction)
    print(f"\n[gemma4:26b Prompt]:\n{prompt_26b}\n")

    if not prompt_12b or not prompt_26b:
        print("Failed to get prompts from models.")
        return

    # Save prompts for verification
    with open("/home/blueaz/Python/Evaluation/ComfyUI/output/prompts_compare.json", "w") as f:
        json.dump({"gemma4_12b": prompt_12b, "gemma4_26b": prompt_26b}, f, indent=2)

    # Start ComfyUI if not running
    comfy_proc = None
    if not check_comfy_running():
        print("Starting ComfyUI server in the background...")
        comfy_proc = subprocess.Popen(
            ["/home/blueaz/miniconda3/envs/facenet_fresh/bin/python3", "/home/blueaz/Python/Evaluation/ComfyUI/main.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        for _ in range(30):
            if check_comfy_running():
                print("ComfyUI server started successfully.")
                break
            time.sleep(2)
        else:
            raise RuntimeError("Failed to start ComfyUI server")
    else:
        print("ComfyUI server is already running.")

    try:
        print("Rendering image for gemma4:12b prompt...")
        images_12b = render_image(prompt_12b, "ats_cartoon_12b")
        print(f"Generated 12b image: {images_12b}")

        print("Rendering image for gemma4:26b prompt...")
        images_26b = render_image(prompt_26b, "ats_cartoon_26b")
        print(f"Generated 26b image: {images_26b}")
    finally:
        if comfy_proc:
            print("Stopping ComfyUI server...")
            comfy_proc.terminate()
            comfy_proc.wait()
            print("ComfyUI server stopped.")

if __name__ == "__main__":
    main()
