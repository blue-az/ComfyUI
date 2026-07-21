import json
import time
import urllib.parse
import urllib.request
import uuid


SERVER_ADDRESS = "127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())

POSITIVE_PROMPT = (
    "finished editorial political cartoon, black and white newspaper ink drawing, "
    "detailed pen and ink crosshatching, sharp satirical illustration, exaggerated proportions, readable composition, "
    "a huge elephant representing companies is itself awkwardly riding a tiny child's tricycle "
    "representing an applicant tracking system, the elephant is seated on the tiny tricycle seat, "
    "its front legs reaching for the handlebars, bent knees, strained posture, unstable balance, "
    "the tricycle is absurdly undersized and fragile, clear satirical visual metaphor, "
    "scattered resumes on the ground, classic newspaper cartoon, strong linework, monochrome, "
    "clean finished illustration, not a sketch"
)

NEGATIVE_PROMPT = (
    "rough sketch, stick figure, scribble, unfinished lines, color, photorealistic, painterly, 3d render, blurry, text, caption, speech bubble, "
    "human, rider, businessman, stroller, carriage, wagon, multiple characters, cluttered background"
)

PROMPT = {
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    },
    "5": {
        "class_type": "LoadImage",
        "inputs": {"image": "ats_cartoon_sketch.png"},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": POSITIVE_PROMPT},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": NEGATIVE_PROMPT},
    },
    "8": {
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["5", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 774120552,
            "steps": 34,
            "cfg": 7.0,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 0.9,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["8", 0],
        },
    },
    "10": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["9", 0], "vae": ["4", 2]},
    },
    "11": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ats_cartoon_img2img", "images": ["10", 0]},
    },
}


def queue_prompt(prompt: dict, prompt_id: str) -> None:
    payload = {"prompt": prompt, "client_id": CLIENT_ID, "prompt_id": prompt_id}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://{SERVER_ADDRESS}/prompt", data=data)
    urllib.request.urlopen(req).read()


def get_history(prompt_id: str) -> dict:
    with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/history/{prompt_id}") as response:
        return json.loads(response.read())


def get_images(prompt: dict) -> list[dict]:
    prompt_id = str(uuid.uuid4())
    queue_prompt(prompt, prompt_id)
    history = None
    for _ in range(600):
        current = get_history(prompt_id)
        if prompt_id in current:
            history = current[prompt_id]
            break
        time.sleep(1)
    if history is None:
        raise RuntimeError(f"Timed out waiting for prompt {prompt_id}")
    saved = []
    for node_output in history.get("outputs", {}).values():
        for image in node_output.get("images", []):
            saved.append(image)
    return saved


if __name__ == "__main__":
    for image in get_images(PROMPT):
        params = urllib.parse.urlencode(
            {
                "filename": image["filename"],
                "subfolder": image["subfolder"],
                "type": image["type"],
            }
        )
        print(f"http://{SERVER_ADDRESS}/view?{params}")
