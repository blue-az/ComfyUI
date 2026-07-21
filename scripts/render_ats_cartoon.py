import json
import time
import urllib.parse
import urllib.request
import uuid

SERVER_ADDRESS = "127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())

POSITIVE_PROMPT = (
    "editorial political cartoon, black and white newspaper ink drawing, "
    "sharp satirical illustration, exaggerated proportions, readable composition, "
    "a huge elephant representing companies is itself awkwardly riding a tiny child's tricycle "
    "representing an applicant tracking system, the elephant is visibly far too large "
    "for the tricycle, the elephant is seated on the tiny tricycle seat and gripping the handlebars, "
    "bent knees, strained posture, unstable balance, frustrated expression, "
    "the tricycle looks flimsy and absurdly undersized, clear visual metaphor about overreliance "
    "on ATS, a few scattered resumes on the ground, simple white background, classic newspaper cartoon"
)

NEGATIVE_PROMPT = (
    "color, photorealistic, painterly, 3d render, blurry, low contrast, "
    "text, caption, speech bubble, watermark, logo, human, person, rider, businessman, "
    "separate bicycle, separate rider, multiple characters, "
    "extra limbs, extra wheels, "
    "deformed anatomy, cluttered background"
)

PROMPT = {
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
            "text": POSITIVE_PROMPT,
        },
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": NEGATIVE_PROMPT,
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
            "filename_prefix": "ats_cartoon_draft",
            "images": ["8", 0],
        },
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
    images = get_images(PROMPT)
    for image in images:
        params = urllib.parse.urlencode(
            {
                "filename": image["filename"],
                "subfolder": image["subfolder"],
                "type": image["type"],
            }
        )
        print(f"http://{SERVER_ADDRESS}/view?{params}")
