from huggingface_hub import snapshot_download

def fetch_models():
    """
    Fetches all models from the HuggingFace model hub.
    """
    snapshot_download("thoriqtau/cyberrealistic-diffuser", 
                      local_dir="./CyberRealistic_V9_FP32")

    snapshot_download("thoriqtau/ipadapter", 
                      local_dir="./ipadapter")

    snapshot_download("thoriqtau/CLIP-ViT-H-14-laion2B-s32B-b79K", 
                      local_dir="./CLIP-ViT-H-14-laion2B-s32B-b79K")
    
if __name__ == '__main__':
    fetch_models()
