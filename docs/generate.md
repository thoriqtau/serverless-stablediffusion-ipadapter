## Request Payload
```json
{
  "input": {
    "prompt": "a portrait of a man, looking directly at the camera",
    "negative_prompt": "(deformed, blurry, long neck, bad collar, cgi, bad anatomy, big body)",
    "face_image": "base64 encoded face image content or image link",
    "style_image": "base64 encoded style image content or image link",
    "num_inference_steps": 80,
    "guidance_scale": 1.0,
    "width": 512,
    "height": 512,
  }
}
```

## Response
### RUN
```json
{
  "id": "83bbc301-5dcd-4236-9293-a65cdd681858",
  "status": "IN_QUEUE"
}
```

### RUNSYNC
```json
{
  "delayTime": 20275,
  "executionTime": 43997,
  "id": "sync-a3b54383-e671-4e24-a7bd-c5fec16fda3b",
  "output": {
    "status": "ok",
    "image": "base64 encoded output image"
  },
  "status": "COMPLETED"
}
```
