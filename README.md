<div align="center">

# IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models | RunPod Serverless Worker

This is the source code for a [RunPod](https://runpod.io?ref=2xxro4sy)
Serverless worker for [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter):
Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models.

<img src="assets/example.png">

</div>

## Model

| Model                                                       | 
|-------------------------------------------------------------|
| [IP-Adapter](https://huggingface.co/h94/IP-Adapter) |
| [CyberRealistic](https://huggingface.co/cyberdelia/CyberRealistic)

## RunPod API Endpoint

You can send requests to your RunPod API Endpoint using the `/run`
or `/runsync` endpoints.

Requests sent to the `/run` endpoint will be handled asynchronously,
and are non-blocking operations.  Your first response status will always
be `IN_QUEUE`.  You need to send subsequent requests to the `/status`
endpoint to get further status updates, and eventually the `COMPLETED`
status will be returned if your request is successful.

Requests sent to the `/runsync` endpoint will be handled synchronously
and are blocking operations.  If they are processed by a worker within
90 seconds, the result will be returned in the response, but if
the processing time exceeds 90 seconds, you will need to handle the
response and request status updates from the `/status` endpoint until
you receive the `COMPLETED` status which indicates that your request
was successful.

### RunPod API Examples

* [JSON Example](docs/generate.md)

### Endpoint Status Codes

| Status      | Description                                                                                                                     |
|-------------|---------------------------------------------------------------------------------------------------------------------------------|
| IN_QUEUE    | Request is in the queue waiting to be picked up by a worker.  You can call the `/status` endpoint to check for status updates.  |
| IN_PROGRESS | Request is currently being processed by a worker.  You can call the `/status` endpoint to check for status updates.             |
| FAILED      | The request failed, most likely due to encountering an error.                                                                   |
| CANCELLED   | The request was cancelled.  This usually happens when you call the `/cancel` endpoint to cancel the request.                    |
| TIMED_OUT   | The request timed out.  This usually happens when your handler throws some kind of exception that does return a valid response. |
| COMPLETED   | The request completed successfully and the output is available in the `output` field of the response.                           |

## Serverless Handler

The serverless handler (`handler.py`) is a Python script that handles
the API requests to your Endpoint using the [runpod](https://github.com/runpod/runpod-python)
Python library.  It defines a function `handler(event)` that takes an
API request (event), runs the inference using [IP-Adapter](
https://github.com/tencent-ailab/IP-Adapter) with the `input`, and returns
the `output` in the JSON response.

## Community and Contributing

Pull requests and issues on [GitHub](https://github.com/thoriqtau/serverless-stablediffusion-ipadapter)
are welcome. Bug fixes and new features are encouraged.
