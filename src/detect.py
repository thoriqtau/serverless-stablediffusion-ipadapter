from insightface.app import FaceAnalysis
from utils import crop_face

INSIGHTFACE_MODEL_NAME = "buffalo_l"
INSIGHTFACE_MODEL = './insightface_models/models'

def face_detection(face_image, style_image):
  app = FaceAnalysis(name=INSIGHTFACE_MODEL_NAME, 
                    root=INSIGHTFACE_MODEL, 
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
  app.prepare(ctx_id=0, det_size=(640, 640))

  face_info = app.get(face_image)
  style_face_info = app.get(style_image)

  if len(face_info) == 0:
    raise Exception("No face detected in the user face image! Please upload an image with one visible face.")
  elif len(face_info) > 1:
    raise Exception("Multiple faces detected in the user face image! Please upload an image with only one face.")

  if len(style_face_info) == 0:
    raise Exception("No face detected in the style image! Please upload a style image with one visible face.")
  elif len(style_face_info) > 1:
    raise Exception("Multiple faces detected in the style image! Please upload a style image with only one face.")
  
  face_crop = crop_face(face_image, face_info[0])

  return face_crop, style_image
