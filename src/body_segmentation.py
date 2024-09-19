from transformers import pipeline
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw


class FaceRemover:
    def __init__(self, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], ctx_id=0, det_size=(640, 640)):
        self.face_analyzer = FaceAnalysis(providers=providers)
        self.face_analyzer.prepare(ctx_id=ctx_id, det_size=det_size)

    def remove_face(self, img, mask):
        img_arr = np.asarray(img)
        faces = self.face_analyzer.get(img_arr)
        if not faces:
            return mask
        face_bbox = self._expand_bbox(faces[0]['bbox'])
        self._draw_blackout(mask, face_bbox)
        return mask

    def _expand_bbox(self, bbox):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return [
            bbox[0] - w * 0.5, bbox[1] - h * 0.5,
            bbox[2] + w * 0.5, bbox[3] + h * 0.2
        ]

    def _draw_blackout(self, mask, bbox):
        face_coords = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
        ImageDraw.Draw(mask).rectangle(face_coords, fill=0)


class Segmenter:
    def __init__(self, model_name="mattmdjaga/segformer_b2_clothes"):
        self.segmenter = pipeline(model=model_name)

    def segment(self, img, labels):
        segments = self.segmenter(img)
        return [s['mask'] for s in segments if s['label'] in labels]

    def combine_masks(self, masks):
        if not masks:
            raise ValueError("No masks to combine.")
        return Image.fromarray(np.logical_or.reduce(masks).astype(np.uint8) * 255)


class ImageProcessor:
    def __init__(self, face_remover, segmenter):
        self.face_remover = face_remover
        self.segmenter = segmenter

    def process(self, img, labels, remove_face=False):
        masks = self.segmenter.segment(img, labels)
        final_mask = self.segmenter.combine_masks(masks)
        if remove_face:
            final_mask = self.face_remover.remove_face(img.convert('RGB'), final_mask)
        img.putalpha(final_mask)
        return img, final_mask


def segment_body(img, include_face=True):
    labels = ["Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt",
              "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", 
              "Bag", "Scarf"]
    return ImageProcessor(FaceRemover(), Segmenter()).process(img, labels, not include_face)


def segment_torso(img):
    labels = ["Upper-clothes", "Dress", "Belt", "Face", "Left-arm", "Right-arm"]
    return ImageProcessor(FaceRemover(), Segmenter()).process(img, labels, remove_face=True)
