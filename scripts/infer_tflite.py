"""Raw TFLite inference for best_int8.tflite (YOLO11s L1 food detector).

Usage:
    python scripts/infer_tflite.py --model best_int8.tflite --image photo.jpg
    python scripts/infer_tflite.py --model best_int8.tflite --image photo.jpg --save out.jpg
"""

import argparse

import cv2
import numpy as np
import tensorflow as tf

L1_CLASSES = [
    "noodle_dish",
    "rice_dish",
    "soup_stew",
    "grilled_fried",
    "banh_bread",
    "beverage",
    "fruit",
    "dessert_snack",
]
IMG_SIZE   = 640
CONF_THRES = 0.25
IOU_THRES  = 0.45


def preprocess(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h0, w0 = img.shape[:2]
    blob = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
    blob = blob.astype(np.float32) / 255.0
    return img, np.expand_dims(blob, 0), (h0, w0)


def xywh2xyxy(boxes: np.ndarray, orig_shape):
    h0, w0 = orig_shape
    sx, sy = w0 / IMG_SIZE, h0 / IMG_SIZE
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([
        (cx - w / 2) * sx, (cy - h / 2) * sy,
        (cx + w / 2) * sx, (cy + h / 2) * sy,
    ], axis=1)


def postprocess(raw: np.ndarray, orig_shape):
    preds       = raw[0].T                  # [8400, 12]
    class_scores = preds[:, 4:]             # [8400, 8]
    class_ids   = class_scores.argmax(1)
    confs       = class_scores.max(1)

    mask = confs >= CONF_THRES
    preds, confs, class_ids = preds[mask], confs[mask], class_ids[mask]
    if len(preds) == 0:
        return []

    boxes   = xywh2xyxy(preds[:, :4], orig_shape)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), confs.tolist(), CONF_THRES, IOU_THRES
    )
    results = []
    for i in indices:
        results.append({
            "label": L1_CLASSES[class_ids[i]],
            "conf":  float(confs[i]),
            "xyxy":  [round(v, 1) for v in boxes[i].tolist()],
        })
    return results


def draw(img: np.ndarray, detections: list) -> np.ndarray:
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        label = f"{d['label']} {d['conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best_int8.tflite")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--save",  default="",   help="Save annotated image to this path")
    ap.add_argument("--conf",  type=float, default=CONF_THRES, help="Confidence threshold")
    args = ap.parse_args()

    global CONF_THRES
    CONF_THRES = args.conf

    # Load interpreter
    interp = tf.lite.Interpreter(model_path=args.model)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    print(f"Input  : {inp_det['name']}  shape={inp_det['shape']}  dtype={inp_det['dtype'].__name__}")
    print(f"Output : {out_det['name']}  shape={out_det['shape']}  dtype={out_det['dtype'].__name__}")

    # Preprocess
    img, blob, orig_shape = preprocess(args.image)
    print(f"Image  : {args.image}  orig={orig_shape}")

    # Quantize input if the model expects uint8
    if inp_det["dtype"] == np.uint8:
        scale, zp = inp_det["quantization"]
        blob = (blob / scale + zp).clip(0, 255).astype(np.uint8)

    # Run
    interp.set_tensor(inp_det["index"], blob)
    interp.invoke()
    raw = interp.get_tensor(out_det["index"]).astype(np.float32)

    # Dequantize output if int8
    if out_det["dtype"] == np.int8:
        scale, zp = out_det["quantization"]
        raw = (raw.astype(np.float32) - zp) * scale

    print(f"Raw output shape: {raw.shape}")

    # Postprocess
    detections = postprocess(raw, orig_shape)

    if not detections:
        print("No detections above conf threshold.")
    else:
        print(f"\n{len(detections)} detection(s):")
        for d in detections:
            print(f"  {d['label']:20s}  conf={d['conf']:.3f}  xyxy={d['xyxy']}")

    if args.save:
        annotated = draw(img, detections)
        cv2.imwrite(args.save, annotated)
        print(f"\nAnnotated image saved to: {args.save}")


if __name__ == "__main__":
    main()
