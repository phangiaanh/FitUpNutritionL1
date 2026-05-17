"""Ultralytics inference for best.pt (YOLO11s L1 food detector).

Usage:
    python scripts/infer_pt.py --model best.pt --image photo.jpg
    python scripts/infer_pt.py --model best.pt --image photo.jpg --save out.jpg
"""

import argparse

from ultralytics import YOLO

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
CONF_THRES = 0.25
IOU_THRES  = 0.45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best.pt")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--save",  default="",    help="Save annotated image to this path")
    ap.add_argument("--conf",  type=float, default=CONF_THRES, help="Confidence threshold")
    ap.add_argument("--iou",   type=float, default=IOU_THRES,  help="NMS IoU threshold")
    args = ap.parse_args()

    model = YOLO(args.model)

    results = model.predict(
        source=args.image,
        conf=args.conf,
        iou=args.iou,
        imgsz=640,
        save=False,
        verbose=False,
    )

    res = results[0]
    h0, w0 = res.orig_shape
    print(f"Image  : {args.image}  orig=({h0},{w0})")
    print(f"Model  : {args.model}")

    if res.boxes is None or len(res.boxes) == 0:
        print("No detections above conf threshold.")
    else:
        print(f"\n{len(res.boxes)} detection(s):")
        for box in res.boxes:
            cls  = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
            label = L1_CLASSES[cls] if cls < len(L1_CLASSES) else str(cls)
            print(f"  {label:20s}  conf={conf:.3f}  xyxy={xyxy}")

    if args.save:
        annotated = res.plot()
        import cv2
        cv2.imwrite(args.save, annotated)
        print(f"\nAnnotated image saved to: {args.save}")


if __name__ == "__main__":
    main()
