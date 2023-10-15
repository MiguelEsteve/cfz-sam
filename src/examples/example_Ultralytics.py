from ultralytics.yolo.engine.model import YOLO

model = YOLO('C:/repos/cfz-sam/src/yolov8n.pt')
m = model.model
print(type(m))

