csv_to_json.py
- Converts csv annotations to json annotations

main.py
- Oscar chair detection to detect bbox around chairs
- Uses manually labelled ROIs JSON to show where chairs are expected
- Uses this ROI to know that only one chair should be found in this ROI (prevents duplicates)
- Yolov8 detection for people
- Checks for overlap between bbox around people and bbox around chairs

main2.py - BAD
- Same as main.py except:
- Has option to use in-built YOLO persistance tracking

main3.py
- Same as main.py except:
- Only person detect every 100 frames
- Do 10 detections
- Occupied if IoU is 30% of the time

main4.py
- Same as main3.py except:
- Add in threshold level to detect bags /coats
- If confidence of chair deteciton is below a certain level vall it occupied
- NEED TO DO - NOT SURE IF WILL WORK