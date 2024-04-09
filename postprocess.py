def get_area(res):
    bbox = res[0]
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) 


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    intersection_w = max(0, intersection_x2 - intersection_x1)
    intersection_h = max(0, intersection_y2 - intersection_y1)

    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)
    area_intersection = intersection_w * intersection_h

    iou = area_intersection / (area_box1 + area_box2 - area_intersection)
    return iou, (area_intersection/area_box1, area_intersection/area_box2)


def remove_smaller_bbox(res1, res2):
    area1, area2 = get_area(res1), get_area(res2)
    if area1 > area2:
        return res2
    else:
        return res1
    

def postprocess(res, upper_intersection_threshold, lower_intersection_threshold):
    res_to_remove = []
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if res[i] in res_to_remove or res[j] in res_to_remove: continue
            iou, intersection_area = calculate_iou(res[i][0], res[j][0])

            if  intersection_area[0] > upper_intersection_threshold or intersection_area[1] > upper_intersection_threshold:
                res_to_remove.append(remove_smaller_bbox(res[i], res[j]))
                continue
            if intersection_area[0]>lower_intersection_threshold or intersection_area[1]>lower_intersection_threshold:
                if res[i][1] > res[j][1]:
                    res_to_remove.append(res[j])
                else:
                    res_to_remove.append(res[i])


    for remove_res in res_to_remove:
        if remove_res in res: res.remove(remove_res)
    
    return res

def postprocess_with_classes_rem_picture(res, upper_intersection_threshold, lower_intersection_threshold):
    res_to_remove = []
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if res[i] in res_to_remove or res[j] in res_to_remove: continue
            iou, intersection_area = calculate_iou(res[i][0], res[j][0])
            
            if res[i][2]=='Picture':
                res_to_remove.append(res[i])
            if res[j][2]=='Picture':
                res_to_remove.append(res[j])

            if  intersection_area[0] > upper_intersection_threshold or intersection_area[1] > upper_intersection_threshold:
                # print("LABEL",res[i][2])
                if res[i][2]!='Table' and res[j][2]!='Table': 
                    res_to_remove.append(remove_smaller_bbox(res[i], res[j]))
                    continue
                elif res[i][2]=='Table' and res[j][2]=='Table': # Fixed if->elif from Feb 8 update
                    res_to_remove.append(remove_smaller_bbox(res[i], res[j]))
                    continue
                elif res[i][2]=='Table':
                    res_to_remove.append(res[j])
                    continue
                elif res[j][2]=='Table':
                    res_to_remove.append(res[i])
                    continue
                    
            elif intersection_area[0]>lower_intersection_threshold or intersection_area[1]>lower_intersection_threshold:
                # print("LABEL LOW",res[i][2])
                if res[i][2]!='Table' and res[j][2]!='Table': 
                
                    if res[i][1] > res[j][1]:
                        res_to_remove.append(res[j])
                    else:
                        res_to_remove.append(res[i])


    for remove_res in res_to_remove:
        if remove_res in res: res.remove(remove_res)
    
    return res

def postprocess_rem_picture_rem_low_conf_table(res, upper_intersection_threshold, lower_intersection_threshold):
    res_to_remove = []
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if res[i] in res_to_remove or res[j] in res_to_remove: continue
            iou, intersection_area = calculate_iou(res[i][0], res[j][0])
            
            if res[i][2]=='Picture':
                res_to_remove.append(res[i])
            if res[j][2]=='Picture':
                res_to_remove.append(res[j])

            if  intersection_area[0] > upper_intersection_threshold or intersection_area[1] > upper_intersection_threshold:
                # print("LABEL",res[i][2])
                if res[i][2]!='Table' and res[j][2]!='Table': 
                    res_to_remove.append(remove_smaller_bbox(res[i], res[j]))
                    continue
                elif res[i][2]=='Table' and res[j][2]=='Table': # Fixed if->elif from Feb 8 update
                    # res_to_remove.append(remove_smaller_bbox(res[i], res[j]))
                    if res[i][1] > res[j][1]:
                        res_to_remove.append(res[j])
                    else:
                        res_to_remove.append(res[i])
                    continue
                elif res[i][2]=='Table':
                    res_to_remove.append(res[j])
                    continue
                elif res[j][2]=='Table':
                    res_to_remove.append(res[i])
                    continue
                    
            elif intersection_area[0]>lower_intersection_threshold or intersection_area[1]>lower_intersection_threshold:
                # print("LABEL LOW",res[i][2])
                if res[i][2]!='Table' and res[j][2]!='Table': 
                
                    if res[i][1] > res[j][1]:
                        res_to_remove.append(res[j])
                    else:
                        res_to_remove.append(res[i])
                        
                elif res[i][2]=='Table' and res[j][2]=='Table':
                    if res[i][1] > res[j][1]:
                        res_to_remove.append(res[j])
                    else:
                        res_to_remove.append(res[i])
                    


    for remove_res in res_to_remove:
        if remove_res in res: res.remove(remove_res)
    
    return res


def postprocess_dit(unprocessed_dict):
    threshold = 0.25
    upper_intersection_threshold = 0.9
    lower_intersection_threshold = 0.30
    
    res_dict = {}
    res = []
    unprocessed_dict = unprocessed_dict['jsonData']
    boxes = unprocessed_dict['boxes']
    classes = unprocessed_dict['classes']
    scores = unprocessed_dict['scores']

    for idx in range(len(boxes)):
        if scores[idx] > threshold:
            res.append((boxes[idx], scores[idx], classes[idx]))
    final_result = postprocess_rem_picture_rem_low_conf_table(res, upper_intersection_threshold, lower_intersection_threshold)

    boxes, classes, scores = [], [], []
    for idx, tab in enumerate(final_result):
        boxes.append(tab[0])
        classes.append(tab[2])
        scores.append(tab[1])

    # res_dict['jsonData'] = {
    #     "boxes": boxes,
    #     "classes": classes,
    #     "scores": scores,
    #     "image" : unprocessed_dict['image']
    #     # 'LEN': unprocessed_dict['LEN'],
    #     # 'IMAGE_SHAPE': unprocessed_dict['IMAGE_SHAPE']
    #  }
    # return res_dict
    sorted_data = sorted(zip(boxes, scores, classes), key=lambda x: x[0][1])
    sorted_boxes, sorted_scores, sorted_classes = zip(*sorted_data)

    res_dict['jsonData'] = {
        "boxes": sorted_boxes,
        "classes": sorted_classes,
        "scores": sorted_scores,
        "image" : unprocessed_dict['image']
    }
    return res_dict    