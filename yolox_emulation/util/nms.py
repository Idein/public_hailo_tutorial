import numpy as np
def iou_np(a, b, a_area, b_area):
    """
    a = (4,)
    b = (N, 4)
    a_area = (1,)
    b_area = (N,)
    """
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    w = np.maximum(0, abx_mx - abx_mn)
    h = np.maximum(0, aby_mx - aby_mn)
    intersect = w*h
    iou_np = intersect / (a_area + b_area - intersect)
    return iou_np

def nms(bboxes, scores, classes, iou_threshold=0.5, per_class=False, max_wh=3000):
    if len(bboxes) == 0:
        return bboxes, scores, classes
    if per_class:
        bboxes[:, 0] += max_wh * classes[:]
        bboxes[:, 1] += max_wh * classes[:]
        bboxes[:, 2] += max_wh * classes[:]
        bboxes[:, 3] += max_wh * classes[:]
    
    sort_index = np.argsort(-scores)
    areas = (bboxes[:,2] - bboxes[:,0]) \
             * (bboxes[:,3] - bboxes[:,1])

    i = 0
    rest = len(areas)
    while(rest > 0):
        max_scr_ind = sort_index[i]
        ind_list = sort_index[i+1:]
        iou = iou_np(bboxes[max_scr_ind], bboxes[ind_list], \
                     areas[max_scr_ind], areas[ind_list])
        del_index, = np.where(iou > iou_threshold)
        sort_index = np.delete(sort_index, del_index + i + 1)
        i += 1
        rest -= (len(del_index) + 1)
    
    bboxes = bboxes[sort_index]
    scores = scores[sort_index]
    classes = classes[sort_index]
    
    if per_class:
        bboxes[:, 0] -= max_wh * classes[:]
        bboxes[:, 1] -= max_wh * classes[:]
        bboxes[:, 2] -= max_wh * classes[:]
        bboxes[:, 3] -= max_wh * classes[:]
    
    return bboxes, scores, classes