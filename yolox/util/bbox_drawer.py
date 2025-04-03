import cv2

COLOR_MAP = [(246, 13, 26),
             (255, 153, 51),
             (80, 238, 171), (117, 243, 84),
             (217, 83, 252), (132, 78, 235),
             (204, 204, 51), (255, 255, 153),
             (244, 76, 134), (251, 80, 205)]

class Drawer():
    def __init__(self, class_names=None, colors=COLOR_MAP, thickness=2, fontScale=0.5):
        self.class_names = class_names
        self.colors = colors
        self.thickness = thickness
        self.fontScale = fontScale

    def draw(self, cv_img, bboxes, scores, classes):
        """
        Args:
            cv_img : draw_image
            bboxes : (N, 4)
            scores : (N,)
            classes : (N,)
        """
        for bbox, score, cls_idx in zip(bboxes, scores, classes):
            x1, y1, x2, y2 = [int(p) for p in bbox]
            col = self.colors[cls_idx % len(self.colors)]
            class_name = self.class_names[cls_idx] if self.class_names else ""
            labeltxt = f"{class_name}:" + "{:.2f}".format(score)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), col, thickness=self.thickness)
            cv2.rectangle(cv_img, (x1, y1 - 15), (x1 + len(labeltxt) * 10, y1), col, -1)
            cv2.putText(cv_img, labeltxt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, self.fontScale, (255, 255, 255), 1)

        return cv_img