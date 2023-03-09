import argparse
import functools
import os
import time

import cv2
import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image

from facenet.detection.face_detect import MTCNN
from facenet.utils.utils import add_arguments, print_arguments

from model import ssdlite_with_weights, get_model
from detection import predict as ssdpred

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('image_path',               str,     'dataset/test.jpg',                 '预测图片路径')
add_arg('face_db_path',             str,     'facenet/face_db',                          '人脸库路径')
add_arg('threshold',                float,   0.6,                                '判断相识度的阈值')
add_arg('mobilefacenet_model_path', str,     'facenet/save_model/mobilefacenet.pth',     'MobileFaceNet预测模型的路径')
add_arg('mtcnn_model_path',         str,     'facenet/save_model/mtcnn',                 'MTCNN预测模型的路径')
args = parser.parse_args()
print_arguments(args)


class Predictor:
    def __init__(self, mtcnn_model_path, mobilefacenet_model_path, face_db_path, threshold=0.7):
        self.threshold = threshold
        self.mtcnn = MTCNN(model_path=mtcnn_model_path)
        self.device = torch.device("cuda")

        # 加载模型
        self.model = torch.jit.load(mobilefacenet_model_path)
        self.model.to(self.device)
        self.model.eval()

        self.faces_db = self.load_face_db(face_db_path)

    def load_face_db(self, face_db_path):
        faces_db = {}
        for path in os.listdir(face_db_path):
            name = os.path.basename(path).split('.')[0]
            image_path = os.path.join(face_db_path, path)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            imgs, _ = self.mtcnn.infer_image(img)
            if imgs is None or len(imgs) > 1:
                print('人脸库中的 %s 图片包含不是1张人脸，自动跳过该图片' % image_path)
                continue
            imgs = self.process(imgs)
            feature = self.infer(imgs[0])
            faces_db[name] = feature[0][0]
        return faces_db

    @staticmethod
    def process(imgs):
        imgs1 = []
        for img in imgs:
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            imgs1.append(img)
        return imgs1

    # 预测图片
    def infer(self, imgs):
        assert len(imgs.shape) == 3 or len(imgs.shape) == 4
        if len(imgs.shape) == 3:
            imgs = imgs[np.newaxis, :]
        # TODO 不知为何不支持多张图片预测
        '''
        imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)
        features = self.model(img)
        features = features.detach().cpu().numpy()
        '''
        features = []
        for i in range(imgs.shape[0]):
            img = imgs[i][np.newaxis, :]
            img = torch.tensor(img, dtype=torch.float32, device=self.device)
            # 执行预测
            feature = self.model(img)
            feature = feature.detach().cpu().numpy()
            features.append(feature)
        return features

    def recognition(self, image_path):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        s = time.time()
        imgs, boxes = self.mtcnn.infer_image(img)
        print('人脸检测时间：%dms' % int((time.time() - s) * 1000))
        if imgs is None:
            return None, None
        imgs = self.process(imgs)
        imgs = np.array(imgs, dtype='float32')
        s = time.time()
        features = self.infer(imgs)
        print('人脸识别时间：%dms' % int((time.time() - s) * 1000))
        names = []
        probs = []
        for i in range(len(features)):
            feature = features[i][0]
            results_dict = {}
            for name in self.faces_db.keys():
                feature1 = self.faces_db[name]
                prob = np.dot(feature, feature1) / (np.linalg.norm(feature) * np.linalg.norm(feature1))
                results_dict[name] = prob
            results = sorted(results_dict.items(), key=lambda d: d[1], reverse=True)
            print('人脸对比结果：', results)
            result = results[0]
            prob = float(result[1])
            probs.append(prob)
            if prob > self.threshold:
                name = result[0]
                names.append(name)
            else:
                names.append('unknow')
        return boxes, names

    def add_text(self, img, text, left, top, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('facenet/simfang.ttf', size)
        draw.text((left, top), text, color, font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 画出人脸框和关键点
    def draw_face(self, image_path, boxes_c, names):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if boxes_c is not None:
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                name = names[i]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # 画人脸框
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                # 判别为人脸的名字
                img = self.add_text(img, name, corpbbox[0], corpbbox[1] -15, color=(0, 0, 255), size=12)
        cv2.imshow("result", img)
        cv2.waitKey(0)

    def draw_box(self, image_path, boxes_face, boxes_body, names):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if boxes_face is not None:
            for i in range(boxes_face.shape[0]):
                bbox_face = boxes_face[i, :4]
                bbox_body = boxes_body[i]
                name = names[i]
                # 画人脸框
                cv2.rectangle(img, (int(bbox_face[0]), int(bbox_face[1])),
                              (int(bbox_face[2]), int(bbox_face[3])), (255, 0, 0), 1)
                # 判别为人脸的名字
                img = self.add_text(img, name, int(bbox_face[0]), int(bbox_face[1]) -15, color=(0, 0, 255), size=15)
                # 画人体框
                cv2.rectangle(img, (int(bbox_body[0]), int(bbox_body[1])),
                              (int(bbox_body[2]), int(bbox_body[3])), (0, 255, 0), 1)
                # 判别为人脸的名字
                img = self.add_text(img, name, int(bbox_body[0]), int(bbox_body[1]) -15, color=(0, 0, 255), size=15)
        cv2.imshow("result", img)
        cv2.waitKey(0)

    def print_net(self):
        from torchsummary import summary
        summary(self.model, (3, 500, 500))

if __name__ == '__main__':
    ssdlite = get_model("cuda", True).eval()
    predictor = Predictor(args.mtcnn_model_path, args.mobilefacenet_model_path, args.face_db_path, threshold=args.threshold)
    start = time.time()
    face_boxes, names = predictor.recognition(args.image_path)
    # print('预测的人脸位置：', boxes.astype(np.int_).tolist())
    # print('识别的人脸名称：', names)
    # print('总识别时间：%dms' % int((time.time() - start) * 1000))
    image = Image.open(args.image_path)
    body_boxes, _, scores, labels = ssdpred(image,ssdlite,"cuda",args.threshold)
    face_body_match = []
    for i, fb in enumerate(face_boxes):
        bodys = []
        for j, bb in enumerate(body_boxes):
            if fb[0] < bb[0]: continue
            if fb[1] < bb[1]: continue
            if fb[2] > bb[2]: continue
            if fb[3] > bb[3]: continue
            bodys.append(bb)
        if len(bodys) > 1:
            
            x_dist = []
            for bb in bodys:
                body_x_center = (bb[0]+bb[2])/2
                face_x_center = (fb[0]+fb[2])/2
                x_dist.append(abs(body_x_center-face_x_center))
            idx = x_dist.index(min(x_dist))
            face_body_match.append(bodys[idx])
        elif len(bodys) == 1:
            face_body_match.append(bodys[0])
        else:
            raise Warning("Human detection is not ideal.")
    predictor.draw_box(args.image_path, face_boxes, face_body_match, names)
