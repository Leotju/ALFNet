import os
import cv2
import cPickle
import numpy as np
from keras_alfnet.utils.bbox import box_op
from scipy import io as scio
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image

ann = COCO('/home/jn1/data/crowdhuman/annotations_new/train_20_02.json')
root_dir = 'data/crowd'
data_dir = '/home/jn1/data/crowdhuman/'
all_img_path = os.path.join(data_dir, 'images')
all_anno_path = os.path.join(data_dir, 'annotations')
types = ['train']
# hei, width = 800, 1200
for type in types:
	res_path = os.path.join('data/cache/crowd', type)
	image_data = []
	valid_count = 0
	iggt_count = 0
	rea_count = 0
	box_count = 0
	for l in ann.imgs:
		imgname = ann.imgs[l]['file_name']

		# anno = annos[index][0][l]
		# cityname = anno[0][0][0][0].encode()
		# imgname = anno[0][0][1][0].encode()
		# gts = anno[0][0][2]
		img_path = os.path.join(all_img_path, type + '/'+imgname)
		boxes = []
		ig_boxes = []
		vis_boxes = []
		gts = ann.imgToAnns[l]
		if len(gts)==0:
			continue
		for i in range(len(gts)):
			bbox= gts[i]['bbox']
			vbbox = gts[i]['vbbox']
			ignore = gts[i]['ignore']
			if ignore:
				ig_boxes.append(bbox)
			else:
				boxes.append(bbox)
				vis_boxes.append(vbbox)

		boxes = np.array(boxes)
		vis_boxes = np.array(vis_boxes)
		ig_boxes = np.array(ig_boxes)

		im = Image.open(img_path)
		w,h = im.size
		wr = 1200.0 / w
		hr = 800.0 / h
		boxes[:, ::2] = boxes[:, ::2] * wr
		boxes[:, 1::2] = boxes[:, 1::2] * hr
		vis_boxes[:, ::2] = vis_boxes[:, ::2] * wr
		vis_boxes[:, 1::2] = vis_boxes[:, 1::2] * hr
		ig_boxes[:, ::2] = ig_boxes[:, ::2] * wr
		ig_boxes[:, 1::2] = ig_boxes[:, 1::2] * hr
		if len(boxes)==0:
			continue
		valid_count += 1
		annotation = {}
		annotation['filepath'] = img_path
		if len(ig_boxes) > 0 and len(boxes) > 0:
			boxig_overlap = box_op(np.ascontiguousarray(boxes, dtype=np.float),
								   np.ascontiguousarray(ig_boxes, dtype=np.float))
			ignore_sum = np.sum(boxig_overlap, axis=1)
			oriboxes = np.copy(boxes)
			boxes = oriboxes[ignore_sum < 0.5, :]
			vis_boxes = vis_boxes[ignore_sum < 0.5, :]

			if ignore_sum.max()>=0.5:
				iggt_count += len(ignore_sum)-len(boxes)
				ig_boxes = np.concatenate([ig_boxes, oriboxes[ignore_sum >= 0.5, :]], axis=-0)
		box_count += len(boxes)
		annotation['bboxes'] = boxes
		annotation['vis_bboxes'] = vis_boxes
		annotation['ignoreareas'] = ig_boxes
		image_data.append(annotation)
	with open(res_path, 'wb') as fid:
		cPickle.dump(image_data, fid, cPickle.HIGHEST_PROTOCOL)
	print '{} has {} images and {} valid images, {} valid gt and {} ignored gt'.format(type, len(gts), valid_count, box_count, iggt_count)