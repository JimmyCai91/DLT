'''
Author: Jinzheng Cai
Email: caijinzhengcn@gmail.com
'''


import os
import json
import math
import numpy as np


def IOU3D(box1, gts, boxlinewidth=1.0):
  # compute overlaps
  ixmin = np.maximum(gts[:, 0], box1[0])
  iymin = np.maximum(gts[:, 1], box1[1])
  izmin = np.maximum(gts[:, 2], box1[2])
  ixmax = np.minimum(gts[:, 3], box1[3])
  iymax = np.minimum(gts[:, 4], box1[4])
  izmax = np.minimum(gts[:, 5], box1[5])
  iw = np.maximum(ixmax - ixmin + boxlinewidth, 0.)
  ih = np.maximum(iymax - iymin + boxlinewidth, 0.)
  id = np.maximum(izmax - izmin + boxlinewidth, 0.)
  inters = iw * ih * id
  # union
  uni = ((box1[3] - box1[0] + boxlinewidth) * (box1[4] - box1[1] + boxlinewidth) * (box1[5] - box1[2] + boxlinewidth) +
         (gts[:, 3] - gts[:, 0] + boxlinewidth) * (gts[:, 4] - gts[:, 1] + boxlinewidth) * (gts[:, 5] - gts[:, 2] + boxlinewidth) - inters)
  overlaps = inters / uni
  return overlaps


class LTEvaluator(object):
  def __init__(self, result_json_file, method='deeds', iou_threshold=0.3, offset_to_init=None, track_conf=None, robust=False, split=-1, 
               do_offset=False, offset_ratio_th=1.0):
    self.incorrect_labeled_lesion_pairs = []
    self.robust = robust
    data = json.load(open(result_json_file))
    if isinstance(data, list):
      self.ret = {'results': data}
    else:
      self.ret = data
    if 'distances' in self.ret.keys():
      self.ret['results'] = self.ret['distances']
    self.offset_to_init = offset_to_init 
    self.track_conf = track_conf
    self.__regulation__()
    self.data_size = len(self.ret['results'])

    #split-wise evaluation
    if split != -1:
      results = self.ret['results']
      step = len(results) // 4
      if split < 3:
        self.ret['results'] = results[split*step:(split+1)*step]
      else:
        self.ret['results'] = results[split*step:]

    #generate results
    if method == 'lt':
      self.__proc_lt__()
    elif method == 'deeds':
      self.__proc_deeds__()
    if do_offset:
      self.__proc_dist__()
    self.do_offset = do_offset
    self.offset_ratio_th = offset_ratio_th
    self.iou_threshold = iou_threshold

  def __regulation__(self):
    if self.offset_to_init is not None:
      replace_counter = 0
      new_results = []
      for ret in self.ret['results']:
        center_pd = ret['predict target center']
        center_init = ret['initial center']
        target_spacing = ret['target spacing']
        offset = ((center_pd[0]-center_init[0]) * target_spacing[0])**2 + \
                 ((center_pd[1]-center_init[1]) * target_spacing[1])**2 + \
                 ((center_pd[2]-center_init[2]) * target_spacing[2])**2
        offset = math.sqrt(offset)
        is_from_registration = 1 if 'from registration' in ret.keys() else ret['from registration']
        if (is_from_registration == 1) and (offset > self.offset_to_init):
          ret['predict target center'] = center_init
          replace_counter += 1
        new_results.append(ret)
      self.ret['results'] = new_results
      print("Offset from init: {} projections have been replaced by initial centers.".format(replace_counter))

    if self.track_conf is not None:
      replace_counter = 0
      new_results = []
      for ret in self.ret['results']:
        is_from_registration = 1 if 'from registration' in ret.keys() else ret['from registration']
        if (is_from_registration == 1) and (ret['predict confidence'] <= self.track_conf):
          ret['predict target center'] = ret['initial center']
          replace_counter += 1
        new_results.append(ret)
      self.ret['results'] = new_results
      print("Track confidence: {} projections have been replaced by initial centers.".format(replace_counter))

  def __proc_dist__(self):
    results = self.ret['results']
    offset_ratios = []
    ground_truth_radius = []
    offset_to_centers = []
    offsets_to_z, offsets_to_x, offsets_to_y = [], [], []
    for ret in results:
      pred_center, target_center = ret['predict target center'], ret['target center']
      target_spacing = ret['target spacing']
      dist_x = ((pred_center[0] - target_center[0])*target_spacing[0])**2
      dist_y = ((pred_center[1] - target_center[1])*target_spacing[1])**2
      dist_z = ((pred_center[2] - target_center[2])*target_spacing[2])**2
      offset_to_center = math.sqrt(dist_x+dist_y+dist_z)
      offsets_to_z.append(math.sqrt(dist_z))
      offsets_to_x.append(math.sqrt(dist_x))
      offsets_to_y.append(math.sqrt(dist_y))

      try:
        target_lesion_diameter = float(max(ret['target recist diameter']) / 2.0)
        target_lesion_diameter *= target_spacing[0]
        offset_ratio = float(offset_to_center) / max(1e-3, float(target_lesion_diameter))
      except:
        offset_ratio = 0.5

      offset_ratios.append(offset_ratio)
      offset_to_centers.append(offset_to_center)
      ground_truth_radius.append(target_lesion_diameter)

    self.offsets_to_z = np.array(offsets_to_z)
    self.offsets_to_x = np.array(offsets_to_x)
    self.offsets_to_y = np.array(offsets_to_y)
    self.offset_ratios = np.array(offset_ratios)
    self.offset_to_centers = np.array(offset_to_centers)
    self.ground_truth_radius = np.array(ground_truth_radius)

  #process results from deeds registration
  def __proc_deeds__(self):
    distances = self.ret['results']
    correct_count = []
    center_distances = []
    proc_times = []
    gt_boxes, pred_boxes = [], []
    instance_pairs = []
    tracking_conf, recognition_conf = [], []
    for dist_id, dist in enumerate(distances):
      if dist_id in self.incorrect_labeled_lesion_pairs:
        continue
      pred_centers, target_box = [dist['predict target center']], dist['target box']
      #distance to the gt center
      target_center = dist['target center']  
      if 'predict target inner points' in dist.keys() and self.robust:
        pred_centers += dist['predict target inner points']
      try:
        target_spacing = dist['target spacing']
      except:
        target_spacing = [1,1,1]

      for pred_center in pred_centers:
        dist_x = ((pred_center[0] - target_center[0])*target_spacing[0])**2
        dist_y = ((pred_center[1] - target_center[1])*target_spacing[1])**2
        dist_z = ((pred_center[2] - target_center[2])*target_spacing[2])**2
        center_distances.append(math.sqrt(dist_x+dist_y+dist_z))
        #lesion tracking accuracty
        in_x = (pred_center[0] >= target_box[0]) and (pred_center[0] <= (target_box[0]+target_box[3]))
        in_y = (pred_center[1] >= target_box[1]) and (pred_center[1] <= (target_box[1]+target_box[4]))
        in_z = (pred_center[2] >= (target_box[2])) and (pred_center[2] <= (target_box[2]+target_box[5]))
        if in_x and in_y and in_z:
          correct_count.append(1)
        else:
          correct_count.append(0)
      instance_pairs.append('{}_{}'.format(dist['source'].replace('.nii.gz', ''), dist['target'].replace('.nii.gz', '')))

      #time for processing
      try:
        proc_times.append(dist['processing time'])
      except:
        proc_times.append(dist['prcessing time'])

      #some confidence scores
      if 'predict confidence' in dist.keys():
        tracking_conf.append(dist['predict confidence'])
      else:
        tracking_conf.append(-1)
      if 'object score' in dist.keys():
        recognition_conf.append(dist['object score'])
      else:
        recognition_conf.append(-1)

      #bounding boxes
      gt_boxes.append([target_box[0], target_box[1], target_box[2],
                       target_box[0] + target_box[3], 
                       target_box[1] + target_box[4], 
                       target_box[2] + target_box[5]])
      if 'predict target box' in dist.keys():
        pred_box = dist['predict target box']
        pred_boxes.append([pred_box[0], pred_box[1], pred_box[2],
                          pred_box[0] + pred_box[3], 
                          pred_box[1] + pred_box[4], 
                          pred_box[2] + pred_box[5]])

    self.correct_count = np.array(correct_count) 
    self.lesion_pair_count = len(distances)
    self.center_distances = center_distances
    self.proc_times = proc_times
    self.gt_boxes = gt_boxes
    self.pred_boxes = pred_boxes
    self.instance_pairs = instance_pairs
    self.tracking_conf = np.array(tracking_conf)
    self.recognition_conf = np.array(recognition_conf)

  #process results from lesion tracker
  def __proc_lt__(self):
    distances = self.ret['results']
    for dist in distances:
      for i in [0,1,2]:
        dist['source center'][i] += dist['source offset'][i]
        dist['target center'][i] += dist['target offset'][i]
        dist['predict target center'][i] += dist['target offset'][i]
        dist['source box'][i] += dist['source offset'][i]
        dist['target box'][i] += dist['target offset'][i]
        if 'predict target box' in dist.keys():
          dist['predict target box'][i] += dist['target offset'][i]
    self.__proc_deeds__()

  #evaluate IOU
  def eval_IOU(self):
    ovrs = [IOU3D(pred, np.array([gt])) for pred, gt in zip(self.pred_boxes, self.gt_boxes)]
    self.overlaps = ovrs

  #report the results
  def report(self):
    #count lesions
    print("{} lesions: {} testing pairs.".format(self.data_size, len(self.correct_count)))
    #tracking accuracy
    acc = float(self.correct_count.sum()) / float(len(self.correct_count))
    self.box_acc = acc
    print("Lesion tracking accuracy {:.4f}.".format(acc))
    #distance histogram to gt centers
    distances = np.array(self.center_distances)
    print("Offsets, mean: {:.4f}, max: {:.4f}, min: {:.4f}, std: {:.4f}.".format(distances.mean(), distances.max(), distances.min(), distances.std()))
    if self.do_offset:
      print("Offsets, mean: {:.4f}, max: {:.4f}, min: {:.4f}, std: {:.4f}.".format(self.offset_to_centers.mean(), self.offset_to_centers.max(), self.offset_to_centers.min(), self.offset_to_centers.std()))
      print("Offsets X, mean: {:.4f}, max: {:.4f}, min: {:.4f}, std: {:.4f}.".format(self.offsets_to_x.mean(), self.offsets_to_x.max(), self.offsets_to_x.min(), self.offsets_to_x.std()))
      print("Offsets Y, mean: {:.4f}, max: {:.4f}, min: {:.4f}, std: {:.4f}.".format(self.offsets_to_y.mean(), self.offsets_to_y.max(), self.offsets_to_y.min(), self.offsets_to_y.std()))
      print("Offsets Z, mean: {:.4f}, max: {:.4f}, min: {:.4f}, std: {:.4f}.".format(self.offsets_to_z.mean(), self.offsets_to_z.max(), self.offsets_to_z.min(), self.offsets_to_z.std()))

    #distance based accuracy
    if self.do_offset:
      if self.offset_ratio_th <= 1.0:
        offset_acc = float((self.offset_ratios < self.offset_ratio_th).sum()) / float(len(self.offset_ratios))
      else:
        # offset_acc = float((self.offset_to_centers < self.offset_ratio_th).sum()) / float(len(self.offset_to_centers))
        offset_thresholds = np.minimum(self.ground_truth_radius, self.offset_ratio_th)
        offset_acc = float((self.offset_to_centers <= offset_thresholds).sum()) / float(len(self.offset_to_centers))
      self.offset_acc = offset_acc
      print("Offset accuracy {:.4f} at threshold {:.2f}.".format(offset_acc, self.offset_ratio_th))

    #average time for processing each case
    proces_times = np.array(self.proc_times)

    # proces_times = proces_times[proces_times > 2.0]
    print("Average processing time: {:.4f} (std {:.4f}) second, measured from {} cases.".format(proces_times.mean(), proces_times.std(), len(proces_times)))

    #accuracy basing on bounding box overlap
    if len(self.pred_boxes) == len(self.gt_boxes):
      self.eval_IOU()
      print("Accuracy @ iou {:.2f} is {:.4f}.".format(self.iou_threshold, (np.array(self.overlaps) >= self.iou_threshold).sum() / self.data_size))


if __name__ == '__main__':
  json_file, offset_to_init, track_conf = './data/DLTMix.json', None, None

  #---- Evaluation Code  
  if json_file is not None:  
    do_offset = True
    offset_accs = []
    offset_ratio_ths = [10,1] if do_offset else [1]
    for offset_ratio_th in offset_ratio_ths:
      print("==="*20)
      print("{}".format(json_file.split('/')[-1]))
      lt_eval = LTEvaluator(json_file, offset_to_init=offset_to_init, track_conf=track_conf, do_offset=do_offset, offset_ratio_th=offset_ratio_th)
      lt_eval.report()
      if do_offset:
        offset_accs.append(lt_eval.offset_acc)
    if do_offset:
      print("+++"*20)
      for acc, th in zip(offset_accs, offset_ratio_ths):
        print('offset threshold and acc: {:.2f}, {:.4f}'.format(th, acc))
      print('mean offset acc: {:.4f}'.format(np.array(offset_accs).mean()))
      for acc in offset_accs:
        print(" & {:.2f}".format(acc*100), end='')
      print("\\\\")
    print("Lesion tracking bounding box accuracy: {:.4f}".format(lt_eval.box_acc))