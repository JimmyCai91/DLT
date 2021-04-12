'''
Author: Jinzheng Cai
Email: caijinzhengcn@gmail.com
'''
import os
import json
import math
import numpy as np

class LTEvaluator(object):
  '''
  Lesion-tracking evaluator
  '''
  def __init__(self, result_json_file, offset_ratio_th=1.0):
    self.ret = json.load(open(result_json_file))
    #-- preprocess lesion tracking results
    self.__load_scores__()
    self.__load_distances__()
    self.data_size = len(self.ret['results'])
    self.offset_ratio_th = offset_ratio_th

  #process results from deeds registration
  def __load_scores__(self):
    distances = self.ret['results']
    correct_count = []
    center_distances = []
    proc_times = []
    instance_pairs = []
    tracking_conf, recognition_conf = [], []
    for dist in distances:
      instance_pairs.append('{}_{}'.format(dist['source'].replace('.nii.gz', ''), dist['target'].replace('.nii.gz', '')))
      proc_times.append(dist['processing time'])
      tracking_conf.append(dist['predict confidence'])
      recognition_conf.append(dist['object score'])
    self.correct_count = np.array(correct_count) 
    self.lesion_pair_count = len(distances)
    self.center_distances = center_distances
    self.proc_times = proc_times
    self.instance_pairs = instance_pairs
    self.tracking_conf = np.array(tracking_conf)
    self.recognition_conf = np.array(recognition_conf)

  def __load_distances__(self):
    results = self.ret['results']
    offset_ratios = []
    ground_truth_radius = []
    offset_to_centers = []
    offsets_to_z, offsets_to_x, offsets_to_y = [], [], []
    for ret in results:
      pred_center, target_center = ret['predict target center'], ret['target center']
      target_spacing = ret['target spacing']
      #-- offsets in x, y, and z directions
      dist_x = ((pred_center[0] - target_center[0])*target_spacing[0])**2
      dist_y = ((pred_center[1] - target_center[1])*target_spacing[1])**2
      dist_z = ((pred_center[2] - target_center[2])*target_spacing[2])**2
      offsets_to_z.append(math.sqrt(dist_z))
      offsets_to_x.append(math.sqrt(dist_x))
      offsets_to_y.append(math.sqrt(dist_y))
      #-- offsets to lesion center
      offset_to_center = math.sqrt(dist_x+dist_y+dist_z)
      offset_to_centers.append(offset_to_center)
      #-- normalize offsets with target lesion radius
      target_lesion_radius = float(max(ret['target recist diameter']) / 2.0)
      target_lesion_radius *= target_spacing[0]
      ground_truth_radius.append(target_lesion_radius)
      offset_ratio = float(offset_to_center) / max(1e-3, float(target_lesion_radius))
      offset_ratios.append(offset_ratio)
    self.offsets_to_z = np.array(offsets_to_z)
    self.offsets_to_x = np.array(offsets_to_x)
    self.offsets_to_y = np.array(offsets_to_y)
    self.offset_ratios = np.array(offset_ratios)
    self.offset_to_centers = np.array(offset_to_centers)
    self.ground_truth_radius = np.array(ground_truth_radius)

  #report the results
  def report(self):
    distances = self.offset_to_centers
    print("Offsets, \tmean: {:.4f}, max: {:.4f}, min: {:.4f}, std: {:.4f}.".format(distances.mean(), distances.max(), distances.min(), distances.std()))
    print("Offsets X, \tmean: {:.4f}, max: {:.4f}, min: {:.4f}, std: {:.4f}.".format(self.offsets_to_x.mean(), self.offsets_to_x.max(), self.offsets_to_x.min(), self.offsets_to_x.std()))
    print("Offsets Y, \tmean: {:.4f}, max: {:.4f}, min: {:.4f}, std: {:.4f}.".format(self.offsets_to_y.mean(), self.offsets_to_y.max(), self.offsets_to_y.min(), self.offsets_to_y.std()))
    print("Offsets Z, \tmean: {:.4f}, max: {:.4f}, min: {:.4f}, std: {:.4f}.".format(self.offsets_to_z.mean(), self.offsets_to_z.max(), self.offsets_to_z.min(), self.offsets_to_z.std()))

    # Center Point Matching Accuracy
    if self.offset_ratio_th <= 1.0:
      #--- CPM@Radius
      offset_acc = float((self.offset_ratios < self.offset_ratio_th).sum()) / float(len(self.offset_ratios))
      print("Offset accuracy {:.4f} at threshold equals to lesion radius.".format(offset_acc))
    else:
      #--- CPM@10mm, i.e., self.offset_ratio_th is 10.
      offset_thresholds = np.minimum(self.ground_truth_radius, self.offset_ratio_th) #--- adjustable threshold  
      offset_acc = float((self.offset_to_centers <= offset_thresholds).sum()) / float(len(self.offset_to_centers))
      print("Offset accuracy {:.4f} at threshold equals to min(lesion radius, {:.2f}mm).".format(offset_acc, self.offset_ratio_th))
    self.offset_acc = offset_acc

    #average time for processing each case
    process_times = np.array(self.proc_times)
    print("Average processing time: {:.4f} (std {:.4f}) second, measured from {} cases.".format(process_times.mean(), process_times.std(), len(process_times)))


if __name__ == '__main__':
  tracking_results = './data/DLTMix.json'

  #--- Evaluation, CPM@Radius (offset_ratio_th=1.0)
  print("="*34 + 'CPM@Raidus' + "="*34)
  lt_eval = LTEvaluator(tracking_results, offset_ratio_th=1.0)
  lt_eval.report()

  #---Evaluation, CPM@10mm (offset_ratio_th=10.0)
  print("="*35 + 'CPM@10mm' + "="*35)
  lt_eval = LTEvaluator(tracking_results, offset_ratio_th=10.)
  lt_eval.report()
