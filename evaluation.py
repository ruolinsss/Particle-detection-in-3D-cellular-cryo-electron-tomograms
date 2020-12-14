from utils.evaluation_helper import evaluation, get_visually_evaluation
from utils.utils import read_mrc,write_mrc

from skimage.measure import label
import time

path_pred = '/home/haicu/ruolin.shen/projects/train/tomo17_processed_d.mrc'
path_target = '/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach_back/labelmap1.mrc'

pred,header = read_mrc(path_pred)
gt,_ = read_mrc(path_target)
pred_path = 'output/'
instance_labelled = True

if instance_labelled == False:
    gt = label(gt)

start = time.time()
TP_particle, merged_particle, TP_particle_gt, merged_particle_gt = evaluation(pred, gt)
end = time.time()
print("evaluation function takes %0.2f seconds to run" % (end - start))

start = time.time()
evaluated_tomo, evaluated_gt = get_visually_evaluation(pred,gt,
                                                      TP_particle, merged_particle,
                                                      TP_particle_gt, merged_particle_gt,
                                                      plot=True,save_path=pred_path,axis=230)
end = time.time()
print("get_visually_evaluation function takes %0.2f seconds to run" % (end - start))

write_mrc(evaluated_tomo,pred_path+'evaluated_tomo.mrc',header_dict=header)
write_mrc(evaluated_gt,pred_path+'evaluated_gt.mrc',header_dict=header)