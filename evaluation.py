from skimage.measure import label
import time

from test.utils.evaluation_helper import evaluation_func, get_visually_evaluation
from test.utils.utils import read_mrc,write_mrc

def evaluation(path_pred,path_target,output_path='output/',
               instance_labelled=True,plot=True,axis=230,write=True):
    """
    This function receives predicted particle file and instance labelled groundtruth to do the evaluation.
    In more detail, particles in prediction and groundtruth are divided into 4 categories: 
        1. True Positive: TP, correctly segmented particles.
        2. Merged particles: wrongly segmented particles - several particles in groundtruth are considered as 1 object in prediction. 
        3. False Positive: FP, no corresponding particles in the groundtruth.
        4. False Negative:FN, missed particles in the groundtruth.
    
    This function could prints following quantitative evaluation results:
        1. Precision (TP / particles in prediction)
        2. Recall (TP / particles in groundtruth)
        3. Merged rate (merged particle number / particles in groundtruth): since the center coordinate of each individual particle is also important in this project, this metrics is also included to check whether adjacent particles could be separated appropriately.
    
    This function also gives options to plot(saved) one 2d slice of the visually evaluation results or to save the whole 3d visually evaluation volume into mrc file. 
    
    Input
    ----------
        path_pred: string
            The path of the final particle detection results.
        path_target: string
            The path of the groundtruth.
        output_path: string - default 'output/'
            The path to save the plots or the volume.
        instance_labelled: bool - default True
            Whether the input groundtruth is instance labelled.
        plot: bool - default True
            Whether user wants to save one 2d slice of the visually evaluation results.
        axis: int - default 0.1
            Required when plot is True. The x axis slice we want to see in the evaluated result, e.g. 230 means it plots and saves evaluated_seg[230,:,:] and evaluated_gt[230,:,:]. 
        write: bool - default True
            Whether user wants to save the whole 3d visually evaluation results.
        
    Returns
    -------
    """
    pred,header = read_mrc(path_pred)
    gt,_ = read_mrc(path_target)

    if instance_labelled == False:
        gt = label(gt)

    TP_particle, merged_particle, TP_particle_gt, merged_particle_gt = evaluation_func(pred, gt)
    evaluated_tomo, evaluated_gt = get_visually_evaluation(pred,gt,
                                                          TP_particle, merged_particle,
                                                          TP_particle_gt, merged_particle_gt,
                                                          plot=plot,save_path=output_path,axis=axis)
    
    if write == True:
        write_mrc(evaluated_tomo,output_path+'evaluated_tomo.mrc',header_dict=header)
        write_mrc(evaluated_gt,output_path+'evaluated_gt.mrc',header_dict=header)

if __name__=='__main__':
    '''
    It takes around 40 mins to run.
    Following information should be given:
    
    path_pred: string
        Path of the instance labelled results.
    path_target: string
        Path of the instance labelled groundtruth. 
        Otherwise you could also set instance_labelled=False in the evaluation function.
    output_path: sting
        Path to save the visually evaluated result. 
    '''
    path_pred = '/home/haicu/ruolin.shen/projects/3dpd/output/pred_tomo.mrc'
    path_target = '/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach_back/labelmap1.mrc'
    output_path = 'output/'
    
    start = time.time()
    evaluation(path_pred,path_target,output_path=output_path)
    end = time.time()
    print("Model took %0.2f seconds to post process" % (end - start))
