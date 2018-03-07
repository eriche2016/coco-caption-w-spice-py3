from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import argparse

from IPython.core.debugger import Tracer
debug_here = Tracer() 

def get_parser():

    parser = argparse.ArgumentParser(description='Eval caption',epilog='The end.')
    parser.add_argument('-results_file', dest='results_file',
                        default = './results/captions_val2014_fakecap_results.json',
                        help='Base name to save model')
    parser.add_argument('-ann_file', dest='ann_file',
                        default = './annotations/captions_val2014.json',
                        help='path to model file to load.')
    return parser

parser = get_parser()
args_dict = parser.parse_args()

ann_file = args_dict.ann_file

#results_file = os.path.join(args_dict.data_folder, 'results',
#                          args_dict.model_name +'_gencaps.json')
results_file = args_dict.results_file
print("Evaluating %s"%(args_dict.results_file)) 
coco = COCO(ann_file)
cocoRes = coco.loadRes(results_file)
cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.params['image_id'] = cocoRes.getImgIds()

cocoEval.evaluate()

# print output evaluation scores
for metric, score in cocoEval.eval.items():
    print ('%s: %.3f'%(metric, score))