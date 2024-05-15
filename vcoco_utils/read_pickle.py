from vcoco_utils.vsrl_eval import VCOCOeval

vsrl_annot_file_s='../data/vcoco/data/vcoco_test.json'
split_file_s='../data/vcoco/data/vcoco_test.ids'

coco_file_s='../data/vcoco/data/instances_vcoco_all_2014.json'
vcocoeval = VCOCOeval(vsrl_annot_file_s, coco_file_s, split_file_s)

file_name= '../logs/log-vcoco.pickle'
vcocoeval._do_eval(file_name, ovr_thresh=0.5)
