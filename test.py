
from new_vcoco_eval import VCOCOeval

vsrl_annot_file = '/root/gudongzhou/PPDM0102V2.0/src/gdz_util/data/vcoco/vcoco_test.json'
coco_annot_file = '/root/gudongzhou/PPDM0102V2.0/src/gdz_util/anno/instances_vcoco_all_2014.json'
split_file = '/root/gudongzhou/PPDM0102V2.0/src/gdz_util/data/vcoco/vcoco_test.ids'
vcocodb = '/root/gudongzhou/PPDM0102V2.0/src/gdz_util/new_data/new_dataset_test.json'


test_path='/root/gudongzhou/PPDM0102V2.0/src/gdz_util/new_data/111.json'
vcocoeval = VCOCOeval(vcocodb, vsrl_annot_file, coco_annot_file, split_file)
vcocoeval._do_eval(test_path, ovr_thresh=0.5)
