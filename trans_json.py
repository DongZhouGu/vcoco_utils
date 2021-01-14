import json
import numpy as np
import pickle


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.uint8):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def dis(A, B):
    distance = np.sqrt(np.sum(np.square(np.array(A) - np.array(B))))
    return distance



def _get_vsrl_data(ix, cls,  valid_action):
    """ Get VSRL data for ann_id."""
    action_id = -np.ones((num_actions), dtype=np.int32)
    role_id = -np.ones((num_actions, 2), dtype=np.int32)
    # check if ann_id in vcoco annotations
    if cls == 1:
        action_id[:] = 0
        role_id[:] = -1
    else:
        return action_id, role_id

    _valid_action = np.reshape(valid_action, (1, -1))
    has_label = np.where(_valid_action[0] == str(ix))[0]
    for aid in has_label:
        aid = int(aid / 3)
        action_name = valid_action[aid][-1]

        # 获取动作名前缀ride_instr --> ride
        if 'on' in action_name:
            ac_name_list = action_name.split('_')
            verb_pre = ac_name_list[0] + '_' + ac_name_list[1] + '_' + ac_name_list[2]
        else:
            verb_pre = action_name.split('_')[0]

        # 获取动作名的role类型，instr / obj
        verb_last = action_name.split('_')[-1]
        # 获取动作名的索引
        i = verb_name[verb_pre]
        action_id[i] = 1
        rids = action_roles[i]
        if len(rids) == 1:
            continue
        elif len(rids) == 2:
            role_id[i, 0] = valid_action[aid][1]
        else:
            if rids[1] == 'obj':
                if verb_last == 'obj':
                    role_id[i, 0] = valid_action[aid][1]
                if verb_last == 'instr':
                    role_id[i, 1] = valid_action[aid][1]

            if rids[1] == 'instr':
                if verb_last == 'obj':
                    role_id[i, 1] = valid_action[aid][1]
                if verb_last == 'instr':
                    role_id[i, 0] = valid_action[aid][1]


    return action_id, role_id


# all_agent = {2: 'cut_instr', 21: 'snowboard_instr', 4: 'cut_obj', 0: 'surf_instr',
#              26: 'skateboard_instr', 7: 'kick_obj', 9: 'eat_obj', 14: 'carry_obj', 15: 'throw_obj',
#              16: 'eat_instr', 17: 'smile', 18: 'look_obj', 19: 'hit_instr', 20: 'hit_obj',
#              1: 'ski_instr', 22: 'run', 10: 'sit_instr', 24: 'read_obj', 5: 'ride_instr', 3: 'walk',
#              23: 'point_instr', 11: 'jump_instr', 8: 'work_on_computer_instr', 25: 'hold_obj',
#              13: 'drink_instr', 12: 'lay_instr', 6: 'talk_on_phone_instr', 27: 'stand',
#              28: 'catch_obj'}
num_actions = 26
verb_name = {'hold': 0, 'stand': 1, 'sit': 2, 'ride': 3, 'walk': 4, 'look': 5, 'hit': 6, 'eat': 7, 'jump': 8,
             'lay': 9, 'talk_on_phone': 10, 'carry': 11, 'throw': 12, 'catch': 13, 'cut': 14, 'run': 15,
             'work_on_computer': 16, 'ski': 17, 'surf': 18, 'skateboard': 19, 'smile': 20, 'drink': 21, 'kick': 22,
             'point': 23, 'read': 24, 'snowboard': 25}
action_roles = [['agent', 'obj'], ['agent'], ['agent', 'instr'], ['agent', 'instr'], ['agent'], ['agent', 'obj'],
                ['agent', 'instr', 'obj'], ['agent', 'obj', 'instr'], ['agent', 'instr'],
                ['agent', 'instr'], ['agent', 'instr'], ['agent', 'obj'], ['agent', 'obj'], ['agent', 'obj'],
                ['agent', 'instr', 'obj'], ['agent'], ['agent', 'instr'], ['agent', 'instr'], ['agent', 'instr'],
                ['agent', 'instr'], ['agent'], ['agent', 'instr'],
                ['agent', 'obj'], ['agent', 'instr'], ['agent', 'obj'], ['agent', 'instr']]

coco_class_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
                   8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter',
                   15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant',
                   23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie',
                   33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
                   39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
                   52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
                   59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
                   67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
                   84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
                   90: 'toothbrush'}
coco_class_dict = {v: k for k, v in coco_class_dict.items()}



res = []
for file_id in range(1, 3):
    annotation_file = '/root/gudongzhou/PPDM0102V2.0/src/gdz_util/new_data/{}.json'.format(file_id)
    anno = json.load(open(annotation_file, 'r'))
    entry = {}
    entry['id'] = anno['imagePath']
    entry['boxes'] = np.empty((0, 4), dtype=np.float32)
    entry['gt_classes'] = np.empty((0), dtype=np.int32)
    entry['gt_actions'] = np.empty((0, num_actions), dtype=np.int32)
    entry['gt_role_id'] = np.empty((0, num_actions, 2), dtype=np.int32)

    valid_objs = []

    for shape in anno['shapes']:
        if shape['shape_type'] == 'rectangle':
            obj_bbox_s = [shape['points'][0][0], shape['points'][0][1], shape['points'][1][0], shape['points'][1][1],
                          shape['label']]
            valid_objs.append(obj_bbox_s)

    valid_action = []
    for shape in anno['shapes']:
        if shape['shape_type'] == 'line':
            person_point = shape['points'][0]
            role_point = shape['points'][1]
            min_person_dis = 1000
            min_role_dis = 1000
            for i, obj in enumerate(valid_objs):
                person_distance = dis(person_point, obj[:2])
                role_distance = dis(role_point, obj[:2])
                if person_distance < min_person_dis:
                    person_id = i
                    min_person_dis = person_distance
                if role_distance < min_role_dis:
                    role_id = i
                    min_role_dis = role_distance

            assert valid_objs[person_id][-1] == 'person', '{}.json文件标注错误'.format(file_id)
            assert valid_objs[role_id][-1] != 'person', '{}.json文件标注错误'.format(file_id)
            valid_action.append([person_id, role_id, shape['label']])

    num_valid_objs = len(valid_objs)
    boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
    gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
    gt_actions = -np.ones((num_valid_objs, num_actions), dtype=entry['gt_actions'].dtype)
    gt_role_id = -np.ones((num_valid_objs, num_actions, 2), dtype=entry['gt_role_id'].dtype)

    for ix, obj in enumerate(valid_objs):
        cls = coco_class_dict[obj[-1]]
        boxes[ix, :] = obj[:4]
        gt_classes[ix] = cls
        gt_actions[ix, :], gt_role_id[ix, :, :] = _get_vsrl_data(ix, cls, valid_action)

    entry['boxes'] = np.append(entry['boxes'], np.array(boxes), axis=0)
    entry['gt_classes'] = np.append(entry['gt_classes'], np.array(gt_classes))
    entry['gt_actions'] = np.append(entry['gt_actions'], np.array(gt_actions), axis=0)
    entry['gt_role_id'] = np.append(entry['gt_role_id'], np.array(gt_role_id), axis=0)

    res.append(entry)

save_path = 'new_dataset_test.json'
with open(save_path, 'w') as outfile:
    outfile.write(json.dumps(res, cls=MyEncoder))
