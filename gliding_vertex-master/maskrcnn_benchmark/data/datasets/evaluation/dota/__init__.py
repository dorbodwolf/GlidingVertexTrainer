'''
Author: your name
Date: 2020-09-03 14:24:44
LastEditTime: 2020-09-14 11:18:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Gliding-vertex-Trainer/gliding_vertex-master/maskrcnn_benchmark/data/datasets/evaluation/dota/__init__.py
'''
from .dota_eval import do_dota_evaluation, do_dota_test
from maskrcnn_benchmark.config import cfg

def dota_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    # 只解译结果，不评估map
    if cfg.TEST.IS_ONLY_INFERENCE:
        return do_dota_test(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
    else:    
        return do_dota_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
