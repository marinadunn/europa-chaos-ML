from src.data_generator import DataGenerator
from src.dataset import EuropaIceBlockDataset, EuropaIceBlockMetaset
from src.info.file_structure import IMG_TEST_PATH, IMG_TRAIN_PATH, LBL_TEST_PATH, LBL_TRAIN_PATH
import torch
from src.utility.custom import get_transform
import src.utility.utils as utils
from src.optuna_utility.models import define_mrcnn_model_01, define_mrcnn_model_02
from src.utility.engine import train_one_epoch
from src.model_objects.rcnn_model import MaskRCNN
from src.optuna_utility.evaluation import MaskRCNNOutputEvaluator

# Objective function for optuna

data_gen = DataGenerator()

def objective(trial):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    crop_size = trial.suggest_categorical("crop_size", [200, 250, 300, 350])
    stride = trial.suggest_categorical("stride", [25, 50, 75, 100])

    data_gen_params = {
        "train_regions": ["Co"],
        "test_regions": ["hh", "ii"],
        "crop_height": crop_size,
        "crop_width": crop_size,
        "stride": stride,
        "min_sheet_area": 50
    }
    data_gen.pruned_sliding_crops_experiment(
        data_gen_params["train_regions"],
        data_gen_params["test_regions"],
        data_gen_params["crop_height"],
        data_gen_params["crop_width"],
        data_gen_params["stride"],
        data_gen_params["min_sheet_area"]
    )

    dataset = EuropaIceBlockDataset('./', IMG_TRAIN_PATH, LBL_TRAIN_PATH,  get_transform(train=False))  
    dataset_test = EuropaIceBlockDataset('./', IMG_TEST_PATH, LBL_TEST_PATH, get_transform(train=False))
    # metaset = EuropaIceBlockMetaset('train', './', img_dir_tr, lbl_dir_tr,  get_transform(train=False))  
    metaset_test = EuropaIceBlockMetaset('test', './', IMG_TEST_PATH, LBL_TEST_PATH, get_transform(train=False))

    bs = trial.suggest_categorical("batch_size", [2, 4, 8]) # 8 default but was getting memory errors
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs, shuffle=False, collate_fn=utils.collate_fn)


    num_classes = 2
    model = define_mrcnn_model_01(trial, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    optimizer = torch.optim.SGD(params, lr=learning_rate)

    #How is LR schedularl working? Gamma value??  what the change is??
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000) #remove lr step, is 3000 step size insane?

    num_epochs = 100
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=(num_epochs+10))
        #lr_scheduler.step()

    # EVALUATE
    # Transfer model to wrapper object
    mask_model = MaskRCNN()
    evaluator = MaskRCNNOutputEvaluator()
    mask_model.set_model(model)
    true_segs, pred_segs = mask_model.calc_cummulative_threshold_matrix(metaset_test)
    iou_score = evaluator.calc_cummulative_threshold_iou(true_segs, pred_segs)
    return iou_score

def avg_f1_objective(trial):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    crop_size = trial.suggest_categorical("crop_size", [200]) # 256
    stride = trial.suggest_categorical("stride", [25])

    data_gen_params = {
        "train_regions": ["A", "aa", "B", "bb", "C", "Co", "D", "dd", "E", "ee", "F", "ff", "G", "gg", "H", "I", "jj", "kk"],
        "test_regions": ["hh", "ii"],
        "crop_height": crop_size,
        "crop_width": crop_size,
        "stride": stride,
        "min_sheet_area": 50
    }
    data_gen.sliding_crops_experiment(
        data_gen_params["train_regions"],
        data_gen_params["test_regions"],
        data_gen_params["crop_height"],
        data_gen_params["crop_width"],
        data_gen_params["stride"],
        data_gen_params["min_sheet_area"]
    )

    dataset = EuropaIceBlockDataset('./', IMG_TRAIN_PATH, LBL_TRAIN_PATH,  get_transform(train=False))  
    dataset_test = EuropaIceBlockDataset('./', IMG_TEST_PATH, LBL_TEST_PATH, get_transform(train=False))
    # metaset = EuropaIceBlockMetaset('train', './', img_dir_tr, lbl_dir_tr,  get_transform(train=False))  
    metaset_test = EuropaIceBlockMetaset('test', './', IMG_TEST_PATH, LBL_TEST_PATH, get_transform(train=False))

    bs = trial.suggest_categorical("batch_size", [2]) # 8 default but was getting memory errors
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs, shuffle=False, collate_fn=utils.collate_fn)


    num_classes = 2
    model = define_mrcnn_model_02(trial, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    learning_rate = 0.006174236796209435
    optimizer = torch.optim.SGD(params, lr=learning_rate)

    #How is LR schedularl working? Gamma value??  what the change is??
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000) #remove lr step, is 3000 step size insane?

    num_epochs = trial.suggest_int('num_epochs', 7, 13)
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=(500))
        #lr_scheduler.step()

    # EVALUATE
    # Transfer model to wrapper object
    mask_model = MaskRCNN()
    evaluator = MaskRCNNOutputEvaluator()
    mask_model.set_model(model)
    dataset_avg_f1_scores = []
    thresh_sweeps = mask_model.get_dataset_thresh_sweeps(dataset_test, 20)
    for thresh_sweep_pair in thresh_sweeps:
        thresh_sweep = thresh_sweep_pair[0]
        true_label = thresh_sweep_pair[1]
        avg_f1,_ = evaluator.calc_min_iou_avg_f1(true_label, thresh_sweep, min_iou=0.3)
        dataset_avg_f1_scores.append(avg_f1)
    dataset_avg_f1 = 0
    if len(dataset_avg_f1_scores) != 0:
        dataset_avg_f1 = sum(dataset_avg_f1_scores)/len(dataset_avg_f1_scores)
    return dataset_avg_f1

def avg_recall_objective(trial):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    crop_size = trial.suggest_categorical("crop_size", [200]) # 256
    stride = trial.suggest_categorical("stride", [32])

    data_gen_params = {
        "train_regions": ["A", "aa", "B", "bb", "C", "Co", "D", "dd", "E", "ee", "F", "ff", "G", "gg", "H", "I", "jj", "kk"],
        "test_regions": ["hh", "ii"],
        "crop_height": crop_size,
        "crop_width": crop_size,
        "stride": stride,
        "min_sheet_area": 200
    }
    data_gen.sliding_crops_experiment(
        data_gen_params["train_regions"],
        data_gen_params["test_regions"],
        data_gen_params["crop_height"],
        data_gen_params["crop_width"],
        data_gen_params["stride"],
        data_gen_params["min_sheet_area"]
    )

    dataset = EuropaIceBlockDataset('./', IMG_TRAIN_PATH, LBL_TRAIN_PATH,  get_transform(train=False))  
    dataset_test = EuropaIceBlockDataset('./', IMG_TEST_PATH, LBL_TEST_PATH, get_transform(train=False))
    # metaset = EuropaIceBlockMetaset('train', './', img_dir_tr, lbl_dir_tr,  get_transform(train=False))  
    metaset_test = EuropaIceBlockMetaset('test', './', IMG_TEST_PATH, LBL_TEST_PATH, get_transform(train=False))

    bs = trial.suggest_categorical("batch_size", [2]) # 8 default but was getting memory errors
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs, shuffle=False, collate_fn=utils.collate_fn)


    num_classes = 2
    model = define_mrcnn_model_02(trial, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.001, 0.0001])
    # learning_rate = 0.001
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    #How is LR schedularl working? Gamma value??  what the change is??
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000) #remove lr step, is 3000 step size insane?

    num_epochs = trial.suggest_int('num_epochs', 7, 14)
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=(500))
        #lr_scheduler.step()

    # EVALUATE
    # Transfer model to wrapper object
    mask_model = MaskRCNN()
    evaluator = MaskRCNNOutputEvaluator()
    mask_model.set_model(model)
    dataset_avg_recall_scores = []
    thresh_sweeps = mask_model.get_dataset_thresh_sweeps(dataset_test, 20)
    for thresh_sweep_pair in thresh_sweeps:
        thresh_sweep = thresh_sweep_pair[0]
        true_label = thresh_sweep_pair[1]
        avg_recall,_ = evaluator.calc_min_iou_avg_recall(true_label, thresh_sweep, min_iou=0.3)
        dataset_avg_recall_scores.append(avg_recall)
    dataset_avg_recall = 0
    if len(dataset_avg_recall_scores) != 0:
        dataset_avg_recall = sum(dataset_avg_recall_scores)/len(dataset_avg_recall_scores)
    return dataset_avg_recall