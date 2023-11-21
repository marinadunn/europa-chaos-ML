from src.model_objects.trainable_maskrcnn import TrainableMaskRCNN

def train_maskrcnn():
    train_obj = TrainableMaskRCNN(2, 2)  # Mask R-CNN version 2; 2 classes
    train_obj.train_model(30)  # train for 30 epochs
    train_obj.save_model()

if __name__ == "__main__":
    # train Mask R-CNN model
    train_maskrcnn()