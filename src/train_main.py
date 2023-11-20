from model_objects.trainable_maskrcnn import TrainableMaskRCNN

def train_maskrcnn():
    train_obj = TrainableMaskRCNN(2, 2)
    train_obj.train_model(30)
    train_obj.save_model()

if __name__ == "__main__":
    train_maskrcnn()