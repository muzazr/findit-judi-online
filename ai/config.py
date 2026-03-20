class Config:
    model_name = "bert-base-multilingual-uncased"
    max_length = 128
    batch_size = 32
    learning_rate = 2e-5
    num_epochs = 5
    weight_decay = 0.01
    warmup_ratio = 0.1
    gradient_accumulation_steps = 1
    seed = 42

    train_path = "./data/splits/train.csv"
    val_path = "./data/splits/val.csv"
    test_path = "./data/splits/test.csv"
    output_dir = "./model"