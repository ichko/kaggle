import data
import vis
import models
import torch
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print('DEVICE:', DEVICE)

    train_dl = data.load_data(
        '.data/training',
        bs=16,
        shuffle=False,
        device=DEVICE,
    )

    val_dl = data.load_data(
        '.data/evaluation',
        bs=4,
        shuffle=False,
        device=DEVICE,
    )

    # task = data.TRAIN['db3e9e38']['train']
    # sample = task[0]

    # vis.plot_sample(sample)

    it = iter(train_dl)
    X, y = next(it)

    saved_model_path = '.models/soft_addressable_cnn.weights'

    model = models.SoftAddressableComputationCNN(input_channels=11)
    model = model.to(DEVICE)
    model.make_persisted(saved_model_path)
    # model = torch.load(f'{saved_model_path}_whole.h5')

    model.summary()

    # score = models.evaluate(
    #     model=model,
    #     dataloader=TRAIN(bs=1, shuffle=False),
    # )
    # print('SCORE', score)  # less is better

    output = model(X)
    print(output.shape)

    model.configure_optim(lr=0.0001)
    models.train(
        epochs=10,
        model=model,
        dataloader=train_dl,
    )

    # model.save()
    # model = torch.load(f'{saved_model_path}_whole.h5')

    train_score = models.evaluate(model, train_dl)
    val_score = models.evaluate(model, val_dl)

    print('FINAL TRAIN SCORE:', train_score)
    print('FINAL VAL SCORE:', val_score)
