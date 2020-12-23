import data
import vis
import models
import torch

import matplotlib.pyplot as plt

if __name__ == '__main__':
    TRAIN, VAL = data.load_data()

    # task = data.TRAIN['db3e9e38']['train']
    # sample = task[0]

    # vis.plot_sample(sample)

    dl = TRAIN(bs=3, shuffle=False)
    it = iter(dl)
    batch = next(it)
    X, y = next(it)

    saved_model_path = '.models/soft_addressable_cnn.weights'

    model = models.SoftAddressableComputationCNN(input_channels=11)
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
        dataloader=TRAIN(bs=1, shuffle=False),
    )

    # model.save()
    # model = torch.load(f'{saved_model_path}_whole.h5')

    train_score = models.evaluate(model, TRAIN(bs=1, shuffle=False))
    val_score = models.evaluate(model, VAL(bs=1, shuffle=False))

    print('FINAL TRAIN SCORE:', train_score)
    print('FINAL VAL SCORE:', val_score)
