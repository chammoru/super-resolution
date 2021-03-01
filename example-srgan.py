from pathlib import Path

import matplotlib.pyplot as plt

from data import DIV2K
from model import resolve_single
from model.srgan import generator, discriminator
from train import SrganTrainer, SrganGeneratorTrainer
from utils import load_image

weights_dir = 'weights/srgan'
weights_path = Path(weights_dir)

pre_generator = generator()
gan_generator = generator()

weights_path.mkdir(parents=True, exist_ok=True)


def weights_file(filename):
    return str(weights_path / filename)


def main():
    div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
    div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')

    train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
    valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)

    pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator')
    pre_trainer.train(train_ds,
                      valid_ds.take(10),
                      steps=1000000,
                      evaluate_every=1000,
                      save_best_only=False)

    pre_trainer.model.save_weights(weights_file('pre_generator.h5'))

    gan_generator.load_weights(weights_file('pre_generator.h5'))

    gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
    gan_trainer.train(train_ds, steps=200000)

    gan_trainer.generator.save_weights(weights_file('gan_generator.h5'))
    gan_trainer.discriminator.save_weights(weights_file('gan_discriminator.h5'))


def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)

    pre_sr = resolve_single(pre_generator, lr)
    gan_sr = resolve_single(gan_generator, lr)

    plt.figure(figsize=(20, 20))

    images = [lr, pre_sr, gan_sr]
    titles = ['LR', 'SR (PRE)', 'SR (GAN)']
    positions = [1, 3, 4]

    for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
        plt.subplot(2, 2, pos)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    plt.show()


def demo():
    pre_generator.load_weights(weights_file('pre_generator.h5'))
    gan_generator.load_weights(weights_file('gan_generator.h5'))

    resolve_and_plot('demo/0869x4-crop.png')
    # resolve_and_plot('demo/0829x4-crop.png')
    # resolve_and_plot('demo/0851x4-crop.png')


if __name__ == "__main__":
    main()

    # demo()
